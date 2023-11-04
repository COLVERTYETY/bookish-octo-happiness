import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tqdm
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from torch.ao.quantization import QuantStub, DeQuantStub
from sklearn.metrics import roc_curve, auc
from ptflops import get_model_complexity_info

torch.backends.cudnn.benchmark = True

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

def myNorm(x):
    x = x.permute(0, 2, 1)
    return (x>0.5).float() -0.5

def random_noise(x):
    noise = torch.randn_like(x) * 0.1
    x = torch.clip(x + noise, 0, 1)
    return x

train_transform = transforms.Compose([
    transforms.ToTensor(),
    # random_noise,
    transforms.RandomPerspective(),
    transforms.RandomAffine(45, translate=(0.1, 0.3), scale=(0.8,1.5) ,shear=(0.1,0.3)),

    transforms.RandomAdjustSharpness(0.5),
    transforms.RandomAdjustSharpness(1.5),

    transforms.RandomInvert(),
    myNorm,
])

cutmix = v2.CutMix(num_classes=47)
mixup = v2.MixUp(num_classes=47)
random_erase = v2.RandomErasing(ratio = (0.1, 0.1), value = 'random' )
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup, random_erase])


def collate(batch):
    return cutmix_or_mixup(*default_collate(batch))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomInvert(),
    myNorm,
])

train_set = torchvision.datasets.EMNIST( root='./data', split='balanced', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.EMNIST( root='./data', split='balanced', train=False, download=True, transform=train_transform)
confidence_set = torchvision.datasets.FashionMNIST( root='./data', train=True, download=True, transform=test_transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
confidence_loader = DataLoader(confidence_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

# count the number of classes
num_classes = 47
# for _, label in tqdm.tqdm(train_set, desc='Counting classes'):
#     if label > num_classes:
#         num_classes = label
# num_classes += 1

# model

class CNBLock(nn.Module):
    def __init__(self, dims, norm_shape, kernel_size=3, exp=4, norm = True, residual=False):
        super(CNBLock, self).__init__()
        self.residual = residual
        self.dwconv = nn.Conv2d(dims, dims, kernel_size=kernel_size, padding=kernel_size//2, groups=dims)
        self.norm = norm
        self.ln = nn.LayerNorm(norm_shape)
        self.expand = nn.Conv2d(dims, dims*exp, kernel_size=1)
        self.act = nn.LeakyReLU()
        self.reduce = nn.Conv2d(dims*exp, dims, kernel_size=1)

    def forward(self, x):
        input_ = x
        x = self.dwconv(x)
        if self.norm:
            x = self.ln(x)
        x = self.expand(x)
        x = self.act(x)
        x = self.reduce(x)
        if self.residual: # not supported by quantization
            x = x + input_
        return x
    
class ReduceBlock(nn.Module):
    def __init__(self, in_dims, out_dims, norm_shape, norm=True):
        super(ReduceBlock, self).__init__()
        self.norm = norm
        self.ln = nn.LayerNorm(norm_shape)
        self.conv = nn.Conv2d(in_dims, out_dims, kernel_size=2, padding=0, stride=2)
    
    def forward(self, x):
        if self.norm:
            x = self.ln(x)
        x = self.conv(x)
        return x
    
class StemBlock(nn.Module):
    def __init__(self,in_dims, out_dims, ksize=3, norm=True):
        super(StemBlock, self).__init__()
        self.norm = norm
        self.mid_size = out_dims
        self.conv = nn.Conv2d(in_dims, self.mid_size, kernel_size=ksize, padding=ksize//2)
        self.ln = nn.LayerNorm([self.mid_size, 28, 28])
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.ln(x)
        return x

class Model(nn.Module):
    def __init__(self, num_classes, ksize=3, exp=2):
        super(Model, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.num_classes = num_classes
        self.ksize = ksize
        self.exp = exp
        self.layers = nn.Sequential(
        StemBlock(1, 32, ksize=5),
        nn.LeakyReLU(),

        ReduceBlock(32, 64, [32, 28, 28]),
        CNBLock(64, [64, 14, 14], ksize, exp),

        ReduceBlock(64, 128, [64, 14, 14]), 
        CNBLock(128, [128, 7, 7], ksize, exp),
        CNBLock(128, [128, 7, 7], ksize, exp),
        CNBLock(128, [128, 7, 7], ksize, exp),

        nn.Conv2d(128, 512, kernel_size=7, groups=128),
        nn.LeakyReLU(),
        nn.Flatten(),
        nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x

class simple_cnn(nn.Module):
    def __init__(self, num_classes):
        super(simple_cnn, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            StemBlock(1, 32, ksize=5, norm=False),
            CNBLock(32, [32, 28, 28], 3, exp=1, norm=False),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups = 32),
            nn.ELU(),
            CNBLock(64, [32, 14, 14], 3, exp=1, norm=False),
            nn.Dropout2d(0.1),
            CNBLock(64, [32, 14, 14], 3, exp=1, norm=False),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups = 64),
            nn.ELU(),
            CNBLock(128, [128, 7, 7],3, exp=1, norm=False),
            nn.Dropout2d(0.1),
            CNBLock(128, [128, 7, 7],3, exp=1, norm=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 512, kernel_size=7, groups=128),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        return self.layers(x)

            
model = Model(num_classes, ksize=3, exp=1) # 9.24 MMac 344.88 k
# model = simple_cnn(num_classes) # 9.88 MMac 267.12 k

summary(model, (1, 28, 28), device='cpu')

# training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# use onecyle lr scheduler
epochs = 100
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
writer = SummaryWriter()

def train(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        if labels.ndim >1: # for cutmix mixed labels
            labels = labels.argmax(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return running_loss/len(train_loader), correct/total

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm.tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
    
    return running_loss/len(test_loader), correct/total

def compute_roc(model, test_loader, confidence_loader, device):
    model.eval()
    
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        # First, get outputs for positive samples
        for images, labels in tqdm.tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = torch.softmax(model(images), dim=1)  # Apply softmax
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Next, get outputs for negative samples (from the confidence_loader)
        for images, _ in tqdm.tqdm(confidence_loader):
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)  # Apply softmax
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend([-1 for _ in range(images.size(0))])
    
    return all_labels, all_outputs

def compute_thresholds(all_labels, all_outputs, num_classes):
    thresholds = []
    for i in range(num_classes):
        labels_i = [1 if label == i else 0 for label in all_labels]
        outputs_i = [output[i] for output in all_outputs]
        fpr, tpr, threshold = roc_curve(labels_i, outputs_i)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        thresholds.append(optimal_threshold)
        print(f'Class {i} threshold: {optimal_threshold}')
    return thresholds

def test_thresholds(model, test_loader, thresholds, device):
    model.eval()
    correct = 0
    total = 0
    thresholds = torch.tensor(thresholds).to(device)
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.softmax(model(images), dim=1)  # Apply softmax
            prediction = (outputs - thresholds.unsqueeze(0)).argmax(1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    return correct / total

def plot_confusion_matrix(y_true, y_pred, classes,
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix, using the labels instead of idnexes.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[np.unique(y_true)]
        classes = [ LABELS[i] for i in classes]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
            # print(cm)
        else:
            print('Confusion matrix, without normalization')
            # print(cm)
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                # ... and label them with the respective list entries
                xticklabels=classes, yticklabels=classes,
                title=title,
                ylabel='True label',
                xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        # print(cm.shape)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j]*100, fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig('confusion_matrix.jpg')
        # plt.show()
        return ax

def compute_confusion_matrix(model, test_loader, device):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_predictions = all_outputs.argmax(1)
    plot_confusion_matrix(all_labels, all_predictions, np.arange(num_classes), normalize=True)

def to_onnx_web(model, acc, test_loader=None):
    model_ = model
    model_.eval()
    model_= model_.cpu()
    if test_loader!=None:
        # quantize with torch before export.
        backend = 'qnnpack' # or 'x86'
        model_.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        torch.quantization.prepare(model_, inplace=True)
        acc=0
        with torch.no_grad():
            for images, labels in tqdm.tqdm(test_loader):
                images = images.cpu()
                labels = labels.cpu()
                outputs = model_(images)
                _, predicted = torch.max(outputs.data, 1)
                acc += (predicted == labels).sum().item()
        print('Accuracy before quantization: ', acc/len(test_loader.dataset))
        torch.quantization.convert(model_, inplace=True)
        print(model_)
        with torch.no_grad():
            acc=0
            for images, labels in tqdm.tqdm(test_loader):
                images = images
                labels = labels
                outputs = model_(images)
                _, predicted = torch.max(outputs.data, 1)
                acc += (predicted == labels).sum().item()
        acc = int(acc/len(test_loader.dataset)*10000)
        print('Accuracy after quantization: ',acc)
    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    torch.onnx.export(model_, dummy_input, f"emnist_{acc}.onnx", verbose=False, input_names=['input'], output_names=['output'], opset_version=13, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f'Exported to onnx !! emnist_{acc}.onnx')


def compute_flops(model):
    macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def main():
    # to_onnx_web(model, 0, test_loader)
    compute_flops(model)
    model.cuda()
    best_loss = 1e10
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, scheduler, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f} lr: {scheduler.get_last_lr()[0]:.6f}')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'emnist_best.pth')
            print('Saved best model !!')
    model.load_state_dict(torch.load('emnist_best.pth'))
    compute_confusion_matrix(model, test_loader, device)
    all_labels, all_outputs = compute_roc(model, test_loader, confidence_loader, device)
    thresholds = compute_thresholds(all_labels, all_outputs, num_classes)
    # save thresholds
    print('avg threshold: ', np.mean(thresholds))
    print('std threshold: ', np.std(thresholds))
    print('max threshold: ', np.max(thresholds))
    print('min threshold: ', np.min(thresholds))
    print('Saving thresholds...')
    # test with thresholds
    print('Testing with thresholds...')
    test_acc = test_thresholds(model, test_loader, thresholds, device)
    acc = int(test_acc*10000)
    np.save(f'thresholds_{acc}.npy', thresholds)
    print('Test accuracy with thresholds: ', test_acc)
    writer.add_scalar('Accuracy/test_thresholds', test_acc, epoch)
    writer.flush()
    writer.close()
    # torch.save(model.state_dict(), 'emnist.pth')
    to_onnx_web(model, acc, test_loader)

if __name__ == '__main__':
    main()
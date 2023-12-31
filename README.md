# Optical Character Recognition with onnx web

This is a simple demo of how to perform OCR on the client side using ONNX web.

[try it out !!](https://colvertyety.github.io/bookish-octo-happiness/)

![demo](./demo.gif)

# Algorithm Overview:

1. train a small CNN with PyTorch on the EMNIST dataset.

2. Find optimal threshold for class confidence using AUC-ROC with FASHIONMNIST as negative class.

3. quantize the trained model to 8-bit integer.

4. convert the trained model to ONNX format.

5. load the ONNX model in a web browser and perform inference on the client side:
   1. Binarize the image.
   2. perform clustering via flood fill algorithm.
   3. extract the bounding boxes of the clusters.
   4. resize and perform inference on the bounding boxes.
   5. compare prediction with confidence threshold.
   6. profit.

# model performance:

![confusion matrix](./confusion_matrix.jpg)

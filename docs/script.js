const video = document.getElementById('webcam');
const switchCamBtn = document.getElementById('switchCam');
const predictionText = document.getElementById('prediction');
let currentStream;
let model;

const index_to_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'];

const thresholds = [9.773934364318848, 8.293187141418457, 9.934375762939453, 9.294990539550781, 7.936954498291016, 9.735556602478027, 9.542527198791504, 8.289621353149414, 7.873318195343018, 9.031131744384766, 7.869201183319092, 9.827280044555664, 10.279926300048828, 9.418956756591797, 10.042513847351074, 9.778995513916016, 10.02375602722168, 7.918678283691406, 8.33338451385498, 9.777628898620605, 10.83350658416748, 9.13015365600586, 10.393794059753418, 9.665188789367676, 9.221426010131836, 7.717484474182129, 9.948719024658203, 8.8203706741333, 9.75184154510498, 7.705641269683838, 7.945072174072266, 8.790163040161133, 14.034773826599121, 7.585292816162109, 9.35817813873291, 9.742826461791992, 8.852142333984375, 8.373862266540527, 11.218235969543457, 8.813253402709961, 7.479157447814941, 9.36019515991211, 8.186284065246582, 9.505398750305176, 8.002181053161621, 7.964801788330078, 8.3523588180542];

// Initialize ONNX model
async function initModel() {
    // model = new ort.InferenceSession({ backendHint: 'webgl' });
    model = await ort.InferenceSession.create("./emnist_8693.onnx");
    // await model.loadModel('./emnist_8693.onnx');
}

var image_list = [];

var tensors = [];
//  mouse info
var mouseX=0;
var mouseY=0;
let lastClickTime = Date.now();

// Update mouse info on click
video.addEventListener('click', function(event) {
    const rect = video.getBoundingClientRect();
    mouseX = event.clientX - rect.left;
    mouseY = event.clientY - rect.top;
    lastClickTime = Date.now();
    console.log(mouseX, mouseY, lastClickTime);
});

// Get webcam access
async function getWebcam(streamName = 'user') {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            facingMode: streamName
        }
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;

    video.onloadedmetadata = function() {
        video.play();
        inferenceLoop();    
    };
}

function drawBoundingBoxes(clusters, predictions, k) {

    // Ensure the overlayCanvas matches the video dimensions
    const overlayCanvas = document.getElementById('overlayCanvas');
    overlayCanvas.width = video.videoWidth;
    overlayCanvas.height = video.videoHeight;

    const ctx = overlayCanvas.getContext('2d');

    // Draw the current frame from the video element onto the canvas
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Loop through the clusters and draw each bounding box
    clusters.forEach((cluster, index) => {
        const { boundingBox } = cluster;

        // Scale the bounding box dimensions and positions back to the original size
        const scaledMinX = boundingBox.minX * k;
        const scaledMinY = boundingBox.minY * k;
        const scaledWidth = (boundingBox.maxX - boundingBox.minX) * k;
        const scaledHeight = (boundingBox.maxY - boundingBox.minY) * k;

        // Set drawing properties
        // ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        // Draw the prediction text next to the bounding box
        if (predictions && predictions[index] !== undefined && predictions[index] !== -1) {
            ctx.strokeStyle = "green";
            ctx.font = "16px Arial";
            ctx.fillStyle = "yellow";
            ctx.fillText(predictions[index].toString(), scaledMinX, scaledMinY - 5);
        }else
        {
            ctx.strokeStyle = "red";
            ctx.font = "16px Arial";
            ctx.fillStyle = "yellow";
            ctx.fillText("?", scaledMinX, scaledMinY - 5);
        }
        // Draw bounding box
        ctx.strokeRect(scaledMinX, scaledMinY, scaledWidth, scaledHeight);
    });
}

function binarize(imageData) {
    // Step 1: Compute mean and std
    let total = 0;
    let totalSquared = 0;
    let numPixels = imageData.width * imageData.height;

    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = 0.299 * imageData.data[i] + 0.587 * imageData.data[i + 1] + 0.114 * imageData.data[i + 2];
        total += gray;
        totalSquared += gray * gray;
    }

    const mean = total / numPixels;
    const variance = (totalSquared / numPixels) - (mean * mean);
    const std = Math.sqrt(variance);

    // Step 2: Normalize and threshold
    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = 0.299 * imageData.data[i] + 0.587 * imageData.data[i + 1] + 0.114 * imageData.data[i + 2];
        const normalized = (gray - mean) / std;
        
        // Step 3: Thresholding
        // You can adjust this threshold if needed. Using 0 as a threshold means values below the mean will be considered as background.
        const binaryValue = normalized > 0.0 ? 255 : 0;  
        
        imageData.data[i] = binaryValue;
        imageData.data[i + 1] = binaryValue;
        imageData.data[i + 2] = binaryValue;
    }

    return imageData;
}

function get_clusters(imageData, k=1) {

    const resizedWidth = imageData.width;
    const resizedHeight = imageData.height;

    const visited = new Set();
    const clusterInfo = [];

    function getNeighbors(x, y) {
        return [
            [x-1, y], [x+1, y], [x, y-1], [x, y+1]
        ].filter(neighbor => {
            return neighbor[0] >= 0 && neighbor[0] < resizedWidth &&
                   neighbor[1] >= 0 && neighbor[1] < resizedHeight;
        });
    }

    function floodFill(x, y, color) {
        const stack = [[x, y]];
        let pixelsCount = 0;
        let minX = x, maxX = x, minY = y, maxY = y;
        
        while (stack.length) {
            const [cx, cy] = stack.pop();
            const idx = cy * resizedWidth + cx;
            if (visited.has(idx)) continue;
            visited.add(idx);
            pixelsCount++;

            // Update bounding box
            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;

            imageData.data[idx * 4] = color[0];
            imageData.data[idx * 4 + 1] = color[1];
            imageData.data[idx * 4 + 2] = color[2];
            for (const [nx, ny] of getNeighbors(cx, cy)) {
                const nidx = ny * resizedWidth + nx;
                if (!visited.has(nidx) && imageData.data[nidx * 4] === 0) {
                    stack.push([nx, ny]);
                }
            }
        }

        return { pixelsCount, boundingBox: { minX: minX , maxX: maxX, minY: minY , maxY: maxY  } }; //! removed K scaling
    }

    function getRandomColor() {
        return [Math.random() * 255, Math.random() * 255, Math.random() * 255];
    }

    for (let y = 0; y < resizedHeight; y++) {
        for (let x = 0; x < resizedWidth; x++) {
            const idx = y * resizedWidth + x;
            if (imageData.data[idx * 4] === 0 && !visited.has(idx)) {
                const color = getRandomColor();
                const { pixelsCount, boundingBox } = floodFill(x, y, color);
                clusterInfo.push({ firstPixel: [x * k, y * k], size: pixelsCount, boundingBox });
            }
        }
    }

    // Display the color clustered image on the canvas
    const displayCanvas = document.getElementById('grayscaleCanvas');
    displayCanvas.width = resizedWidth;
    displayCanvas.height = resizedHeight;
    const displayCtx = displayCanvas.getContext('2d');
    displayCtx.putImageData(imageData, 0, 0);

    return clusterInfo;
}

function preprocessFrame(k = 1) {
    // Get video element
    const video = document.getElementById('webcam');

    // Create a canvas to capture and resize a frame from the video
    const captureCanvas = document.createElement('canvas');
    const resizedWidth = Math.floor(video.videoWidth / k);
    const resizedHeight = Math.floor(video.videoHeight / k);
    captureCanvas.width = resizedWidth;
    captureCanvas.height = resizedHeight;
    const captureCtx = captureCanvas.getContext('2d');
    captureCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, resizedWidth, resizedHeight);

    // Get the resized image data
    const imageData = captureCtx.getImageData(0, 0, resizedWidth, resizedHeight);

    
    binarize(imageData);

    return imageData;
}

async function downloadTensor(tensor) {
    // Assuming 'tensor' is an ONNX tensor and 'tensor.data' is the method to get the actual data as a typed array.
    // If 'tensor.data' is not the correct method, replace it with the actual method to get the data from your ONNX tensor.
    console.log("in dowload, ", tensor)
    const tensorData = tensor.data; // Make sure this is an asynchronous call if necessary

    // Convert the typed array to a string (for example, by joining values with a newline character)
    const tensorDataString = Array.from(tensorData).join('\n');

    // Convert the string to a Blob
    const blob = new Blob([tensorDataString], { type: 'text/plain' });

    // Rest of your download logic...
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tensor_data.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

async function predict(tempCanvas, boundingBox, margin, square=false) {

    if (square) {
        // Calculate width and height of bounding box
        const width = boundingBox.maxX - boundingBox.minX;
        const height = boundingBox.maxY - boundingBox.minY;
        
        // Determine the larger dimension and square the bounding box
        const maxDim = Math.max(width, height);
        const halfDiffWidth = (maxDim - width) / 2;
        const halfDiffHeight = (maxDim - height) / 2;

        boundingBox.minX -= halfDiffWidth;
        boundingBox.maxX += halfDiffWidth;
        boundingBox.minY -= halfDiffHeight;
        boundingBox.maxY += halfDiffHeight;
    }

    // Resize the bounding box
    const marginX = Math.floor((boundingBox.maxX - boundingBox.minX) * margin);
    const marginY = Math.floor((boundingBox.maxY - boundingBox.minY) * margin);
    boundingBox.minX = Math.max(0, boundingBox.minX - marginX);
    boundingBox.maxX = Math.min(tempCanvas.width, boundingBox.maxX + marginX);
    boundingBox.minY = Math.max(0, boundingBox.minY - marginY);
    boundingBox.maxY = Math.min(tempCanvas.height, boundingBox.maxY + marginY);

    // Crop the image to the bounding box
    const croppedCanvas = document.createElement('canvas');
    const croppedCtx = croppedCanvas.getContext('2d');
    const croppedWidth = boundingBox.maxX - boundingBox.minX;
    const croppedHeight = boundingBox.maxY - boundingBox.minY;
    croppedCanvas.width = croppedWidth;
    croppedCanvas.height = croppedHeight;

    croppedCtx.drawImage(tempCanvas, boundingBox.minX, boundingBox.minY, croppedWidth, croppedHeight, 0, 0, 28, 28);

    

    // Extract image data from resized canvas
    const croppedImageData = croppedCtx.getImageData(0, 0, 28, 28);
    const input_ = new Float32Array(28 * 28);
    for (let i = 0; i < croppedImageData.data.length; i += 4) {
        // Since it's already grayscale, we can simply normalize using the red channel (or any single channel since they are all the same in grayscale)
        input_[i / 4] = croppedImageData.data[i] > 127 ? 0.5 : -0.5;
    }
    // var thestring = "";
    // for (let i = 0; i < input.length; i++) {
    //     thestring += input[i].toString() + ", ";
    // }
    // console.log(thestring);

    // Create an ONNX Tensor from imageData
    const tensorInput = new ort.Tensor('float32', input_,[ 1, 1, 28, 28]);
    // console.log("in predict", tensorInput);
    // Run model with Tensor
    const res = await model.run({'input':tensorInput});

    // const output = res.values().next().value.data;
    const output = res.output.data;
    // const sum = output.reduce((a, b) => a + b, 0);
    //  normalize using softmax
    const sum = output.reduce((a, b) => a + Math.exp(b), 0);
    const normalized = output.map(x => Math.exp(x) / sum);
    // console.log(Math.max(...normalized));
    const max_index = normalized.indexOf(Math.max(...normalized));
    // console.log(normalized[max_index])
    if (normalized[max_index] < 0.1) {
        return -1;
    }
    return [index_to_char[max_index], tensorInput];
}

// Inference loop
async function inferenceLoop() {
    // const ctx = video.getContext('2d');
    const k=2;
    var binary = preprocessFrame(k);
    // Create a temporary canvas to draw the ImageData
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = binary.width;
    tempCanvas.height = binary.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(binary, 0, 0);
    // get_clusters
    const clusterinfo = get_clusters(binary, k);
    const cut_off_min = 100;
    const cut_off_max = cut_off_min*8;
    const filtered_clusterinfo = clusterinfo.filter(cluster => cluster.size > cut_off_min && cluster.size < cut_off_max);
    // console.log("There are " + filtered_clusterinfo.length + " clusters");
    var predictions = [];
    const margin = 0.2;
    if (filtered_clusterinfo.length > 0) {
        // const predictions = filtered_clusterinfo.map(cluster => await predict(tempCanvas, cluster.boundingBox, margin));
        
        for (var i = 0; i < filtered_clusterinfo.length; i++) {
            const res = await predict(tempCanvas, filtered_clusterinfo[i].boundingBox, margin);
            predictions.push(res[0]);
            if((Date.now() - lastClickTime) < 100){
                // check collision 
                const { boundingBox } = filtered_clusterinfo[i];
                const scaledMinX = boundingBox.minX * k;
                const scaledMinY = boundingBox.minY * k;
                const scaledMaxX = boundingBox.maxX * k;
                const scaledMaxY = boundingBox.maxY * k;
                // Check if the click is within any bounding box
                if (mouseX >= scaledMinX && mouseX <= scaledMaxX && mouseY >= scaledMinY && mouseY <= scaledMaxY) {
                    tensors.push(res[1]);
                    // console.log("in Q", tensors[tensors.length-1]);
                    downloadTensor(tensors[tensors.length-1]);
                }
            }
        }
        predictionText.innerHTML = predictions;
        // console.log(predictions);
        //  check clickTime
        
    }
    drawBoundingBoxes(filtered_clusterinfo, predictions, k);

    const now = Date.now();
    const fps = 1000 / (now - (window.lastInferenceTime || now));
    window.lastInferenceTime = now;
    // console.log(`FPS: ${fps.toFixed(2)}`);

    requestAnimationFrame(inferenceLoop);
}

switchCamBtn.addEventListener('click', () => {
    const facingMode = video.srcObject.getVideoTracks()[0].getSettings().facingMode;
    getWebcam(facingMode === 'user' ? 'environment' : 'user');
});

// Start everything
// getWebcam();

initModel().then(() => {
    getWebcam();
});
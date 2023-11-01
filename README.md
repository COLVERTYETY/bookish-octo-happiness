# Optical Character Recognition with onnx web

This is a simple demo of how to perform OCR on the client side using ONNX web.

# algorithm overview:

1. train a small CNN with PyTorch on the EMNIST dataset.

2. convert the trained model to ONNX format.

3. load the ONNX model in a web browser and perform inference on the client side.
   1. Binarize the image.
   2. perform clustering via flood fill algorithm.
   3. extract the bounding boxes of the clusters.
   4. resize and perform inference on the bounding boxes.
   5. profit.
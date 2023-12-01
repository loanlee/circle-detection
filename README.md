# Circle Detection using CNNs

**Overview:**

This project trains a Convolutional Neural Network to detect circles in noisy images. Synthetic images of circles are generated with added noise, a CNN model is trained to recognize these circles, and its performance is evaluated on a test dataset. The model outputs parameters that define the detected circle's position and radius.

**Key Features:**

- Generation of synthetic images with circles and noise.
- Training a CNN model for circle detection.
- Evaluation of the model's performance using Intersection over Union (IoU) scores.
- Accuracy metrics at various IoU thresholds to assess detection reliability.

**Contents:**

1. [Model Architecture](#model-architecture)
2. [Usage](#usage)
    - [Generating Synthetic Data](#generating-synthetic-data)
    - [Training the Model](#training-the-model)
    - [Testing the Model](#testing-the-model)
    - [Visualizing Model Performance](#visualizing-model-performance)
3. [Results](#results)

# Model Architecture

This project uses a Convolutional Neural Network (CNN) architecture with the following key components:

1. **Convolutional Blocks:**
   - ConvBlock1: Convolutional layer (32 filters, kernel size 5x5) with Batch Normalization, ReLU activation, and MaxPooling (2x2).
   - ConvBlock2: Convolutional layer (64 filters, kernel size 3x3) with Batch Normalization, ReLU activation, and MaxPooling (2x2).
   - ConvBlock3: Convolutional layer (128 filters, kernel size 3x3) with Batch Normalization, ReLU activation, and MaxPooling (2x2).
   - ConvBlock4: Convolutional layer (4 filters, kernel size 1x1) with Batch Normalization and ReLU activation.

2. **Fully Connected Layers:**
   - FC1: Fully connected layer with 128 neurons and ReLU activation.
   - FC2: Fully connected layer with 16 neurons and ReLU activation.
   - Output Layer: Fully connected layer with 3 neurons for circle parameter prediction (row, col, radius).

The model architecture is designed to process input images (1 channel, 100x100 pixels) and output predictions for circle parameters, enabling accurate detection in the presence of noise.


# Usage

Clone this repo. 

```bash
git clone https://github.com/loanlee/circle-detection.git
```

Run the notebook in sequence.

The `Model Training` section in the notebook can be skipped as long as the model weights are present in the right directory. 

## Generating Synthetic Data

Noisy images with circles are generated using helper functions. A wrapper dataset class called `CircleDataset` uses this generator to provide examples to feed to our model. 

## Training the Model

The dataset class in instantiated, which is fed into a dataloader that prepares batches of examples for our training. The model is trained to minimize `MSELoss` using the `Adam` optimizer for 30 epochs. 

In our training loop, we scale our targets down by a factor of hundred for stable training. (The original targets are in the range `(0,100) `for a `100*100` image)

```bash
loss = criterion(outputs, targets/100)
```

The weights are then saved for later use. 

## Testing the Model

After loading our saved weights, a dataloader for test examples feeds data into the model for testing. The metric used is accuracy over thresholded IOU - that is the fraction of examples whose predictions have IOU scores over a certain threshold. 

A couple of helper functions are implemented to handle the tensors while calculating IOUs and further calculating the accuracy over thresholded IOU over the entire test dateset. 

The model is then tested over a range of threshold values - `[0.5, 0.75, 0.9, 0.95]`.

## Visualizing Model Performance

Some sample circles are passed through our model to obtain predictions. The circles are plotted and the model predictions for the circle parameters are compared with the actual parameters of the circle. The IOU scores for our prediction are calculated.

# Results

The following table shows the results for a test set of 200 noisy images with circles.

| IoU Threshold | Average Accuracy |
|---------------|------------------|
| 0.5           | 1.0              |
| 0.75          | 1.0              |
| 0.9           | 0.975            |
| 0.95          | 0.915            |




# Deep Learning Image Classification with PyTorch
-------------------------------------------------

Overview:
---------
This project is an image classification project built using PyTorch. It demonstrates
how to develop a robust convolutional neural network (CNN) to classify images into 20
different categories. The project employs data augmentation techniques, a custom dataset,
and training strategies such as learning rate scheduling and early stopping to enhance
model performance.

Features:
---------
- Data Augmentation:
  * Resizes images to 100x100 pixels.
  * Applies random horizontal and vertical flips.
  * Performs random rotations and color jittering.
  
- Custom Dataset:
  * Extends a base dataset class (ImagesDataset) to incorporate the image
    transformations.

- CNN Architecture:
  * Multi-layer CNN using convolutional layers, batch normalization, LeakyReLU activations,
    dropout regularization, and max pooling.

- Training Pipeline:
  * Splits the dataset into training and testing sets.
  * Uses a learning rate scheduler (ReduceLROnPlateau) to adjust the learning rate.
  * Implements early stopping based on validation loss.
  * Saves the best model weights to 'model.pth'.

- Evaluation:
  * Evaluates the model on the test set and outputs the test loss and accuracy.

Project Structure:
------------------
The repository is structured as follows:

    ├── README.txt          # This file
    ├── requirements.txt    # Python dependencies
    ├── dataset.py          # Custom dataset implementation (ensure this file is present)
    ├── train.py            # Main training script containing the model, training, and evaluation loops
    └── model.pth           # Saved model weights (generated after training)



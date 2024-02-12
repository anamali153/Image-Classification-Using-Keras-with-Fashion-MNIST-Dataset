This repository contains a Keras implementation of an image classifier for the Fashion MNIST dataset using Convolutional Neural Networks (CNNs). Fashion MNIST is a dataset consisting of 28 by 28 grayscale images of various clothing items categorized into 10 labels.

Dataset
The Fashion MNIST dataset comprises the following labels:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
The dataset is split into a training set of 60,000 images and a test set of 10,000 images.
Approach
Data Loading and Visualization: The dataset is loaded using Keras, and a sample image is visualized using matplotlib.
Data Preprocessing:
Normalization: The pixel values of the images are normalized by dividing by 255.0 to scale them between 0 and 1.
Reshaping: The input image arrays are reshaped to include a fourth dimension for the single channel.
One-hot Encoding: The labels are converted to one-hot encoded vectors to facilitate categorical analysis.
Model Architecture: The CNN model is constructed using Keras with the following layers:
Convolutional Layer: 2D convolutional layer with 32 filters and a kernel size of (4, 4), followed by ReLU activation.
Pooling Layer: MaxPooling layer with a pool size of (2, 2).
Flatten Layer: Flattens the output from the previous layers.
Dense Layers: Two dense layers with 128 and 10 neurons respectively, with ReLU and softmax activations.
Model Training: The model is compiled using categorical cross-entropy loss and the RMSprop optimizer. It is then trained on the training data for 5 epochs with a batch size of 64.
Model Evaluation: The trained model is evaluated on the test data, and performance metrics such as accuracy are computed. Additionally, a classification report is generated using scikit-learn.
Usage
Clone the repository.
Ensure dependencies are installed (tensorflow, keras, numpy, matplotlib, sci-kit-learn).
Run the Jupyter Notebook or Python script to train and evaluate the model.
Results
The model achieves an accuracy of 0.90  on the test set.
The classification report provides detailed performance metrics for each class.# Image-Classification-Using-Keras-with-Fashion-MNIST-Dataset

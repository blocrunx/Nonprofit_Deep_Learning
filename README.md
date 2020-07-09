# Nonprofit_Deep_Learning

### Overview

This project explores applying neural networks and deep learning techniques to a dataset of Nonprofit organizations that have received funding from venture capital financers Alphabet Soup. The goal is to find a machine learning model that will analyze the success of past ventures in order to make more informed decisions about future ventrues. To accomplish this task the following steps will be taken:

  - Drop dimensions that are not needed.
  - Combine rare categorical values via bucketing.
  - Encode categorical variables using one-hot encoding.
  - Standardize numerical variables using Scikit-Learn’s Standard Scaler class.
  - Using a TensorFlow neural network design, create a binary classification model that can predict if an Alphabet Soup funded organization will be successful based on the features in the dataset.
  
### Resources

- Data Source: LoanStats_2019Q1.csv
- Software: Python 3.7.6, Jupyter Lab 1.2.6, Scikit-Learn 0.23.0, TensorFlow 2.1.0

### Summary

It was decided that the initial implimentation of the model would use the reLu activation function for 39 hidden neurons in 3 hidden layers:
- 27 neurons in the first layer.
- 9 neurons in the second layer. 
- 3 neurons in the third layer. 
- a single sigmoid neuron in the output function.

After 50 epochs the performance of the model was sub-par with a loss of 0.5733, an accuracy of 0.7186 , and was showing signs of overfitting. The model was initially tested with a single target and single output using the binary cross entropy loss function. To address performance issues the target column was encoded and split into success and failure columns. The next model would use the same hidden layer and neuron structure, using the newly encoded success column as a target. This modification helped the loss, accuracy, and overfitting so the epochs were increased to 100 which produced a loss of 0.5669, an accuracy of 0.



# Nonprofit_Deep_Learning

### Overview 

  

This project applies neural net and deep learning models to predict which nonprofit organizations will make good use of funding. The goal is to find a machine learning model that will analyze the success of past ventures in order to make more informed decisions in the future. To accomplish this task the following steps will be taken: 

  

  - Drop dimensions that are not needed. 

  - Combine rare categorical values via bucketing. 

  - Encode categorical variables using one-hot encoding. 

  - Standardize numerical variables using Scikit-Learn’s Standard Scaler class. 

  - Using a TensorFlow neural network design, create a binary classification model that can predict if an Alphabet Soup funded organization will be successful based on the features in the dataset. 

   

### Resources 

  

- Data Source: LoanStats_2019Q1.csv 

- Software: Python 3.7.6, Jupyter Lab 1.2.6, Scikit-Learn 0.23.0, TensorFlow 2.1.0 

  

### Summary 

  

It was decided that the initial implementation of the model would use the reLu activation function with 39 hidden neurons in 3 hidden layers: 

- 27 neurons in the first layer. 

- 9 neurons in the second layer.  

- 3 neurons in the third layer.  

- a single sigmoid neuron in the output function. 

  

After 50 epochs the performance of the model was sub-par with a loss of 0.5733, an accuracy of 0.7186, and signs of overfitting. The model was initially tested with a single target and single output using the binary cross entropy loss function. To address performance issues the target column was encoded and split into success and failure columns. The next model would use the same hidden layer and neuron structure, using the newly encoded success column as a target. This modification helped the loss, accuracy, and overfitting, so the epochs were increased to 100 which produced a loss of 0.5669, an accuracy of 0.7241. The model was able to be further improved by making both success and failure columns the targets, changing the loss function to categorical cross entropy and increasing the number of neurons in each layer. Although this improved the loss to 0.5648, and the accuracy to 0.7285, more work was needed to improve the model. 

  

The next step was to try adding more neurons and layers to the model which ended up decreasing performance. It was time to take another look at the preprocessing. More data was bucketed, some previously removed data was re-added which gave another increase in accuracy. The final model (found in the notebook neural_net_experiments.ipynb) has a loss of 0.5397, an accuracy of 0.7404 and no signs of overfitting. This model uses categorical cross entropy, and 4 hidden layers: 

- 60 neurons with sigmoid activation. 

- 20 neurons with sigmoid activation. 

- 10 neurons with sigmoid activation. 

- 10 neurons with sigmoid activation. 

- 2 neurons in the output function with softmax activation. 

  

In conclusion the model fell just short of reaching the 75% target. Further modification of the data and models may lead to a model that is overfit, which would lead to poor predictability. It would be advisable to explore other machine learning options such as random forests as they can prove to be a better fit for certain classification use cases. 

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from Neural_Network import Neural_Network
from sklearn import linear_model
from matplotlib import pyplot as plt
import numpy as np

#params
EPOCHS = 10000
LEARNING_RATE = 1.2
HIDDEN_LAYER_SIZE = 4

#load dataset
X, Y = load_planar_dataset()

#get shape info
print(f'X shape : {X.shape}')
print(f'Y shape : {Y.shape}')
sample_count = Y.shape[1]
print(f'example counts : {sample_count}')

#import logistic regression model and see how logistic regression perform on this dataset
logistic_regression_model = linear_model.LogisticRegression()
logistic_regression_model.fit(X.T, Y.T.ravel())

#print accuracy
predictions = logistic_regression_model.predict(X.T)
accuracy = (Y.shape[1] - np.sum(np.abs(np.subtract(np.array(Y, dtype="int"), np.array(predictions, dtype="int"))))) / sample_count * 100
print(f'accuracy : %{accuracy}')

#create multi layers neural network
model = Neural_Network(EPOCHS, LEARNING_RATE, HIDDEN_LAYER_SIZE)
model.set_layer_sizes(X, Y)
model.train_model(X, Y)
model.test_model(X, Y)









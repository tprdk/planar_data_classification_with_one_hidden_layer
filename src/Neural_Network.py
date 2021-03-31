import numpy as np
from planar_utils import sigmoid


#cross-entropy loss
def compute_cost(sample_count, output, label):
    logistic_probabilities = np.multiply(np.log(output), label) + np.multiply((1 - label), np.log(1 - output))
    cost                   = - np.sum(logistic_probabilities) / sample_count
    np.squeeze(cost)
    return cost

class Neural_Network():
    def __init__(self, epochs, learning_rate, hidden_layer_size):
        np.seterr(all='warn')
        self.epochs         = epochs
        self.learning_rate  = learning_rate

        #init weights and bias
        self.weights_1 = None
        self.weights_2 = None
        self.bias_1   = None
        self.bias_2   = None

        self.d_weights_1 = None
        self.d_weights_2 = None
        self.d_bias_1    = None
        self.d_bias_2    = None

        #init layer node counts
        '''
            - n_x: the size of the input layer
            - n_h: the size of the hidden layer (set this to 4)
            - n_y: the size of the output layer
        '''
        self.n_h    = hidden_layer_size
        self.n_x    = 0
        self.n_y    = 0

        '''
            -cache vectors from forward propagation
        '''
        self.firs_layer_output      = None
        self.tanh_firs_layer_output = None

        self.second_layer_output         = None
        self.sigmoid_second_layer_output = None


    def set_layer_sizes(self, x_train, y_train):
        self.n_x = x_train.shape[0]
        self.n_y = y_train.shape[0]
        print(f'input feature count : {self.n_x}')
        print(f'hidden layer count : {self.n_h}')
        print(f'output node count : {self.n_y}')

        '''
            set the weights shape and init it with random values
            and set bias shapes and fill it with zeros 
            w_1 = (n_h ,n_x)  # first hidden layer weights
            b_1 = (n_h, 1)
            
            w_2 = (n_y, n_h)
            b_2 = (n_y, 1)
        '''
        np.random.seed(2)
        self.weights_1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.weights_2 = np.random.randn(self.n_y, self.n_h) * 0.01

        self.bias_1 = np.zeros(shape=(self.n_h, 1))
        self.bias_2 = np.zeros(shape=(self.n_y, 1))

        print(f'parameters :')
        print(f'weights_1 :\n{self.weights_1}')
        print(f'weights_2 :\n{self.weights_2}')
        print(f'bias_1 : {self.bias_1}')
        print(f'bias_2 : {self.bias_2}\n')


    def train_model(self, x_train, y_train):
        x_train = np.array(x_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64)

        for i in range(self.epochs):
            prediction = self.forward_propagation(x_train)
            cost = compute_cost(y_train.shape[1], prediction, y_train)

            if i % 1000 == 0:
                print(f'iteration : {i} -> cost : {cost}')

            self.backward_propagation(x_train, y_train)
            self.update_weights_biases()




    def forward_propagation(self, x_train):
        self.firs_layer_output       = np.dot(self.weights_1, x_train) + self.bias_1
        self.tanh_firs_layer_output  = np.tanh(self.firs_layer_output)

        self.second_layer_output         = np.dot(self.weights_2, self.tanh_firs_layer_output) + self.bias_2
        self.sigmoid_second_layer_output = sigmoid(self.second_layer_output)

        assert (self.sigmoid_second_layer_output.shape == (1, x_train.shape[1]))

        return self.sigmoid_second_layer_output


    def backward_propagation(self, x_train, y_train):
        sample_count = x_train.shape[1]

        d_second_layer_output = self.sigmoid_second_layer_output - y_train
        self.d_weights_2      = (1 / sample_count) * np.dot(d_second_layer_output, self.tanh_firs_layer_output.T)
        self.d_bias_2         = (1 / sample_count) * np.sum(d_second_layer_output, axis=1, keepdims=True)

        d_first_layer_output = np.multiply(np.dot(self.weights_2.T, d_second_layer_output), (1 - np.power(self.tanh_firs_layer_output, 2)))
        self.d_weights_1     = (1 / sample_count) * np.dot(d_first_layer_output, x_train.T)
        self.d_bias_1        = (1 / sample_count) * np.sum(d_first_layer_output, axis=1, keepdims=True)


    def update_weights_biases(self):
        self.weights_1 = self.weights_1 - self.learning_rate * self.d_weights_1
        self.weights_2 = self.weights_2 - self.learning_rate * self.d_weights_2

        self.bias_1 = self.bias_1 - self.learning_rate * self.d_bias_1
        self.bias_2 = self.bias_2 - self.learning_rate * self.d_bias_2


    def test_model(self, x_test, y_test):
        prediction = self.forward_propagation(x_test)
        accuracy = np.sum(np.abs(np.subtract(y_test, np.round(prediction))))
        accuracy = y_test.shape[1] - accuracy
        print(f'accuracy : %{accuracy / y_test.shape[1] * 100}')

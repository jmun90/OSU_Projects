import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as activation_funciton

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

class nNetwork:
    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_out_hidden = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

    def train(self):
        pass

    def run(self):
        pass

  
if __name__ == "__main__":
    simple_network = nNetwork(no_of_in_nodes=3, no_of_out_nodes=2, no_of_hidden_nodes=4, learning_rate=0.1)

    print(simple_network.weights_in_hidden)
    print(simple_network.weights_out_hidden)
import numpy as np
import helpers as helpers

class Layer:
    def __init__(self, nNeurons, nInputs, activationFunction):
        #fill weights with neuron rows and inputs elements in each row
        self.weights = np.random.uniform(-1, 1, (nNeurons, nInputs))
        #create bias array of ones with size of neurons
        self.biases = np.zeros((nNeurons, 1))
        self.activationFunction = activationFunction
        pass

    def __repr__(self) -> str:
        return "\nneurons: {}\nweights: {}\nbiases: {}\n".format(len(self.biases), self.weights, self.biases)

    #determine dot product of weight matrix and previous layers inputs as 1D array
    def calculateOutput(self, inputs):
        output = np.dot(self.weights, inputs)
        output = output + self.biases
        return output

    #put outputs through activation function
    def calculateActivation(self, inputs):
        return self.activationFunction(inputs)

    #use deltas input from previous layer and training rate to update layer weight matrix
    def updateWeights(self, delta, input, rate):
        changes = np.dot(delta, input.T) * rate 
        self.weights = self.weights + changes
        return True

    def updateBiases(self, gradient):
        self.biases = self.biases + gradient
        pass
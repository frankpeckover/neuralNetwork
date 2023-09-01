from neuralNetwork import NeuralNetwork
import helpers
import dataProcessor

x_train, x_test, y_train, y_test = dataProcessor.handleCSV('water_data.csv', 'is_safe', 0.1)

#initialise neural network
networkShape = [20, 20, 20, 5]
activationFunctions = [helpers.relu, helpers.tanh, helpers.sigmoid, helpers.relu]
trainingRate = 0.2

brain = NeuralNetwork(networkShape, activationFunctions, trainingRate)

#testing and training the network
print("-------------    BEFORE TRAINING ------------")
print("Accuracy: ", brain.accuracy(x_test, y_test) * 100)

errors = brain.train(x_train, y_train)

print("-------------    AFTER TRAINING  ------------")
print("Accuracy: ", brain.accuracy(x_test, y_test) * 100)
print("---------------------------------------------")
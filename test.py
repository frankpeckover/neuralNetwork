import numpy as np
from neuralNetwork import NeuralNetwork

def matchOutputs(totalData):
    expectedOutputs = []
    for data in totalData:
        if data[0][0] > 0:
            if data[1][0] > 0:
                expectedOutputs.append("A")
            else:
                expectedOutputs.append("D")
        else:
            if data[1][0] > 0:
                expectedOutputs.append("B")
            else:
                expectedOutputs.append("C")
    return expectedOutputs


#generate training and testing data
trainingData = np.random.uniform(-1, 1, (10000, 2, 1))
testData = np.random.uniform(-1, 1, (500, 2, 1))
# print("Training Data: ", trainingData)

expectedTrainingOutputs = matchOutputs(trainingData)
# print("ExpectedTrainingOuputs: ", expectedTrainingOutputs)

expectedTestOutputs = matchOutputs(testData)
# print("ExpectedTestOuputs: ", expectedTestOutputs)

#generate neural network and instantiate
networkShape = [2, 10, 10, 4]
trainingRate = 0.2

brain = NeuralNetwork(networkShape, trainingRate)


#testing and training the network
print("-------------    BEFORE TRAINING ------------")
print("Accuracy: ", brain.accuracy(testData, expectedTestOutputs))

errors = brain.train(trainingData, expectedTrainingOutputs)

print("-------------    AFTER TRAINING  ------------")
print("Accuracy: ", brain.accuracy(testData, expectedTestOutputs))



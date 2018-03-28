
from Neuron import neuron
from sklearn.metrics import confusion_matrix

import random

class NeuralNetwork:

    random.seed(15)

    # Static paremeters of the Network Layers
    inputs = 11
    outputs = 3

    # Static parameters of the Network
    bias = 1.0
    learningRate = 0.2

    # Define the hidden and output layers in the network
    hiddenLayer = []
    outputLayer = []


    def __init__(self, hiddenSize):

        #Create the hidden layer of the network and asign random weights
        for i in range(hiddenSize):
            tempWeights = []
            for x in range(self.inputs):
                tempWeights.append(float(random.uniform(-1, 1)))
            self.hiddenLayer.append(neuron(tempWeights, self.bias, 0))

        # Create the output layer of the network. There is one neuron per classification (Wine Rating)
        tempWeights1 = []
        for x in range(hiddenSize):
            tempWeights1.append(float(random.uniform(-1, 1)))
        self.outputLayer.append(neuron(tempWeights1, self.bias, 5))

        tempWeights2 = []
        for x in range(hiddenSize):
            tempWeights2.append(float(random.uniform(-1, 1)))
        self.outputLayer.append(neuron(tempWeights2, self.bias, 7))

        tempWeights3 = []
        for x in range(hiddenSize):
            tempWeights3.append(float(random.uniform(-1, 1)))
        self.outputLayer.append(neuron(tempWeights3, self.bias, 8))


    def forwardPropogation(self, dataPoint):

        hiddenLayerOutputWeights = []
        outputLayerOutputWeights = []

        # First go through the hidden layer
        for hiddenNeuron in self.hiddenLayer:
            hiddenLayerOutputWeights.append(hiddenNeuron.getNeuronOutput(dataPoint))

        # Second go through the output layer
        for outputNeuron in self.outputLayer:
            outputLayerOutputWeights.append(outputNeuron.getNeuronOutput(hiddenLayerOutputWeights))

        return outputLayerOutputWeights



    def backPropogation(self, expectedValue):

        # First loop handels the output layer neurons
        for outputNeuron in self.outputLayer:

            # Here we check to see if this neuron in the out put layer is what it should be
            if str(outputNeuron.classification) == str(expectedValue):
                outputNeuron.error = 1 - outputNeuron.neuronOutput
            else:
                outputNeuron.error = 0 - outputNeuron.neuronOutput

            # Applying the derivative of the transfer function
            outputNeuron.delta = outputNeuron.error * (outputNeuron.neuronOutput * (1.0 - outputNeuron.neuronOutput))


        # Second loop handels the hidden layer nodes
        for i in range(len(self.hiddenLayer)):

            errorSum = 0.0

            for outputNeuron in self.outputLayer:
                errorSum = errorSum + (outputNeuron.weights[i] * outputNeuron.delta)

            # Create variable to refrence current node in the hidden layer
            hiddenLayerNode = self.hiddenLayer[i]

            # Setting hidden layer nodes error and delta
            hiddenLayerNode.error = errorSum
            hiddenLayerNode.delta = hiddenLayerNode.error * (hiddenLayerNode.neuronOutput * (1.0 - hiddenLayerNode.neuronOutput))




    def train(self, epoch, wineTrainingData, expectedWineRatings):

        for i in range(epoch):

            print("epoch: ", i)
            counter = 0

            for dataPoint in wineTrainingData:

                self.forwardPropogation(dataPoint)
                self.backPropogation(expectedWineRatings[counter])

                outputLayerInputs = []


                # Go through nodes and update the weights after forward and back prop
                for hiddenLayerNeuron in self.hiddenLayer:

                    for i in range(len(dataPoint)):

                        hiddenLayerNeuron.weights[i] += self.learningRate * hiddenLayerNeuron.delta * dataPoint[i]

                    hiddenLayerNeuron.bias += self.learningRate * hiddenLayerNeuron.delta
                    outputLayerInputs.append(hiddenLayerNeuron.neuronOutput)


                for outputLayerNeuron in self.outputLayer:

                    for j in range(len(outputLayerInputs)):

                        outputLayerNeuron.weights[j] += self.learningRate * outputLayerNeuron.delta * outputLayerInputs[j]

                    outputLayerNeuron.bias += self.learningRate * outputLayerNeuron.delta

                counter += 1


    def test(self, testWineData, testWineRatings):

        results = []
        counter = 0
        correctCounter = 0

        expectedValues = []
        predictedValues = []

        for dataPoint in testWineData:

            outputValues = self.forwardPropogation(dataPoint)

            classification = outputValues.index(max(outputValues))

            if classification == 0:
                prediction = 5
            elif classification == 1:
                prediction = 7
            else:
                prediction = 8

            expectedValue = testWineRatings[counter]

            expectedValues.append(int(expectedValue))
            predictedValues.append(int(prediction))

            if str(prediction) == str(expectedValue):
                result = "Predicted to be: " + str(prediction) + " Actually: " + str(expectedValue) + " Result: Correct\n"
                results.append(result)
                correctCounter += 1
            else:
                result = "Predicted to be: " + str(prediction) + " Actually: " + str(expectedValue) + "Result: Incorrect\n"
                results.append(result)

            counter += 1

        accuracyStatement = "\n\nThe accuracy is: " + str(correctCounter/len(testWineData) * 100) + "%"

        matrix = confusion_matrix(expectedValues, predictedValues, labels=[5, 7, 8])

        results.append(accuracyStatement)
        return {'x1': results, 'x2': matrix}


















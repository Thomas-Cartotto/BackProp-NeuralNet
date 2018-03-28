# Thomas Cartotto

import csv
from sklearn import preprocessing
import random
from NeuralNetwork import NeuralNetwork

# data will be an array of arrays, each array will hold one rows data. This is the training segement
trainWineData = []
trainWineRatings = []

# Same as above but the segment used for testing the accuracy.
testWineData = []
testWineRatings = []

# Parameters of network that can be varied
hiddenLayerSize = 10
epoch = 3000
percentageForTraining = 0.7


def main():

    with open('data.csv') as csvDataFile:

        dataSet = csv.reader(csvDataFile)

        totalWineData = []
        totalWineResults = []

        for row in dataSet:

            totalWineResults.append(row[11])
            dataPoint = []

            for x in row[:-1]:
                dataPoint.append(x)

            totalWineData.append(dataPoint)

        # Here we want to remove the first value in the array as that is the column heading (string)
        totalWineResults.pop(0)
        totalWineData.pop(0)

        # Use external library to normalize the data between 0-1
        normalizedWineData = preprocessing.normalize(totalWineData)
        dataLength = len(normalizedWineData)

        # Determine the number of data points to allocate to training
        numberOfTrainingPoints = int(round(percentageForTraining * dataLength))

        # Each array holds the indicies we want to take from the normalized data set
        indexesOfTraining = []
        indexesOfTesting = []

        for i in range(len(totalWineResults)):
            if totalWineResults[i] == 8 and len(indexesOfTraining) < 122:
                indexesOfTraining.append(i)

        # Fill the test data set while not repeating any of the values in it
        while len(indexesOfTraining) != numberOfTrainingPoints:
            index = random.randint(0, dataLength - 1)
            print(index)
            if index not in indexesOfTraining:
                indexesOfTraining.append(index)

        for i in range(dataLength):
            if i not in indexesOfTraining:
                indexesOfTesting.append(i)

        for value in indexesOfTraining:
            trainWineData.append(normalizedWineData[value])
            trainWineRatings.append(totalWineResults[value])

        for value in indexesOfTesting:
            testWineData.append(normalizedWineData[value])
            testWineRatings.append(totalWineResults[value])

        # Create instance of the network, train it and then test it.
        neuralNetwork = NeuralNetwork(hiddenLayerSize)
        neuralNetwork.train(epoch, trainWineData, trainWineRatings)
        results = neuralNetwork.test(testWineData, trainWineRatings)

        # Create the results file that is out putted
        resultsFile = open("results.txt", "w+")
        resultsFile.write("//////////////////////////////////////////////////////////\nTEST RESULTS:\n\n")

        for result in results['x1']:
            resultsFile.write(result)

        resultsFile.write("\n\n//////////////////////////////////////////////////////////\nFINAL NODE WEIGHTS:")

        for nodeH in neuralNetwork.hiddenLayer:
            resultsFile.write("\n\nHidden Layer node: ")
            for weight in nodeH.weights:
                resultsFile.write(str(weight) + ", ")

        for nodeO in neuralNetwork.outputLayer:
            resultsFile.write("\n\nOutput Layer node: ")
            for weight in nodeO.weights:
                resultsFile.write(str(weight) + ", ")


        resultsFile.write("\n\n//////////////////////////////////////////////////////////\nCONFUSTION MATRIX:\n\n")

        matrix = results['x2']

        resultsFile.write("   P5           P7        P8")
        resultsFile.write("\nA5 %f  %f  %f" % (matrix[0][0], matrix[0][1], matrix[0][2]))
        resultsFile.write("\nA7 %f  %f  %f" % (matrix[1][0], matrix[1][1], matrix[1][2]))
        resultsFile.write("\nA8 %f  %f  %f" % (matrix[2][0], matrix[2][1], matrix[2][2]))


        resultsFile.write("\n\n//////////////////////////////////////////////////////////\nPRECISION AND RECALL:\n\n")

        resultsFile.write("Rating 5: Precision =  %f  Recall = %f\n\n" % ((matrix[0][0]/(matrix[0][0] + matrix[1][0] + matrix[2][0])),(matrix[0][0]/(matrix[0][0] + matrix[0][1] + matrix[0][2]))))
        resultsFile.write("Rating 7: Precision =  %f  Recall = %f\n\n" % ((matrix[1][1]/(matrix[0][1] + matrix[1][1] + matrix[2][1])),(matrix[1][1]/(matrix[1][0] + matrix[1][1] + matrix[1][2]))))
        resultsFile.write("Rating 8: Precision =  %f  Recall = %f\n\n" % ((matrix[2][2]/(matrix[0][2] + matrix[1][2] + matrix[2][2])),(matrix[2][2]/(matrix[2][0] + matrix[2][1] + matrix[2][2]))))



if __name__ == "__main__":
    main()

from Data import DataReader, DataGenerator
from NeuralNetwork import NeuralNetwork

def main():
    dataReader = DataReader("D:\Dokumenty\Studia\Praca magisterska\Software\przebiegi\combined.json")
    dataGenerator = DataGenerator(dataReader, batchSize=30, length=50)
    neuralNetwork = NeuralNetwork(dataGenerator)
    return 0

if __name__ == "__main__":
    main()
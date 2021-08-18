import json
import random
import numpy as np
from tensorflow import keras


class DataReader():
    def __init__(self, filePath, timeFrameLen = 30) -> None:
        self.filePath = filePath
        self.timeFrameLen = timeFrameLen
        self.jsonDict = {}
        self.positive = []
        self.negative = []
        self.x_data = []
        self.y_data = []

        with open(filePath, 'r') as jsonFile:
            self.jsonDict = json.load(jsonFile)

        #read oryginal data
        for data in self.jsonDict.values(): 
            for cut, isCorrect in zip(data['cutPoints'], data['isCorrect']):
                try:
                    data['dataPoints'][cut + self.timeFrameLen]
                    if isCorrect:
                        self.positive.append(data['dataPoints'][cut : cut + self.timeFrameLen])
                    else:
                        self.negative.append(data['dataPoints'][cut : cut + self.timeFrameLen])
                except IndexError:  #shift data to make them have equal length
                    if isCorrect:
                        self.positive.append(data['dataPoints'][-self.timeFrameLen:])
                    else:
                        self.negative.append(data['dataPoints'][-self.timeFrameLen:])

        #create additional linear combinattion of negative sample
        newNegative = []
        for negative_1 in self.negative:
            for negative_2 in self.negative:
                newNegative.append(list(map(lambda x, y: (x+y)/2, negative_1, negative_2)))
        self.negative = newNegative

        


        # timeseries = np.array(self.x_data)
        # print(timeseries.shape)
        # timeseries = timeseries.reshape((timeseries.shape[0], timeseries.shape[1], 1))
        # print(timeseries.shape)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataReader, batchSize, length) -> None:
        super().__init__()
        self.dataReader = dataReader
        self.batchSize = batchSize
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        half_size = int(self.batchSize/2)
        sample = list(zip(random.sample(self.dataReader.positive, half_size), [True]  * half_size))
        sample = sample + list(zip(random.sample(self.dataReader.negative, half_size), [False]  * half_size))
        np.random.shuffle(sample)
        #print(sample)
        x, y = zip(*sample)
        y = [int(value) for value in y]

        x = np.array(x)
        
        #print(x)
        x = x.reshape((x.shape[0], x.shape[1], 1))

        return np.array(x), np.array(y)

    def on_epoch_end(self):
        pass
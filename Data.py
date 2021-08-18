import json
import numpy as np
from tensorflow import keras

class DataReader():
    def __init__(self, filePath, timeFrameLen = 45) -> None:
        self.filePath = filePath
        self.timeFrameLen = timeFrameLen
        self.jsonDict = {}
        self.x_data = []
        self.y_data = []

        with open(filePath, 'r') as jsonFile:
            self.jsonDict = json.load(jsonFile)

        for data in self.jsonDict.values():
            for cut in data['cutPoints']:
                try:
                    data['dataPoints'][cut + self.timeFrameLen]
                    self.x_data.append(data['dataPoints'][cut : cut + self.timeFrameLen])
                except IndexError:  #shift data to make them have equal length
                    self.x_data.append(data['dataPoints'][-self.timeFrameLen:])
            for isCorrect in data['isCorrect']:
                self.y_data.append(int(isCorrect))  #1 if True, 0 if False

        timeseries = np.array(self.x_data)
        print(timeseries.shape)
        timeseries = timeseries.reshape((timeseries.shape[0], timeseries.shape[1], 1))
        print(timeseries.shape)
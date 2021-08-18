from tensorflow import keras

class NeuralNetwork():
    def __init__(self, inputShape, epochs = 250, batchSize = 30) -> None:
        self.epochs = epochs
        self.batchSize = batchSize
        self.model = self.buildModel(inputShape)

    def buildModel(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(1, activation="softmax")(gap)
        #output_layer = keras.layers.Dense(2, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def runModel(self, generator):
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

        self.history = self.model.fit_generator(
            generator = generator,
            use_multiprocessing=True,
            workers=6,
        )

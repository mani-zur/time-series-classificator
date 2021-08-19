from tensorflow import keras

class NeuralNetwork():
    def __init__(self, generator, epochs = 200) -> None:
        self.epochs = epochs
        self.generator = generator
        self.model = self.buildModel((self.generator.dataReader.timeFrameLen, 1))
        self.runModel()

    def buildModel(self, input_shape):
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

        #output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
        output_layer = keras.layers.Dense(2, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def buildModel_2(self, input_shape):
        input_layer = keras.layers.Input(input_shape)

        dens1 = keras.layers.Dense(150, activation = 'relu')(input_layer)

        dens2 = keras.layers.Dense(90 , activation = 'relu')(dens1)

        dens3 = keras.layers.Dense(30 , activation = 'relu')(dens2)

        #output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
        output_layer = keras.layers.Dense(2, activation="softmax")(dens3)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)


    def runModel(self):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        self.model.compile(
            optimizer=keras.optimizers.SGD(learning_rate = 0.0001) ,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.history = self.model.fit(
            self.generator,
            validation_data = self.generator,
            epochs=self.epochs,
            #use_multiprocessing=True,
            #workers=1,
        )
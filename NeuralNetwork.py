from tensorflow import keras

class NeuralNetwork():
    def __init__(self, generator = None, epochs = 200, model = None) -> None:
        self.epochs = epochs
        if generator is None and model is None:
            raise AttributeError("No model and no generator, please give one of them!")
        self.generator = generator
        if model is None:
            self.model = self.buildModel((self.generator.dataReader.timeFrameLen, 1))
            self.runModel()
        else:
            self.model = keras.models.load_model(model)

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

        bat = keras.layers.BatchNormalization()(input_layer)

        dens1 = keras.layers.Dense(150, activation = 'relu')(bat)

        dens2 = keras.layers.Dense(90 , activation = 'relu')(dens1)

        dens3 = keras.layers.Dense(30 , activation = 'relu')(dens2)

        #output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
        gap = keras.layers.GlobalAveragePooling1D()(dens3)

        output_layer = keras.layers.Dense(2, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)


    def runModel(self):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.generator.dataReader.logDir + "/best_model" , save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=50, verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=self.generator.dataReader.logDir, histogram_freq=1
            ),
        ]
        self.model.compile(
            optimizer=keras.optimizers.SGD(learning_rate = 0.003) ,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.history = self.model.fit(
            self.generator,
            validation_data = self.generator,
            epochs=self.epochs,
            callbacks = callbacks
            #use_multiprocessing=True,
            #workers=1,
        )

        self.model.save(self.generator.dataReader.logDir +"/final_model")
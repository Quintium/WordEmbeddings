import tensorflow as tf
import numpy as np
from random import randint

from Vocabulary import Vocabulary

class DataGen(tf.keras.utils.Sequence):
    data: np.ndarray
    numClasses: int
    batch_size: int
    shuffle: bool

    def __init__(self, data: np.ndarray, numClasses: int, batch_size: int = 32, shuffle: bool = True):     
        self.data = data
        self.numClasses = numClasses
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)
    
    def __getitem__(self, index: int):
        batchData = self.data[:, index * self.batch_size : (index + 1) * self.batch_size]

        X, y = tf.keras.utils.to_categorical(batchData, self.numClasses)     
        return X, y
    
    def __len__(self):
        return self.data.shape[1] // self.batch_size

class Model:
    model: tf.keras.Sequential

    def __init__(self, dimensions: int, vocabulary: Vocabulary):
        input = tf.keras.Input(shape=(vocabulary.vocabSize,))
        hidden = tf.keras.layers.Dense(dimensions)(input)
        output = tf.keras.layers.Dense(vocabulary.vocabSize, activation="softmax")(hidden)
        self.model = tf.keras.Model(inputs=input, outputs=output)

    def train(self, texts: list, vocabulary: Vocabulary, wordPercentage: float, epochsAmount: int, contextWindowSize: int = 10):
        inputs = []
        outputs = []

        for text in texts:
            for i in range(int(wordPercentage * len(text.indexList))):
                index = randint(0, len(text.indexList) - 1)
                start = max(index - contextWindowSize, 0)
                end = min(index + contextWindowSize, len(text.indexList) - 1)
                inputs.append(text.indexList[index])
                outputs.append(text.indexList[randint(start, end)])

        data = np.array([inputs, outputs])
        trainGen = DataGen(data, vocabulary.vocabSize, batch_size=32, shuffle=True)

        self.model.compile(optimizer="adam", loss="categorical_crossentropy")
        self.model.fit(trainGen, epochs=epochsAmount)

    def predictWord(self, word: int, vocabulary: Vocabulary):
        input = np.array([tf.keras.utils.to_categorical(vocabulary.index(word), vocabulary.vocabSize)])
        prediction = np.argsort(self.model.predict(input))[0][-20:][::-1]
        predictionString = ", ".join([vocabulary.word(int(index)) for index in prediction])
        print(f"Predictions for word {word}: {predictionString}")

    def saveEmbeddings(self, fileName: str, vocabulary: Vocabulary):
        np.save("models/" + fileName, self.model.layers[1].get_weights()[0])
        np.save("models/" + fileName + "_vocab", vocabulary.wordList)


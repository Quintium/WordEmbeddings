import numpy as np
from Vocabulary import Vocabulary

class Embeddings:
    vectors: list
    vocab: Vocabulary

    def __init__(self, fileName: str):
        weights = np.load("models/" + fileName + ".npy")
        self.vectors = []
        for i in range(weights.shape[0]):
            self.vectors.append(weights[i])

        self.vocab = Vocabulary(list(np.load("models/" + fileName + "_vocab.npy")))

    def wordVector(self, word: str | np.ndarray) ->np.ndarray:
        if isinstance(word, str):
            return self.vectors[self.vocab.index(word)] 
        else:
            return word
    
    def closestWords(self, word: str | np.ndarray, n: int = 1) -> str:
        distances = [(np.linalg.norm(v - self.wordVector(word)), i) for i, v in enumerate(self.vectors)]
        distances = sorted(distances, key=lambda x: x[0])
        return [self.vocab.word(pair[1]) for pair in distances[:n]]

    def indexDistance(self, index1: str, index2: str) -> float:
        return np.linalg.norm(self.vectors[index1] - self.vectors[index2])

    def distance(self, word1: str, word2: str) -> float:
        return self.indexDistance(self.vocab.index(word1), self.vocab.index(word2))
    
    def add(self, word1: str | np.ndarray, *words: list) -> np.ndarray:
        newWord = self.wordVector(word1).copy()

        for word in words:
            newWord += self.wordVector(word)

        return newWord
    
    def subtract(self, word1: str | np.ndarray, *words: list) -> np.ndarray:
        newWord = self.wordVector(word1).copy()

        for word in words:
            newWord -= self.wordVector(word)

        return newWord



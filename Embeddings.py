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

    def wordVector(self, word: str | np.ndarray) -> np.ndarray:
        if isinstance(word, str):
            return self.vectors[self.vocab.index(word)] 
        else:
            return word
    
    def closestWords(self, word: str | np.ndarray, n: int = 1) -> str:
        similarities = [(i, self.similarity(word, v)) for i, v in enumerate(self.vectors)]
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return [(self.vocab.word(pair[0]), pair[1]) for pair in similarities[:n]]

    def similarity(self, word1: str, word2: str) -> float:
        v1, v2 = self.wordVector(word1), self.wordVector(word2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
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



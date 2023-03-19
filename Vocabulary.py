from collections import Counter

class Vocabulary:
    wordList: list
    indexMap: dict
    vocabSize: int = 5000

    def __init__(self, words: list):
        self.wordList = words
        self.indexMap = {val:i for i, val in enumerate(self.wordList)}

    @classmethod
    def from_texts(cls, texts: list):
        counter = Counter()
        for text in texts:
            counter.update(text)

        words = [word for word, count in counter.most_common(Vocabulary.vocabSize - 1)]
        return cls(words)

    def index(self, word: str):
        if word in self.indexMap:
            return self.indexMap[word]
        else:
            return Vocabulary.vocabSize - 1
        
    def word(self, index: int):
        if index == Vocabulary.vocabSize - 1:
            return "-"
        else:
            return self.wordList[index]
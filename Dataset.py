from datasets import load_dataset, Dataset
from random import choice
from ProcessedText import ProcessedText

class Dataset:
    def __init__(self):
        pass

    def chooseText(self) -> str:
        pass

    def createTexts(self, n: int) -> tuple:
        texts = [ProcessedText(self.chooseText()) for i in range(n)]
        vocabulary = ProcessedText.createVocabulary(texts)
        return texts, vocabulary

class WikipediaEN(Dataset):
    dataset: Dataset

    def __init__(self):
        self.dataset = load_dataset("wikipedia", "20220301.en")

    def chooseText(self) -> str:
        return choice(self.dataset["train"])["text"]
    
class WikipediaDE(Dataset):
    dataset: Dataset

    def __init__(self):
        self.dataset = load_dataset("wikipedia", "20220301.de")

    def chooseText(self) -> str:
        return choice(self.dataset["train"])["text"]
    
class IMDB(Dataset):
    dataset: Dataset

    def __init__(self):
        self.dataset = load_dataset("imdb")

    def chooseText(self) -> str:
        return choice(self.dataset["unsupervised"])["text"]
from Vocabulary import Vocabulary

class ProcessedText:
    wordList: list
    indexList: int

    def __init__(self, rawText: str):
        self.wordList = []
        currentWord = ""

        rawText = rawText.replace("<br />", "")

        for i in range(len(rawText)):
            if rawText[i].isalpha() or rawText[i] in ["'"]:
                currentWord += rawText[i]
            else:
                if currentWord:
                    self.wordList.append(currentWord.lower())
                    currentWord = ""

        if currentWord:
            self.wordList.append(currentWord.lower())
            currentWord = ""

    @classmethod
    def createVocabulary(cls, processedTexts: list, vocabSize: int) -> Vocabulary:
        vocabulary = Vocabulary.from_texts([text.wordList for text in processedTexts], vocabSize)

        for text in processedTexts:
            text.index(vocabulary)

        return vocabulary

    def index(self, vocabulary: Vocabulary):
        self.indexList = [vocabulary.index(word) for word in self.wordList]
from Dataset import WikipediaEN, WikipediaDE, IMDB
from Model import Model

dataset = IMDB()
texts, vocabulary = dataset.createTexts(30000)

model = Model(64)
for i in range(1):
    model.train(texts, 0.1, 50)

model.predictWord("american", vocabulary)
model.predictWord("nice", vocabulary)
model.predictWord("extreme", vocabulary)
model.predictWord("home", vocabulary)
model.predictWord("amazing", vocabulary)
model.predictWord("mother", vocabulary)
model.predictWord("your", vocabulary)
model.predictWord("cat", vocabulary)

fileName = input("File name: ")
model.saveEmbeddings(fileName, vocabulary)




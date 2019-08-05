from model_training import train_spacy
from format_converter import spacy_format
import pandas as pd

data = pd.read_csv("")

#getting the labels from the dataFrame directly.
labels = list(data)
labels.remove("non labels from dataframe")

split_index = int(len(data)*0.8)
data_train = TRAIN_DATA[:split_index]
data_test = TRAIN_DATA[split_index:]

prdnlp = train_spacy(data_train, labels, 20)

# Save our trained Model
prdnlp.to_disk(modelfile)

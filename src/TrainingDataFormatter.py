import random
def spacy_format(data, labels):
    TRAINING_DATA=[]
    for i,sentence in enumerate(data.D_SIG):
        entities = []
        for label in labels:
            string = str(data[label].iloc[i])
            if(string == 'NA'):
                continue
            start_index = sentence.find(string)
            if(start_index == -1):
                continue
            end_index = start_index+ len(string)
            entities.append((start_index, end_index, label))
        annotations = {"entities" : entities}
        TRAINING_DATA.append((sentence, annotations))

    random.shuffle(TRAINING_DATA)     
    return TRAINING_DATA

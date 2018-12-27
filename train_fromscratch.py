import spacy
import glob
import json
import random
from tqdm import tqdm, tqdm_notebook
from spacy.util import minibatch, compounding
from sklearn.utils import shuffle
from pathlib import Path

TRAIN_DATA = []
files = glob.glob("train/normal/*.json")
for f in files:
    with open(f) as fl:
        js = json.load(fl)
        for j in js:
            TRAIN_DATA.append(tuple(j))


### NEW TRAIN 
# output="models/"
# n_iter=100
# batch_size = 1024

# nlp = spacy.blank('en')

# ner = nlp.create_pipe('ner')
# nlp.add_pipe(ner)

# ner.add_label("DISEASE")
# ner.add_label("GENE")

# optimizer = nlp.begin_training(use_gpu=True, device=0)

# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
# with nlp.disable_pipes(*other_pipes):  # only train NER
#     for itn in range(n_iter):
#         print("Epoch : ", itn, " batch : ", len(TRAIN_DATA)/batch_size)
#         random.shuffle(TRAIN_DATA)
#         losses = {}
#         batches = minibatch(TRAIN_DATA, size=batch_size)
#         for batch in tqdm(batches):
#             texts, annotations = zip(*batch)
#             nlp.update(
#                 texts,  # batch of texts
#                 annotations,  # batch of annotations
#                 drop=0.5,  # dropout - make it harder to memorise data
#                 losses=losses,sgd=optimizer
#             )
#         print("Losses", losses)
#         output_dir = Path(output+"fs_normal/epoch_"+str(itn))
#         if not output_dir.exists():
#             output_dir.mkdir()
        
#         nlp.meta['name'] = "ner_fs_normal"
#         nlp.to_disk(output_dir)





### CONTINU TRAIN

output="models/"
n_iter=100
batch_size = 1024

nlp = spacy.load('models/fs_normal/epoch_32')

ner = nlp.get_pipe("ner")


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        print("Epoch : ", itn+33, " batch : ", len(TRAIN_DATA)/batch_size)
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=batch_size)
        for batch in tqdm(batches):
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  # batch of texts
                annotations,  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                losses=losses
            )
        print("Losses", losses)
        output_dir = Path(output+"fs_normal/epoch_"+str(itn+33))
        if not output_dir.exists():
            output_dir.mkdir()
        
        nlp.meta['name'] = "ner_fs_normal"
        nlp.to_disk(output_dir)
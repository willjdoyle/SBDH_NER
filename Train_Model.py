# William Doyle
# COMP.5520 Foundations of Digital Health
# Project 3 - NER Training Using spaCy
# Due 4/15/23

# Given annotated MIMIC III data (train.csv), train a model.

# This script will train the model.

# Reference:
# https://github.com/uml-digital-health/Labs/blob/main/NER_examples/NER_using_Scispacy_vectors.ipynb

# -------
# Imports
# -------
import os
import spacy
import random
import warnings
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.tokens import DocBin

warnings.filterwarnings('ignore') # ignore the scibert compatibility warning

# -----------
# Setup Model
# -----------
print("Setting up model...")

# use GPU for faster training
if(not spacy.prefer_gpu()):
    print("GPU not found, training will be slower.")
else:
    spacy.require_gpu()

nlp = spacy.load("en_core_sci_scibert")
ner_model = spacy.blank('en')
ner = ner_model.add_pipe('ner', last=True)
ner_model.vocab.vectors = nlp.vocab.vectors

# ------------------
# Load Training Data
# ------------------
print("Loading training data...")

# load from local file (created by SBDH_NER.py)
db_train = DocBin().from_disk("train.spacy")

# create list from the deserialized DocBin
# reference: https://github.com/explosion/spaCy/discussions/10717
loaded_data = []
for doc in db_train.get_docs(nlp.vocab):
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))

    loaded_data.append((doc.text, {"entities": entities}))

# ---------------
# Preprocess Data
# ---------------
# Convert training data to Example objects
# reference: https://github.com/uml-digital-health/Labs/blob/main/NER_examples/NER_using_Scispacy_vectors.ipynb
print("Preparing data and adding labels...")

train_x = []
labels = []
for text, anns in loaded_data:
    for start, end, label in anns['entities']:
        if label not in labels:
            labels.append(label)
    example = Example.from_dict(nlp.make_doc(text),anns)
    train_x.append(example)

# Add the NER labels to the model
for label in labels:
    ner.add_label(label)

# ---------------
# Train the Model
# ---------------
# reference: https://stackoverflow.com/questions/66342359/nlp-update-issue-with-spacy-3-0-typeerror-e978-the-language-update-method-ta
print("Training model...")

n_iter = 750
batch_size = 3000
ner_model.begin_training()
for i in range(n_iter):
    random.shuffle(train_x)
    batches = minibatch(train_x, size=compounding(4.0, 32.0, 1.01)) # ORIGINAL: (4.0, 32.0, 1.001)
    losses = {}
    for batch in batches:
        ner_model.update(batch, losses=losses, drop=0.01) # ORIGINAL: drop=0.5
    print(f"Iteration {i}: Loss={losses['ner']}")

# ----------
# Save Model
# ----------
ner_model.to_disk('ner_based_on_scibert')
nlp.to_disk('new_scibert')
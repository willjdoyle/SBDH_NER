# William Doyle
# COMP.5520 Foundations of Digital Health
# Project 3 - NER Training Using spaCy
# Due 4/15/23

# Given annotated MIMIC III data (train.csv), train a model.

# This auxiliary script is not necessary, and only serves to visualize/check data.

# -------
# Imports
# -------
import spacy
from spacy.tokens import DocBin
from spacy import displacy

nlp = spacy.load("ner_based_on_scibert2")

# ------------------
# Load Training Data
# ------------------
print("Loading training data...")

# load from local file (created by SBDH_NER.py)
db_train = DocBin().from_disk("test.spacy")

# ------------
# Display Data
# ------------
# create list from the deserialized DocBin
# reference: https://github.com/explosion/spaCy/discussions/10717

# TEMP TESTING-- DELETE LATER
docs = list(db_train.get_docs(nlp.vocab))
# ---------------------------

# specify which doc you would like to have visualized
DOC_SELECTION = 50

counter = 0
for doc in db_train.get_docs(nlp.vocab):
    if(counter == DOC_SELECTION):
        displacy.serve(doc, style="ent")
    counter += 1
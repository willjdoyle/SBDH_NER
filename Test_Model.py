# William Doyle
# COMP.5520 Foundations of Digital Health
# Project 3 - NER Training Using spaCy
# Due 4/15/23

# Given annotated MIMIC III data (train.csv), train a model.

# This script will test the model.

# -------
# Imports
# -------
import os
import spacy
import random
from spacy.training import Example
from spacy.tokens import DocBin
from spacy.scorer import Scorer

# -------------------------
# Load Model & Testing Data
# -------------------------
nlp = spacy.load("ner_based_on_scibert_BEST")
ner_model = spacy.blank('en')
ner = ner_model.add_pipe('ner', last=True)
ner_model.vocab.vectors = nlp.vocab.vectors

# load from local file
db_test = DocBin().from_disk("test.spacy")

# create list from the deserialized DocBin
# reference: https://github.com/explosion/spaCy/discussions/10717
loaded_data = []
for doc in db_test.get_docs(nlp.vocab):
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))

    loaded_data.append((doc.text, {"entities": entities}))

# --------------
# Evaluate Model
# --------------
# reference: https://stackoverflow.com/questions/68213223/how-to-evaluate-trained-spacy-version-3-model
examples = []
scorer = Scorer()

for text, annotations in loaded_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    example.predicted = nlp(str(example.predicted))
    examples.append(example)

results = scorer.score(examples)
print(results)
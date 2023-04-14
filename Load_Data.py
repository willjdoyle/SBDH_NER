# William Doyle
# COMP.5520 Foundations of Digital Health
# Project 3 - NER Training Using spaCy
# Due 4/15/23

# Given annotated MIMIC III data (train.csv), train a model.

# This script will load, clean, and preprocess the data.

# -------
# Imports
# -------
import psycopg2
import pandas as pd
import math
import tqdm
import spacy
from spacy.tokens import DocBin
from spacy import displacy

# ----------------------------------
# Split Target into 80/20 Train/Test
# ----------------------------------
print('Reading in train.csv data...')
# number of rows = 27979, so 80% is ~22000
train_y = pd.read_csv('train.csv', nrows=22000, skiprows=[33,1237,6888,11047,11291,12263,13392,18360])
test_y = pd.read_csv('train.csv', skiprows=range(1,22001))

# -----------------------------
# Data Cleaning & Preprocessing
# -----------------------------
print('Preprocessing data...')
# note that there are unclassified rows at: 33, 1237, 6888, 11047, 11291, 12263, 13392, 18360, 23236
# need to iterate through the two dataframes to remove these NaN values
for i in range(len(train_y)):
    if(type(train_y.at[i,'sbdh']) is float):
        train_y.drop(i, inplace=True)

for i in range(len(test_y)):
    if(type(test_y.at[i,'sbdh']) is float):
        test_y.drop(i, inplace=True)

# reset indices (in case .drop() was called)
train_y = train_y.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)

# remove duplicate row ids to make query more efficient
# for training set:
row_ids_train = train_y.iloc[:,0].unique()
row_ids_string_train = map(str, row_ids_train)
row_ids_string_train = ','.join(row_ids_string_train)

# for testing set:
row_ids_test = test_y.iloc[:,0].unique()
row_ids_string_test = map(str, row_ids_test)
row_ids_string_test = ','.join(row_ids_string_test)

# -----------------------
# Connect to MIMIC III DB
# -----------------------
print('Running MIMIC III query...')
conn = psycopg2.connect(host="172.16.34.1", port="5432", user="mimic_demo", password="mimic_demo", database="mimic")
cur = conn.cursor()

# postgresql query for getting notes from each row_id
query_train = """
SELECT row_id, text from mimiciii.noteevents n
WHERE n.row_id IN ("""+row_ids_string_train+""")
GROUP BY row_id;
"""

query_test = """
SELECT row_id, text from mimiciii.noteevents n
WHERE n.row_id IN ("""+row_ids_string_test+""")
GROUP BY row_id;
"""

results_train = pd.read_sql_query(query_train, conn)
results_test = pd.read_sql_query(query_test, conn)

# closing database connection
conn.close()

# -----------------------
# Add Data to spaCy Docs
# -----------------------
# official spaCy tutorial used as reference:
# https://spacy.io/usage/training
#nlp = spacy.blank("en")
nlp = spacy.load("en_core_web_lg")
# extract specified substrings in train.csv from the note texts and add as doc

# format training data to be added to docbin
print("Preparing training data...")
train_x = []
prev_row_id = -1
counter = -1 # keeps track of which index that data is being taken from

for i in tqdm.tqdm(range(len(train_y))):
    curr_row_id = train_y.at[i,'row_id']
    note_row_id = results_train[results_train['row_id']==curr_row_id].index.values
    curr_note = results_train.at[note_row_id[0],'text']

    # if this note is new, create a new doc
    if(curr_row_id != prev_row_id):
        # need to trim down the note (curr_note) to only include the social history section
        # section seems to always start with "social history:" and end with two newlines ("\n\n")
        social_history_start = curr_note.lower().find("social history:")
        if(social_history_start > 0): # will return -1 if not found
            curr_note = curr_note[social_history_start + len("social history:"):] # start of note trimmed to start of social history section
            curr_note = curr_note[:curr_note.find("\n\n")] # trimmed note to end of social history section
        else:
            print("Error! \"Social History\" section not found!")
            exit(0)

        # shifting the start and end values forward after changing the size of the note
        start = train_y.at[i,'start'] - (social_history_start + len("social history:"))
        end = train_y.at[i,'end'] - (social_history_start + len("social history:"))

        train_x.append([curr_note, [(start, end, train_y.at[i,'sbdh'])]])
        counter += 1 # need a way to save the index "i" in case there are more instances of this row_id

    # if this note isn't new, skip the social history processing and instead just add on the label
    else:
        # shifting the start and end values forward after changing the size of the note
        start = train_y.at[i,'start'] - (social_history_start + len("social history:"))
        end = train_y.at[i,'end'] - (social_history_start + len("social history:"))
        train_x[counter][1].append((start, end, train_y.at[i,'sbdh']))
    
    # makes it easier to compare if notes are the same or not
    prev_row_id = curr_row_id

# add each row as individual doc in docbin
print("\nAdding as doc...")
db = DocBin()
counter = 0
for text, annotations in tqdm.tqdm(train_x):
    doc = nlp(text)
    ents = []

    for start, end, label in annotations:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    if(ents[0] is not None): # make sure it isn't empty
        temp = []
        for i in ents:
            if(i is not None):
                temp.append(i)
        doc.ents = spacy.util.filter_spans(temp)
        db.add(doc)

    counter += 1 # increment coutner to keep track of place in train_x

db.to_disk("./train.spacy")

# format testing data to be added to docbin
print("Preparing testing data...")
test_x = []
prev_row_id = -1
counter = -1 # keeps track of which index that data is being taken from

for i in tqdm.tqdm(range(len(test_y))):
    curr_row_id = test_y.at[i,'row_id']
    note_row_id = results_test[results_test['row_id']==curr_row_id].index.values
    curr_note = results_test.at[note_row_id[0],'text']

    # if this note is new, create a new doc
    if(curr_row_id != prev_row_id):
        # need to trim down the note (curr_note) to only include the social history section
        # section seems to always start with "social history:" and end with two newlines ("\n\n")
        social_history_start = curr_note.lower().find("social history:")
        if(social_history_start > 0): # will return -1 if not found
            curr_note = curr_note[social_history_start + len("social history:"):] # start of note trimmed to start of social history section
            curr_note = curr_note[:curr_note.find("\n\n")] # trimmed note to end of social history section
        else:
            print("Error! \"Social History\" section not found!")
            exit(0)

        # shifting the start and end values forward after changing the size of the note
        start = test_y.at[i,'start'] - (social_history_start + len("social history:"))
        end = test_y.at[i,'end'] - (social_history_start + len("social history:"))

        test_x.append([curr_note, [(start, end, test_y.at[i,'sbdh'])]])
        counter += 1 # need a way to save the index "i" in case there are more instances of this row_id

    # if this note isn't new, skip the social history processing and instead just add on the label
    else:
        # shifting the start and end values forward after changing the size of the note
        start = test_y.at[i,'start'] - (social_history_start + len("social history:"))
        end = test_y.at[i,'end'] - (social_history_start + len("social history:"))
        test_x[counter][1].append((start, end, test_y.at[i,'sbdh']))
    
    # makes it easier to compare if notes are the same or not
    prev_row_id = curr_row_id

# add each row as individual doc in docbin
print("\nAdding as doc...")
db_test = DocBin()
counter = 0
for text, annotations in tqdm.tqdm(test_x):
    doc = nlp(text)
    ents = []

    for start, end, label in annotations:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    if(ents[0] is not None): # make sure it isn't empty
        temp = []
        for i in ents:
            if(i is not None):
                temp.append(i)
        doc.ents = spacy.util.filter_spans(temp)
        db_test.add(doc)

    counter += 1 # increment coutner to keep track of place in train_x

db_test.to_disk("./test.spacy")

# -----
# Done!
# -----
print("Finished loading data! Please run the training command now.")
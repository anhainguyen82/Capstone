#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:27:53 2019

@author: andyho
"""

import tensorflow as tf
import tensorflow_hub as hub
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import requests; print("Requests", requests.__version__)
import pandas as pd; print("Pandas", pd.__version__)
import numpy as np; print("Numpy", np.__version__)
import nltk; print("NLTK", nltk.__version__)
import re; print("Re", re.__version__)
import os
import nltk; print("nltk", nltk.__version__)
import spacy; print("spacy", spacy.__version__)
import unidecode
import unicodedata
import pattern; print ("pattern", pattern.__version__)
import string
import time
import pickle
print (os.environ['CONDA_DEFAULT_ENV'])

## Data Analyst Embeddings
df = pd.read_csv('DataAnalyst_Corpus.csv')
print(df.head())
print(len(df))

tempCorpus = df['Corpus']
dummy = []

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  messages = tempCorpus.to_list()
  output = embed(messages)
 
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    dataAnalyst_embeddings = session.run(output)

## Data Scientist Embeddings
df = pd.read_csv('DataScientist_Corpus.csv')
print(df.head())
print(len(df))

tempCorpus = df['Corpus']
dummy = []

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  messages = tempCorpus.to_list()
  output = embed(messages)
 
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    dataScientist_embeddings = session.run(output)

## Data Engineer Embeddings
df = pd.read_csv('DataEngineer_Corpus.csv')
print(df.head())
print(len(df))

tempCorpus = df['Corpus']
dummy = []

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  messages = tempCorpus.to_list()
  output = embed(messages)
 
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    dataEngineer_embeddings = session.run(output)

## Software Engineer Embeddings
df = pd.read_csv('SoftwareEngineer_Corpus.csv')
print(df.head())
print(len(df))

tempCorpus = df['Corpus']
dummy = []

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  messages = tempCorpus.to_list()
  output = embed(messages)
 
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    softwareEngineer_embeddings = session.run(output)
    
## Statistician Embeddings
df = pd.read_csv('Statistician_Corpus.csv')
print(df.head())
print(len(df))

tempCorpus = df['Corpus']
dummy = []

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  messages = tempCorpus.to_list()
  output = embed(messages)
 
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    statistician_embeddings = session.run(output)
    
## Database Admin Embeddings
df = pd.read_csv('Database_Corpus.csv')
print(df.head())
print(len(df))

tempCorpus = df['Corpus']
dummy = []

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  messages = tempCorpus.to_list()
  output = embed(messages)
 
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    databaseAdmin_embeddings = session.run(output)

np.savetxt("dataAnalyst_FV.csv", dataAnalyst_embeddings, delimiter=",")
np.savetxt("dataScientist_FV.csv", dataScientist_embeddings, delimiter=",")
np.savetxt("dataEngineer_FV.csv", dataEngineer_embeddings, delimiter=",")
np.savetxt("statistician_FV.csv", statistician_embeddings, delimiter=",")
np.savetxt("softwareEngineer_FV.csv", softwareEngineer_embeddings, delimiter=",")
np.savetxt("database_FV.csv", databaseAdmin_embeddings, delimiter=",")



'''
tdf = pd.DataFrame(databaseAdmin_embeddings)
tdf.to_csv(index=False, header=False)

df['Feature_Vector'] = pd.Series(databaseAdmin_embeddings)

df.append(pd.DataFrame(databaseAdmin_embeddings, columns=df.columns))


# Add Feature Vectors to dataframe
df['Feature_Vector'] = message_embeddings
print(dummy)
print(df)

# Export dataframe as pickle
temp.to_pickle('my_df.pickle')
'''
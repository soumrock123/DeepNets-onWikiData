# DeepNets-onWikiData
*********Basic deep learning implementations for toy problems using Wiki Data*********

A beginners guide to deep learning algorithms containing concepts about Embeddings, Convolutional Neural Nets, RNN, Activation Functions, Epochs, Dense Layer, and so on. In order to have a hands-on understanding it is important to implement these concepts on considerably simple data. For this project we use the data out of the following paper - "Learning to generate one-sentence biographies from Wikidata", EACL 2017 [https://aclanthology.info/pdf/E/E17/E17-1060.pdf].

The data contains structured information about Wiki biography pages - attribute-value pairs, and the first Wikipedia sentence about the subject. For example -

TITLE mathias tuomi 
SEX OR GENDER male 
DATE OF BIRTH 1985-09-03
OCCUPATION squash player
CITIZENSHIP finland
Mathias Tuomi, (born September 30, 1985 in Espoo) is a professional squash player who represents Finland.

This project is divided into two parts:
1. Attribute prediction from firt Wiki sentence
2. First Wiki sentence generation from attributes

This project has basic implementations of CNN, RNN (vanilla and LSTM), and various combinations thereof.
We used Keras with TensorFlow backend.
We used GloVe vectors for the word embeddings, which can be downloaded from here: https://nlp.stanford.edu/projects/glove/

## Attribute prediction from first Wiki sentence

The file cnn_eacl17gen.py solves the problem of predicting any single attribute (for example SEX OR GENDER) from the Wiki sentence. CNN performs better than RNN in this problem because local features capture the attributes efficiently, and no long-term dependencies are required.

## First Wiki sentence generation from attributes

This is a sequence-to-sequence problem addressed in the EACL paper cited above.
We will soon upload a simple implementation for this problem.


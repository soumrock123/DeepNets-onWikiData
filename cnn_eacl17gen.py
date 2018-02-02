__author__ = 'Debjit, Sreyasi, Soumya'
"""
This class uses Convolutional Neural Network to classify the gender 
"""
import keras
import json
import numpy as np
import sklearn
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences, sequence
from keras import layers
from keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout, MaxPooling1D, Convolution1D, Embedding
from keras.models import Sequential, Model
from keras.layers.merge import Concatenate
from sklearn.metrics import classification_report, f1_score
import itertools 
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
import argparse

#-------------------------------------------------------------------------------------------------------------------
class Convolution(object):
       def __init__(self,X_train,y_train,X_val, y_val,target_class,embedding_matrix, attr2index, term2index, dim_embedding):
                            self.X_train = X_train
                            self.y_train = y_train
                            self.X_val = X_val
                            self.y_val = y_val
                            self.target_class = target_class
                            self.embedding_matrix = embedding_matrix
                            self.attr2index = attr2index
                            self.term2index = term2index
                            self.dim_embedding = dim_embedding

       def train_CNN(self,X_train,y_train,X_val, y_val,target_class,embedding_matrix, attr2index, term2index, dim_embedding):
            '''
              @Input : Training and Validation data, and target class attribute 
              @Input_type : 2D Numpy array of shape, string
              @Output : trained model
              @Output_type : sequential model
            '''
            filter_sizes = (3,7)
            num_filters = 10
            inputX = Input(shape=(35,), name="X")
            z = Embedding(embedding_matrix.shape[0], dim_embedding, weights=[embedding_matrix], trainable=False)(inputX)
            conv_blocks = []
            for sz in filter_sizes:
                  conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
                  conv = MaxPooling1D(pool_size=2)(conv)
                  conv = Flatten()(conv)
                  conv_blocks.append(conv)

            z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
            z = Dense(50, activation="relu")(z)
            softmax = Dense(len(attr2index[target_class]), activation='softmax', name='attr_prob')(z)

            model = Model(inputs=inputX, outputs=softmax)
            model.compile(optimizer='adam', loss='categorical_crossentropy')

            hist = model.fit(X_train, y_train[target_class],
                 validation_data=(X_val, y_val[target_class]),
                 batch_size=32, epochs=5, shuffle=True)
            return(model)
#-------------------------------------------------------------------------------------------------------------------

def test_CNN(X_test, model):
          '''
           @Input : Test data and the model
           @Input_type : 2D Numpy array of shape, sequential model
           @Output : predicted probability of the classes, predicted highest probability
           @Output_type : list,float
          '''
          probs = model.predict(X_test)
          preds = np.argmax(probs, axis=-1)
          return(probs,preds)

def score(probs,preds,y_test, target_class, attr2index):
          '''
           @Input : predicted probability of the classes, highest probability, Test class data
           @Input_type : list,float,list
           @Output : F1 score 
           @Output_type : float
          '''

          true = np.argmax(y_test[target_class], axis=-1)
          names, labels = zip(*attr2index[target_class].items())
          print(classification_report(true, preds, labels=labels, target_names=names))
          print('Macro F1:', f1_score(true, preds, average='macro'))

#-------------------------------------------------------------------------------------------------------------------
def preprocessing(data, attr2index, term2index, maxlen=35):
    '''
    @Input : Data in a json format
    @Input_type : dictionary
    @Output : One hot representation
    @Output_type : 2D Numpy array of shape, dictionary 
    '''

    X = []
    y = {attr: [] for attr in attr2index}
    for attrs, sent in data:
        termids = [term2index.get(term, term2index['UNK']) for term in sent.split(' ')]
        X.append(termids)
        
        for attr in attr2index:
            onehot_vec = np.zeros(len(attr2index[attr]))
            if attr in attrs:
                val = attrs[attr]
                if val in attr2index[attr]:
                    onehot_vec[attr2index[attr][val]] = 1
            y[attr].append(onehot_vec)
            
    X = pad_sequences(X, padding='post', truncating='post', maxlen=maxlen)
    y = {k: np.array(v) for k, v in y.items()}
    return X, y

#--------------------------------------------------------------------------------------------------------------

def fetch_embedding(data_path, pre_embedding_path, dim_embedding):
    '''
    @Input : Path of the data and the embedding as a string, dimension of the embedding
    @Input_type : string, string, int
    @Output : Embedding of the data (Using Pre-trained 'GloVe: Global Vectors for Word Representation') 
    @Output_type : List of list  
    '''

    data = json.load(open(data_path, 'rt'))
    wp_vocab = set(token for attrs, sent in data['train'] for token in sent.split(' '))
    valid_attrs = ['#SEX_OR_GENDER', '#GIVEN_NAME', '#OCCUPATION', '#COUNTRY_OF_CITIZENSHIP']
    attr_vocabs = {}
    for attrs, sent in data['train']:
       for attr, v in attrs.items():
           if attr in valid_attrs:
              attr_vocabs.setdefault(attr, set()).add(v)
            
    attr2index = {attr: {term: i for i, term in enumerate(sorted(terms))}
            for attr, terms in attr_vocabs.items()}
        
    unk = np.random.uniform(-0.2, 0.2, dim_embedding)
    embeddings = {'UNK': unk}
    for line in open(pre_embedding_path, 'rt'):
          fields = line.strip().split(' ')
          token = fields[0]
          if token in wp_vocab:
                  embeddings[token] = np.array([float(x) for x in fields[1:]])
        
    embedding_matrix = np.zeros((len(embeddings) + 1, dim_embedding))
    term2index = {term: i+1 for i, term in enumerate(sorted(embeddings))}
    for term, i in term2index.items():
          embedding_matrix[i] = embeddings[term]
    return(embedding_matrix, data, attr2index,term2index)


#-------------------------------------------------------------------------------------------------------------------
def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path',default='/data/users/dpaul/Thesis/Practical/Project/eacl17_filtered.json',help="training, dev and testing data file containing descriptions in json format")
  parser.add_argument('--pre_embedding_path',default='/data/users/dpaul/Thesis/Practical/Project/glove.6B.300d.txt',help="pretrained embeddings")
  parser.add_argument('--dim_embedding',default=300,help="embedding dimension")
  parser.add_argument('--target_class',default='#SEX_OR_GENDER',help="training labels") 
  args = parser.parse_args()
  embedding_matrix, data, attr2index, term2index = fetch_embedding(args.data_path, args.pre_embedding_path, args.dim_embedding)
  X_train, y_train = preprocessing(data['train'], attr2index, term2index)
  X_val, y_val = preprocessing(data['dev'], attr2index, term2index)
  X_test, y_test = preprocessing(data['train'],attr2index, term2index)
  cnn= Convolution(X_train,y_train,X_val, y_val, args.target_class,embedding_matrix, attr2index, term2index, args.dim_embedding)
  model = cnn.train_CNN(X_train,y_train, X_val, y_val, args.target_class, embedding_matrix, attr2index, term2index, args.dim_embedding)
  probs, preds = test_CNN(X_test, model)
  score(probs,preds,y_test, args.target_class,attr2index)



if __name__ == '__main__':
    main()


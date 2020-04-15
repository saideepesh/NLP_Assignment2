
import pandas as pd
from transformers import *
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Classifier:
    """The Classifier : The preprocess is to extract bert representation of inputs, and then to train a
    Dense NN on this data"""
    def __init__(self):
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.clf = self.build_clf()
        self.labels2nb = {'positive' : 0,
             'neutral' : 2,
             'negative' : 1}
        self.nb2labels = {v : k for k,v in self.labels2nb.items()}

    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        inputs, labels = self.load_data(trainfile)
        clf_inputs = self.get_bert_encoding(*inputs)
        labels = [self.labels2nb[l] for l in labels]
        X_train, X_val, Y_train, Y_val = train_test_split(clf_inputs, labels, test_size = 0.1)
        cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        self.clf.fit(np.array(X_train), np.array(Y_train), epochs = 500, validation_data=(np.array(X_val), np.array(Y_val)), callbacks = [cb])
        

        


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        inputs, labels = self.load_data(datafile)
        clf_inputs = self.get_bert_encoding(*inputs)
        predictions_probas = self.clf.predict(np.array(clf_inputs))
        predictions = [np.argmax(pred) for pred in predictions_probas]
        predictions_labels = [self.nb2labels[l] for l in predictions]
        return predictions_labels
        
        
    def preprocess_data(self, data):
        """
        

        Parameters
        ----------
        data : List[str]
            A list of string 

        Returns
        -------
        list
            A list of inputs for BERT

        """
        inputs_ids = []
        attention_mask = []
        token_type_ids = []
        for sentence in data:
            encoded = self.bert_tokenizer.encode_plus(sentence, max_length = 32, pad_to_max_length = True)
            inputs_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])
            token_type_ids.append(encoded['token_type_ids']) 
        return [np.array(inputs_ids), np.array(attention_mask), np.array(token_type_ids)]
    
    def get_bert_encoding(self, sentences, terms, aspects):
        """
        Takes the sentences, aspect term and aspect category in order to output the BERT representation.

        Parameters
        ----------
        sentences : List[str]
            The sentences
        terms : List[str]
            The aspect terms
        aspects : List[str]
            The aspect categories

        Returns
        -------
        final_encoding : np.ndarray
            BERT encoding of the data, with shape (len(sentences), 96, 768)

        """
        sentences_inputs = self.preprocess_data(sentences)
        terms_inputs = self.preprocess_data(terms)
        aspects_inputs = self.preprocess_data(aspects)
        sentence_representation = self.bert_model.predict(sentences_inputs)[0]
        terms_representation = self.bert_model.predict(terms_inputs)[0]
        aspects_representation = self.bert_model.predict(aspects_inputs)[0]
        final_encoding = np.concatenate((sentence_representation, terms_representation, aspects_representation), axis = 1)
        return final_encoding
    
    def build_clf(self):
        """
        Build the classifier

        Returns
        -------
        clf : keras model
            The classifier

        """
        clf = tf.keras.Sequential()
        clf.add(tf.keras.layers.Dense(768,input_shape = (96,768) ))
        clf.add(tf.keras.layers.Dense(512 ))
        clf.add(tf.keras.layers.Dropout(0.4))
        clf.add(tf.keras.layers.Dense(256))
        clf.add(tf.keras.layers.Flatten())
        clf.add(tf.keras.layers.Dropout(0.3))
        clf.add(tf.keras.layers.Dense(64))
        clf.add(tf.keras.layers.Dropout(0.3))
        clf.add(tf.keras.layers.Dense(3, activation = 'softmax'))
        opt = tf.keras.optimizers.Adam(0.001, clipvalue = 0.000000005)
        clf.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        return clf
        

    
    @staticmethod  
    def load_data(file):
        data = pd.read_csv(file, sep='\t', names = ['Polarity', 'Aspect category','Aspect term', 'offset', 'sentence'])
        labels = data['Polarity'].values
        sentences = data['sentence'].values
        terms = data['Aspect term'].values
        aspects = data['Aspect category'].values
        return [sentences, terms, aspects], labels
    
    
    



import sys
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

import pickle


def load_data(database_filepath):
    '''
    This function loads the dataframe from the database where it is stored, and splits it into predictors and outputs dataframes.

    INPUT:
    database_filepath - the name of the file containing the dataframe

    OUTPUT:
    X - predictor dataframe, containing the message text
    Y - output dataframe, containing the message categories
    '''

    # Load data from database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('msg_df', con=engine)

    # Split into X and Y
    X = df.iloc[:, 1]
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    return X, Y


def tokenize(text):
    '''
    This function takes a text as input and performs case normalisation, tokenization and lemmatization on it.
    
    INPUT:
    text - the text to be processed
    
    OUTPUT:
    clean_tokens - list of the cleaned tokens extracted from the input text
    '''

     # Case normalisation
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Initialise lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Create a list comprehension to store the clean tokens
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return clean_tokens  


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
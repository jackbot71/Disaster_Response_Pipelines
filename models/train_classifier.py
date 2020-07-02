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
    '''
    This function builds a model with a pipeline that tokenizes and extracts features from the messages, and creates a multi-output classifier with it using gridsearch.

    INPUT:
    None

    OUTPUT:
    cv - model including text processing and multi-output classifier
    '''
    # Create the feature extraction and modelling pipeline
    pipeline = Pipeline([
    
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Specify the parameters for the grid search
    parameters = {
        'text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    # Create gridsearch object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=100)

    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    This function calculates the precision, recall, f1-score and accuracy, prints and returns them in a dataframe.
    
    INPUT:
    model - the model to evaluate
    X_test - ndarray containing the predictions from the model
    Y_test - dataframe containing the test values
    
    OUTPUT:
    score_df - dataframe containing the precision, recall, f1-score and accuracy for every output
    '''

    # Calculate Y_pred using the model
    Y_pred = model.predict(X_test)

    # Create a dataframe that will hold the precision, recall and f1-score for each column
    score_df = pd.DataFrame(columns=['precision', 'recall', 'f1_score', 'support'])

    # Iterate through the columns to calculate the precision, recall and f1-score and append it as a row to score_df
    for col in range(len(Y_test.columns)):
        col_score = pd.Series(score(np.array(Y_test)[:, col], Y_pred[:, col], average='binary'), index=score_df.columns)
        score_df = score_df.append(col_score, ignore_index=True)

    # Label the index of score_df with the column names
    score_df.index = Y_test.columns

    # Drop the "support" scores, as we are dealing with binary data here (2-class output)
    score_df.drop(columns='support', inplace=True)
    
    # Create empty list to hold accuracy values
    accuracy = []
    
    # Iterate through columns and calculate accuracy
    for col in range(len(Y_test.columns)):
        accuracy.append(accuracy_score(np.array(Y_test)[:, col], Y_pred[:, col]))
    
    # Add accuracy to the score_df
    score_df['accuracy'] = pd.Series(accuracy, index=Y_test.columns)

    # Print score_df
    print(score_df)
    
    return score_df
   

def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
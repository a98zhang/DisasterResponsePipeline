import sys
import os
import nltk
nltk.download(['punkt', 'wordnet','omw-1.4'])

import re
from random import random
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.multioutput import MultiOutputClassifier 

import pickle

def load_data(database_filepath):
    engine = create_engine(''.join(['sqlite:///', os.path.abspath(database_filepath)]))
    df = pd.read_sql_table('data', engine)
    X = df['message']   # id, message, original, genre
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]
    return X, Y, category_names

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()    #considering removing stop words
    return [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]


def build_model():
    pipeline = Pipeline([
        ('textpipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('Tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        #'textpipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    '''
    cv.best_params_
    {'clf__estimator__min_samples_split': 2,
    'clf__estimator__n_estimators': 100,
    'textpipeline__vect__ngram_range': (1, 2)}
    '''

    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)

    reports = dict()
    for i, cat in enumerate(category_names):
        reports[cat] = classification_report(Y_test[cat], Y_pred[:, i], zero_division=0, output_dict=True)
    accuracies = [reports[report]['accuracy'] for report in reports]
    print(np.array(accuracies).mean())


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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
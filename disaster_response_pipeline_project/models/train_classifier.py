# import libraries
import re
# numpy as np
import pandas as pd
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
#from sklearn.ensemble import RandomForestClassifier
import pickle
#import os
import sys

nltk.download('punkt','wordnet')
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'



def load_data(database_filepath):
    '''
    INPUT
    database_filepath - location of the database
    
    OUTPUT
    returns dataframe split into X and Y and a list of column names
    X - 'message' column of database
    Y - every other column
    category_names - list of column names from Y dataframe
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disasterResponse',engine)
        
    X = df['message']
    y = df.iloc[:,4:]

    category_names = y.columns
    return X, y, category_names
    

def tokenize(text):
    '''
    INPUT
    text - a string that will be tokenized
    
    OUTPUT
    Returns a tokenized version of the original string
    clean_tokens - tokenized version
    
    '''
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#Note - While choosing a classifier for this project, I tried using RandomForest, Kneighbors and the SGD.
#       Only the SGD was able to generate a pickle file below 100mb, so I decided to apply the gridsearchcv
#       only to this classifier to search for the best combination of parameters
def build_model():
    '''
    OUTPUT
    Returns a pipeline based for a SGD Classifier model
    
    model - SGD Classifier model
    '''
    pipeline3 = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier()))
        ])
    
    parameters = [{'vect__max_df': (0.5, 0.75, 1.0),
                   'clf__estimator__alpha': (0.00001, 0.000001),
                   'clf__estimator__penalty': ('l2', 'elasticnet')
                  }]
    
    model = GridSearchCV(pipeline3, parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - pipelined model
    X_test - X df split into test and train
    Y_test - Y df split into test and train
    category_names - names of categories of Y df
    
    OUTPUT
    Report the f1 score, precision and recall for each output category of the dataset.
    '''
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    saves the modle in pickle format
    '''
    pickle.dump(model, open( model_filepath, 'wb' ) )
    


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
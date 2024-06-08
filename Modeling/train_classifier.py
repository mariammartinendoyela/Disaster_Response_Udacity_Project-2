import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Load data from an SQLite database.

    Args:
    database_filepath (str): Path to the SQLite database file.

    Returns:
    tuple: A tuple containing:
        - X (pandas.Series): Features for model training.
        - Y (pandas.DataFrame): Target variables for model training.
    """
    # Create a database engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Load data from the database table into a DataFrame
    df = pd.read_sql_table("DisasterResponse", con=engine)
    
    # Extract the 'message' column as the feature set
    X = df['message']
    
    # Extract all columns starting from the fifth column as the target set
    Y = df.iloc[:, 4:]
    
    return X, Y

def tokenize(text):
    """
    Tokenize and lemmatize the input text.

    Args:
    text (str): Input text to tokenize.

    Returns:
    list: List of cleaned tokens.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize and clean each token
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline and tune it using GridSearchCV.

    Returns:
    GridSearchCV: A GridSearchCV object with the specified pipeline and parameters.
    """
    # Define a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Tokenize and vectorize the text
        ('tfidf', TfidfTransformer()),  # Apply TF-IDF transformation
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classifier using Random Forest
    ])
    
    # Define the parameter grid for GridSearchCV
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),  # Use unigrams or bigrams
        'clf__estimator__n_estimators': [50, 100],  # Number of trees in the forest
        'clf__estimator__min_samples_split': [2, 5, 10]  # Minimum number of samples required to split a node
    }
    
    # Initialize GridSearchCV with the pipeline and parameter grid
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)  # Verbose output for logging
    
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model's performance on test data.

    Args:
    model: Trained classifier model.
    X_test: Test feature set.
    Y_test: Test target set.

    Returns:
    None
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Loop through each target column and print the classification report
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
    model: Trained model to save.
    model_filepath (str): Filepath to save the model.
    """
    # Save the model to a file using pickle
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Execute the complete machine learning pipeline: load data, train model, evaluate model, and save model.
    """
    # Ensure correct number of command-line arguments are provided
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        # Load data from the database
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        # Split data into training and test sets
        print('Splitting data into training and test sets...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Build the machine learning model
        print('Building model...')
        model = build_model()
        
        # Train the model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Evaluate the model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        # Save the model to a file
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Model saved successfully!')
    else:
        # Print usage instructions if incorrect arguments are provided
        print('Please provide the filepath of the disaster messages database as the first argument '
              'and the filepath of the pickle file to save the model as the second argument.\n\n'
              'Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()

import json
import plotly
import pandas as pd
from pickle import load

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

class DisasterResponseWebApp:
    """
    Class for running a Flask web application for disaster response message classification.
    """

    def __init__(self):
        """
        Initializes the DisasterResponseWebApp class.
        """
        self.web_app = Flask(__name__)
        self.fetch_data()
        self.fetch_model()
        self.initialize_routes()

    def fetch_data(self):
        """
        Loads data from a SQLite database into a Pandas DataFrame.
        """
        db_engine = create_engine('sqlite:///./Data/DisasterResponse.db')
        self.data_frame = pd.read_sql_table('DisasterResponse', db_engine)

    def fetch_model(self):
        """
        Loads a pre-trained machine learning model.
        """
        filename = 'Modeling/model.pkl'

        # Ensure the directory exists
        import os
        if not os.path.exists('Modeling'):
            os.makedirs('Modeling')

        # Load the model using pickle
        with open(filename, 'rb') as file:
            self.classification_model = load(file)

    def tokenize(self, text):
        """
        Tokenizes and lemmatizes text.

        Args:
        text (str): The input text to tokenize.

        Returns:
        list: A list of tokens.
        """
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        cleaned_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
        return cleaned_tokens

    def initialize_routes(self):
        """
        Sets up Flask routes for the web application.
        """
        @self.web_app.route('/')
        @self.web_app.route('/index')
        def home():
            """
            Renders the main page of the web application with data visualizations.
            """
            genre_message_counts = self.data_frame.groupby('genre').count()['message']
            genre_names = list(genre_message_counts.index)
            df_without_text = self.data_frame.drop(['id', 'message', 'original', 'genre'], axis=1)
            category_message_counts = df_without_text.sum(axis=0)
            category_names = df_without_text.columns
            visualizations = [
                {
                    'data': [
                        Bar(
                            x=genre_names,
                            y=genre_message_counts
                        )
                    ],
                    'layout': {
                        'title': 'Distribution of Message Genres',
                        'yaxis': {
                            'title': "Count"
                        },
                        'xaxis': {
                            'title': "Genre"
                        }
                    }
                },
                {
                    'data': [
                        Bar(
                            x=category_names,
                            y=category_message_counts
                        )
                    ],
                    'layout': {
                        'title': 'Distribution of Message Categories',
                        'yaxis': {
                            'title': "Count"
                        },
                        'xaxis': {
                            'title': "Category"
                        }
                    }
                }
            ]
            graph_ids = ["graph-{}".format(i) for i, _ in enumerate(visualizations)]
            graph_JSON = json.dumps(visualizations, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('master.html', ids=graph_ids, graphJSON=graph_JSON)

        @self.web_app.route('/go')
        def classify_message():
            """
            Handles user query and displays model results.
            """
            user_query = request.args.get('query', '')
            classification_labels = self.classification_model.predict([user_query])[0]
            classification_results = dict(zip(self.data_frame.columns[4:], classification_labels))
            return render_template('go.html', query=user_query, classification_result=classification_results)

    def run_server(self):
        """
        Runs the Flask web application.
        """
        self.web_app.run(host='0.0.0.0', port=1309, debug=True)

if __name__ == '__main__':
    app = DisasterResponseWebApp()
    app.run_server()

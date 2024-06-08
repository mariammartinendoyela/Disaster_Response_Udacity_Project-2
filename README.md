# Disaster Response Pipeline Project

### Project Objective
The project involved analyzing disaster data from Appen and developing an API to classify real disaster messages. This required creating an ETL pipeline, a ML pipeline, and a Web app to categorize messages during disasters, ensuring swift and accurate routing to appropriate relief agencies and displaying data visualizations. The goal was to streamline disaster response efforts through advanced data engineering and real-time processing.

### Files
Files
1. App
- master.html (main page)
- go.html (classification result)
- run.py (file that runs app)

2. Data
- categories.csv (Data to categories)
- messages.csv (Data to messages)
- process_data.py (Data to exploration and cleaning)
- DisasterResponse.db (database to save clean data - Table name DisasterResponse)

3.	Modeling
- train_classifier.py (ML pipeline)    
- model.pkl (saved model)

### README.md    

1. ETL Pipeline
The `process_data.py` Python script implements a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores the cleaned data in a SQLite database
A Jupyter notebook was used for the ETL pipeline preparation, performing exploratory data analysis (EDA) to aid in writing the `process_data.py` script.

2. ML Pipeline
The `train_classifier.py` Python script creates a machine learning pipeline that:

- Loads data from the SQLite database and splits it into training and test sets
- Constructs a text processing and machine learning pipeline, training and tuning a model using GridSearchCV
- Outputs test set results and saves the final model as a pickle file
A Jupyter notebook was used for ML pipeline preparation, conducting EDA to assist in developing the `train_classifier.py` script.

3. Flask Web App
The Flask web application allows emergency workers to input new messages and receive classifications along with data visualizations. The web app also displays visualizations of the data.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To execute an ETL pipeline for data cleaning and database storing
        `python Data/process_data.py Data/messages.csv Data/categories.csv Data/DisasterResponse.db`
    - ML pipeline to be run in order to train the classifier and save
        `python Modeling/train_classifier.py Data/DisasterResponse.db Modeling/model.pkl`

2. To run your web application, enter the following command in the app's directory.
    `python run.py`

3. Go to http://0.0.0.0:1309/



# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Author, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The project code is using numpy, pandas and sklearn for data processing and model training. The nltk package is used as well for processing the text messages and for feature extraction. To save the processed data in a SQLite data file, the SQLAlchemy package is used.

In the second part of the project, flask and plotly are used to create a web app displaying some information about the training data, as well as allowing a user to input a message to be classified by the model trained in the first part of the project.

After cloning the files in the repo, the model can be trained and the web app started by following these steps in the command line:

- Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db YourTableName_df`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db YourTableName_df models/classifier.pkl`

- Run the following command in the app's directory to run your web app.
    `python run.py`

- Go to http://0.0.0.0:3001/

## Project Motivation <a name="motivation"></a>

This project is part of the Udacity Data Scientist Nanodegree, and consists of two main parts:

- Use 2 files, one containing some messages sent in the wake of emergency situations, and another with classification of these messages by type of disaster that created the emergency situation, to train and optimize a model able to classify the type of disaster situation from NLP of a text message.

- Using flask, create a web interface that can display some information about the training dataset, as well as handle a query from a user to classify a message into the disaster categories that the model from the first part was trained on.

## File Descriptions <a name="files"></a>

There are 3 folders in this repository:

- "data" contains the raw data files (disaster_messages.csv and disaster_categories.csv), the code to extract the data from them and clean it (process_data.py) and the resulting SQLite database file (DisasterResponse.db).

- "models" contains the code to process the text, extract the features, create a pipeline and train an optimize a multi-output classifier model (train_classifier.py), and a pickle file for the saved model (classifier.pkl).

- "app" contains the files necessary for running the web app, including a run.py script to create and export the graphs, as well as use the model for the classification of the user query, and 2 html files - one for the main page of the app, with the graphs, and the other to display the results of the user query.

## Results <a name="results"></a>

Using the code and the data provided in the repository, a multi-output classification model using the random forest ensemble method was created. This model was trained on a split of 80% of the data provided, and tested on the other 20% as a measure of its performance, using several metrics like precision, recall, f1-score and accuracy.

Some tuning of the parameters of the model was done using GridSearchCV, namely on the use of tfidf and the minimum sample split for the random forest. The tested parameters were fairly limited because of considerations of computational power and run time, but with a more powerful machine, a better optimisation could most likely be found.

The model obtained here was nevertheless good enough for the purpose of this exercise.

In addition, it is worth noting that some categories were severely underrepresented in the data, like the child-alone category: of course, this impacts the ability of the model to classify messages in that category. Some messages from this category should be added to the data, maybe even "fake" messages generated by a group of people could help the model to improve its predictions in this category (though there is likely to be some degree of bias in the model depending on the nature and heterogeneity of the group of people generating the fake messages).

## Licensing, Author and Acknowledgements <a name="licensing"></a>

A big thanks to Figure8 for providing the data, and to Udacity for laying the foundations of the project, including the html code and most of the run.py. 

The code from the two python scripts to process the data and train the model can be re-used at your convenience.



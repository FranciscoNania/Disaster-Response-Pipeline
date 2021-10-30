# Project2-Udacity
# Disaster Response Pipeline

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files Description
</pre>
├── app

│   ├── run.py                           Flask file that runs app

│   └── templates

│       ├── go.html                      Classification result page of web app

│       └── master.html                  Main page of web app

├── data

│   ├── disaster_categories.csv          Dataset including all the categories

│   ├── disaster_messages.csv            Dataset including all the messages

│   └── process_data.py                  Data cleaning

├── models

│   ├── train_classifier.py              Train ML model

│   └── classifier.pkl                   pikkle file of model

|   

|── requirements.txt                     contains versions of all libraries used.

|

└── README.md
</pre>
## Project Motivation
The main purpose of this project was developing my skills as a Data Scientist



## Acknowledments
Credit to Udacity for a great Data Science course and several tips!


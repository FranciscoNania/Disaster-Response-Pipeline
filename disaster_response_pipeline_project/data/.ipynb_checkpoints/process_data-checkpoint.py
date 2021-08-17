import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    '''
    load messages dataset and merge both datasets
    
    INPUTS
    messages_filepath - disaster_messages.csv file
    categories_filepath - categories.csv file

    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df


def clean_data(df):
    '''
    
    Split into separate categories
    convert category values to binary
    replace original categories column
    drop duplicates
    
    INPUT
    df - dataframe
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    categories = categories.astype(str)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x.strip()[-1] for x in categories[column]]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    Save the clean dataset into an sqlite database
    
    INPUT
    df - cleaned dataframe
    database_filename - 'ETLPipeline'
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql(database_filename, engine, index=False)
     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        #print(df)
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
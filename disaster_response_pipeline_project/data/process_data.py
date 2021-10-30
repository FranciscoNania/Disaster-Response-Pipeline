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
    #The empty 'child_alone' column conflicts with the chosen classifier later on, so I dropped it.
    df = df.drop(['child_alone'],axis=1)

    #Some values on dataframe are not binary, so a small lambda function to change '2' into '1'. 
    #Those values could have been dropped as well, but it seems they are supposed to be '1'
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    
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
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disasterResponse', engine, index=False, if_exists = 'replace')
     


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
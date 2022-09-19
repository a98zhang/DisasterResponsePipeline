# import libraries
import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Extract data and returns a dataset that includes messages and their categories
            
        Parameters:
            messages_filepath (str): file path of the message data
            categories_filepath (str): file path of the category data

        Returns:
            df (dataframe): Pandas dataframe of the merged data   
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    '''
    Create category binary indications and return clean data
            
        Parameters:
            df (dataframe): Pandas dataframe of the merged data 

        Returns:
            df (dataframe): cleaned and de-duped version of the dataset   
    '''

    categories = df['categories'].str.split(';', expand=True)
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2])
    for column in categories:
        categories[column] = categories[column].astype("string").str[-1].astype(int)
    df = pd.concat([df.drop(columns=['categories']), categories], axis=1)
    return df.drop_duplicates()


def save_data(df, database_filename):
    '''
    Load the clean data into database
            
        Parameters:
            df (dataframe): Pandas dataframe of the clean data
            database_filename (str): database file path to which the dataset is loading
        Returns:  
    '''
    engine = create_engine(''.join(['sqlite:///', os.path.abspath(database_filename)]))
    df.to_sql('data', con=engine, if_exists='replace', index=False)


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function imports the csv files for the messages and their associated categories, and combines them in one dataframe.
    
    INPUT:
    messages_filepath - path to the csv file containing the messages
    categories_filepath - path to the csv file containing the categories for each message
    
    OUTPUT:
    df - dataframe combining the messages and the raw data for their associated categories
    '''
    
    # Import message data
    messages = pd.read_csv(messages_filepath)
    
    # Import category data
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two datasets together using the id
    df = pd.merge(messages, categories, on='id')
        
    return df

def clean_data(df):
    '''
    This function takes the dataframe containing the combined message and category data, encodes the categories as binary columns, and removes duplicates and misclassified rows.

    INPUT:
    df - dataframe combining the messages and the raw data for their associated categories

    OUTPUT:
    clean_df - the dataframe with the categories encoded as binary columns, and duplicates and misclassified rows removed
    '''

    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0, :]
    
    # Use this row to extract a list of new column names for categories
    category_colnames = [x[0:-2:1] for x in row]
    
    # Use this list to rename the columns of the categories dataframe
    categories.columns = category_colnames
    
    # Convert category values to just 0s and 1s
    for column in categories.columns:
        
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(start=-1)
    
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Drop the original categories column from df
    df.drop(columns=['categories'], inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop the rows where related values equal 2
    df = df.drop(index=df[df['related'] == 2].index)

    # Drop duplicates
    clean_df = df.drop(index=df[df.duplicated()].index)

    return clean_df


def save_data(df, database_filename):
    pass  


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
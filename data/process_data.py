import sys
import pandas as pd
from sqlalchemy import create_engine


def extract_data(messages_filepath, categories_filepath):
    """
    A function to extract the 'messages' and 'categories' datasets 
    
    Args: 
        messages_filepath: path to CSV file
        categories_filepath: path to CSV file
    Returns:
        Single dataframe containing both raw datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #both datasets have a common column 'id' --> can be merged
    return messages.merge(categories, on='id')


def transform_data(df):
    """
    A function to clean/transform the inputted dataframe by sorting column headings and
    removing duplicated data.
    
    Args: 
        df: a dataframe of the raw 'messages' and 'categories' datasets
    Returns:
        df: same data after cleaning
    """

    # split 'categories' into separate columns:
    split_df = df['categories'].str.split(pat=';',expand=True)

    # making text into column headings by iterating over the 1st row:
    split_df.columns = [i.split('-')[0] for i in split_df.loc[0]]

    # removing text from split_df contents --> just numerical values:
    for col in split_df.columns:
        x = {col+'-1': 1, col+'-0': 0}
        split_df[col] = split_df[col].map(x)
        
    #drop unneccesary columns and merge original df with split_df:
    df = df.drop(columns=['categories','original','id'])
    df = pd.concat([df,split_df],axis=1)
    
    # checking for duplicates
    #print (df[df.duplicated()].shape) #170 rows are affected
    df = df.drop_duplicates()
    #print (df[df.duplicated()].shape) #checking are removed

     #TODO remove nans and change column to a single dtype
    return df


def load_data(df, database_filename):
    """
    A function to load the cleaned data to a SQL database.
    
    Args: 
        df: a dataframe of the cleaned 'messages' and 'categories' data
        database_filename
    """

    engine = create_engine('sqlite:///'+database_filename+'.db')
    df.to_sql('Disaster_messages_categories', engine, index=False)
    
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print ('Commence ETL pipeline')
        
        print('Extracting raw data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = extract_data(messages_filepath, categories_filepath)

        print('Transforming data...')
        df = transform_data(df)
        
        print('Loading data to...\n    DATABASE: {}'.format(database_filepath))
        load_data(df, database_filepath)
        
        print('Cleaned data saved to database')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'raw/disaster_messages.csv raw/disaster_categories.csv '\
              'processed/DisasterResponse.db')


if __name__ == '__main__':
    main()
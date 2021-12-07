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

def combining_3_columns_into1(df, colname1, colname2, colname3, newcolname):
    df[newcolname] = df[colname1] + df[colname2] + df[colname3]
    df[newcolname] = [1 if new_m >= 1 else 0 for new_m in df[newcolname]] #changing 2 and 3 back to 1
    df = df.drop([colname1, colname2, colname3],axis=1)
    return df

def combining_2_columns_into1(df, colname1, colname2, newcolname):
    df[newcolname] = df[colname1] + df[colname2] 
    df[newcolname] = [1 if new_m >= 1 else 0 for new_m in df[newcolname]] #changing 2 back to 1
    df = df.drop([colname1, colname2],axis=1)
    return df

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
        
    # removing meaningless and empty columns:
    split_df = split_df.drop(['related', 'request', 'offer','direct_report', 'child_alone'],axis=1)
    
    #combining categories with similar meanings
    split_df = combining_3_columns_into1(split_df, 'aid_related','other_aid','aid_centers','aid_needed')
    split_df = combining_2_columns_into1(split_df, 'infrastructure_related','other_infrastructure','infrastructure')
    split_df = combining_2_columns_into1(split_df, 'weather_related','other_weather','weather')
    split_df = combining_3_columns_into1(split_df, 'medical_products','medical_help','hospitals','medical')
    
    #Imbalanced categories --> can aid_needed or weather be removed? Are they independent?
    #'aid needed' is dropped from the category list as it's not directly helpful as all categories require aid (weather is kept)
    split_df = split_df.drop(['aid_needed'],axis=1)
    split_df.shape # now 23 independent categories

    #drop unneccesary columns and merge original df with split_df:
    df = pd.concat([df,split_df],axis=1)
    df = df.drop(['categories','original','id'],axis=1)
    
    # checking for duplicates
    #print (df[df.duplicated()].shape) #170 rows are affected
    df = df.drop_duplicates()
    #print (df[df.duplicated()].shape) #checking are removed

    return df


def load_data(df, database_filename):
    """
    A function to load the cleaned data to a SQL database.
    
    Args: 
        df: a dataframe of the cleaned 'messages' and 'categories' data
        database_filename
    """

    engine = create_engine('sqlite:///processed/DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)
    
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
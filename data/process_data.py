## import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
from pandas.io import sql


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    ## create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    ## select the first row of the categories dataframe
    row = categories.iloc[0]

    ## Extract column names from row with str slicing 
    category_colnames = row.apply(lambda x: x[:-2])

    ## rename the columns of `categories`
    categories.columns = category_colnames

    ## Convert category values to just numbers 0 or 1.
    for column in categories:
        ## set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)

        ## convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    ## drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    ## concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    ## When creating the classification reports, it was noticed that the related column has an additional label with the value 2. 
    ## This is not the case with the other columns. After searching the mentor's help, I found that others had the same problem as well.
    ## For all rows with the value related = 2, all other columns have the value 0, which also applies to the related = 0 rows. 
    ## For this reason I will change the value from 2 to 0.
    df.related.replace(2, 0, inplace=True)

    ## drop duplicates
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):
    ## create database engine to access database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    sql.execute('DROP TABLE IF EXISTS %s'%'labeled_messages', engine)

    df.to_sql('labeled_messages', engine, index=False)  


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
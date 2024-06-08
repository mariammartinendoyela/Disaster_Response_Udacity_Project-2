import pandas as pd
from sqlalchemy import create_engine

class DataProcessor:
    def __init__(self):
        pass

    def read_and_merge_data(self, messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
        """
        Reads messages and categories from CSV files and merges them.

        Parameters:
            messages_filepath (str): Path to the messages CSV file.
            categories_filepath (str): Path to the categories CSV file.

        Returns:
            pd.DataFrame: Merged DataFrame containing messages and categories.
        """
        # Load datasets
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)

        # Merge datasets on common id
        df = messages.merge(categories, how='outer', on='id')
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by splitting categories into separate columns,
        converting category values to binary, and removing duplicates.

        Parameters:
            df (pd.DataFrame): DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        # Split categories into separate columns
        categories = df['categories'].str.split(';', expand=True)

        # Extract category names
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x: x[:-2])

        # Rename the columns of categories DataFrame
        categories.columns = category_colnames

        # Convert category values to binary
        for column in categories:
            # Set each value to be the last character of the string
            categories[column] = categories[column].astype(str).str[-1]
            # Convert column from string to numeric
            categories[column] = categories[column].astype(int)

        # Replace values of 2 with 1 in 'related' column
        categories['related'] = categories['related'].replace(2, 1)

        # Drop the original categories column from df
        df.drop('categories', axis=1, inplace=True)

        # Concatenate the original DataFrame with the new categories DataFrame
        df = pd.concat([df, categories], axis=1)

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        return df

    def upload_data(self, df: pd.DataFrame, database_filepath: str) -> None:
        """
        Stores the DataFrame in a SQLite database.

        Parameters:
            df (pd.DataFrame): DataFrame to be stored.
            database_filepath (str): Path for the SQLite database.
        """
        engine = create_engine(f'sqlite:///{database_filepath}')
        df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

    def process_data(self, messages_filepath: str, categories_filepath: str, database_filepath: str) -> None:
        """
        Orchestrates the loading, cleaning, and storing of data into a database.

        Parameters:
            messages_filepath (str): Path to the messages CSV file.
            categories_filepath (str): Path to the categories CSV file.
            database_filepath (str): Path for the SQLite database.
        """
        # Load data
        print('Reading and merging data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = self.read_and_merge_data(messages_filepath, categories_filepath)

        # Clean data
        print('Cleaning data...')
        df = self.clean_data(df)

        # Save data
        print('Uploading data...\n    DATABASE: {}'.format(database_filepath))
        self.upload_data(df, database_filepath)

        print('Cleaned data saved to database!')

# Example usage:
if __name__ == '__main__':
    processor = DataProcessor()
    messages_filepath = input("Enter the filepath for the messages CSV file: ")
    categories_filepath = input("Enter the filepath for the categories CSV file: ")
    database_filepath = input("Enter the filepath for the SQLite database: ")
    processor.process_data(messages_filepath, categories_filepath, database_filepath)

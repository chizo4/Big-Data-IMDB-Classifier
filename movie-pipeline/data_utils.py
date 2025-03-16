'''
--------------------------------------------------------------
FILE:
    movie-pipeline/data_utils.py

INFO:
    Utily class for data implementing atomic tools associated with
    loading data from CSV and JSON files into Spark DataFrames,
    pre-processing sub-steps, or feature engineering.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import glob
from logger import get_logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import re
import unicodedata


class DataUtils:
    '''
    -------------------------
    DataLoader - A utility class for atomic data-related tasks; e.g., loading data
                 from CSV and JSON files into Spark DataFrames, pre-processing
                 procedures, feature engineering, etc.
    -------------------------
    '''

    # Class logger for Spark operations.
    logger = get_logger(__name__)

    @staticmethod
    def load_json(spark: SparkSession, json_path: str) -> 'DataFrame':
        '''
        Load a JSON file into a Spark DataFrame.

            Parameters:
            -----------
            spark : SparkSession
                The active Spark session for loading data.
            json_path : str
                The path to the JSON file to load.

            Returns:
            -----------
            spark.DataFrame : The loaded Spark DataFrame.
        '''
        return spark.read.json(json_path)

    @staticmethod
    def load_csv(spark: SparkSession, csv_path: str) -> 'DataFrame':
        '''
        Load a CSV file into a Spark DataFrame.

            Parameters:
            -----------
            spark : SparkSession
                The active Spark session for loading data.
            csv_path : str
                The path to the CSV file to load.

            Returns:
            -----------
            spark.DataFrame : The loaded Spark DataFrame.
        '''
        return spark.read.option('header', True).csv(csv_path)

    @staticmethod
    def load_train_csv(spark: SparkSession, train_path_pattern: str) -> 'DataFrame':
        '''
        Find all CSV files matching the pattern for training data and load them
        into a single Spark DataFrame.

            Parameters:
            -----------
            spark : SparkSession
                The active Spark session for loading data.
            train_path_pattern : str
                The path pattern to search for training CSV files (e.g., 'data/train-*.csv')

            Returns:
            -----------
            DataFrame : A single Spark DataFrame containing all training data.
        '''
        # Find all train files matching the pattern.
        train_files = glob.glob(train_path_pattern)
        if not train_files:
            DataUtils.logger.info(f'ERROR: No TRAIN files found matching pattern "{train_path_pattern}".')
            raise ValueError(f'ERROR: No TRAIN files found matching pattern "{train_path_pattern}".')
        train_files.sort()
        DataUtils.logger.info(f'Found {len(train_files)} training files:\n{train_files}')
        # Load and union all train data files.
        train_df = None
        for file in train_files:
            current_df = DataUtils.load_csv(spark, file)
            if train_df is None:
                train_df = current_df
            else:
                train_df = train_df.union(current_df)
        # Debug train data details.
        DataUtils.logger.info(f'Training data count: {train_df.count()}.')
        DataUtils.logger.info('Training Data Schema:')
        train_df.printSchema()
        return train_df

    @staticmethod
    def preprocess_text(text: str) -> str:
        '''
        Cleans a given text string by:
        - Converting accented characters to standard English letters.
        - Removing unnecessary punctuation.
        - Stripping leading/trailing spaces.
        - Formatting to title-case.

            Parameters:
            -----------
            text : str
                The input text string.

            Returns:
            -----------
            str : The cleaned text string.
        '''
        if text is None or text.strip() == '':
            return None
        # Normalize accented characters to ASCII.
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        # Remove any remaining unwanted special characters but keep letters, numbers, and spaces.
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Trim any spaces and convert to title-case.
        return text.strip().title()

    @staticmethod
    def preprocess_numeric_cols(df: 'DataFrame', cols: list, num_type: str = 'integer') -> 'DataFrame':
        '''
        Handle pre-processing operations for numeric columns.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.
            cols : list
                The list of numeric column names to preprocess.
            num_type : str
                The target numeric type for conversion.

            Returns:
            -----------
            df : DataFrame
                The preprocessed DataFrame.
        '''
        for col_name in cols:
            # Convert missing values to None before type conversion. Required for Spark type casting operations.
            df = df.withColumn(col_name, when(col('runtimeMinutes') == '\\N', None).otherwise(col(col_name)))
            # Convert the current column to the proper numeric type: INT.
            df = df.withColumn(col_name, col(col_name).cast(num_type))
        return df

    @staticmethod
    def calc_median_col(df: 'DataFrame', col_name: str) -> int:
        '''
        Calculate median values per specified numeric column.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.
            col_name : str
                The name of the column to be processed.

            Returns:
            -----------
            col_median_int : int
                The calculated median value for the specified column.
        '''
        # Compute median for the target column.
        col_median_int = int(df.approxQuantile(col_name, [0.5], 0.0)[0])
        DataUtils.logger.info(f'Median: {col_name} = {col_median_int}')
        return col_median_int

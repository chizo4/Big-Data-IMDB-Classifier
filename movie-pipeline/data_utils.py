'''
--------------------------------------------------------------
FILE:
    movie-pipeline/data_utils.py

INFO:
    Utily class for data implementing core tools associated with
    loading data from CSV and JSON files into Spark DataFrames,
    saving results to disk, etc.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import glob
from logger import get_logger
import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p
import re
import unicodedata


class DataUtils:
    '''
    -------------------------
    DataLoader - A utility class for core data-related tasks; e.g., loading data
                 from CSV and JSON files into Spark DataFrames, or saving results.
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
    def load_data(spark: SparkSession, **kwargs) -> dict:
        '''
        Load data from CSV and JSON files into Spark DataFrames.

            Parameters:
            -----------
            spark : SparkSession
                The active Spark session for loading data.
            kwargs : dict
                Dictionary containing file paths for CSV and JSON datasets;
                these include train paths, test paths, JSON metadata, etc.

            Returns:
            -----------
            data_dict : dict
                Dictionary containing Spark DataFrames for all loaded datasets.
        '''
        # Extract data paths from kwargs.
        train_path = kwargs.get('train_path')
        val_path = kwargs.get('val_path')
        test_path = kwargs.get('test_path')
        directing_path = kwargs.get('directing_path')
        writing_path = kwargs.get('writing_path')
        if not all([train_path, val_path, test_path, directing_path, writing_path]):
            raise ValueError('ERROR: Missing required file paths in kwargs.')
        # Load JSON files with metadata.
        directing_df = DataUtils.load_json(spark, directing_path)
        writing_df = DataUtils.load_json(spark, writing_path)
        # Load CSV files.
        val_df = DataUtils.load_csv(spark, val_path)
        test_df = DataUtils.load_csv(spark, test_path)
        # For train data: detect all CSV files and load them accordingly.
        train_df = DataUtils.load_train_csv(spark, train_path)
        data_dict = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'directing': directing_df,
            'writing': writing_df
        }
        return data_dict

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
    def inject_median_values(df: 'DataFrame', col_name: str) -> 'DataFrame':
        '''
        Inject median values for missing entries in a numeric column.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.
            col_name : str
                The name of the column to be processed.

            Returns:
            -----------
            df : DataFrame
                The DataFrame with imputed median values.
        '''
        # Compute median for the target column.
        col_median_int = int(df.approxQuantile(col_name, [0.5], 0.0)[0])
        DataUtils.logger.info(f'Median: {col_name} = {col_median_int}')
        # Fill missing values with the computed median.
        df = df.withColumn(col_name, when(col(col_name).isNull(), col_median_int).otherwise(col(col_name)))
        return df

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
    def preprocess(df: 'DataFrame') -> 'DataFrame':
        '''
        Preprocess the data by handling missing values, fixing data
        types, normalizing text fields, handling outliers, etc.

        FULL PROCEDURE:
        (1) Pre-process numeric columns by converting them to integer type.
        (2) Inject median values for missing records: runtimeMinutes and numVotes.
        (3) Log-transform numVotes to reduce high skewness.
        (4) Ensure startYear ≤ endYear and handle missing values.
        (5) Normalize text fields ("primaryTitle", "originalTitle").

            Parameters:
            -----------
            df : DataFrame
                The input dataframe to preprocess.

            Returns:
            -----------
            df : DataFrame
                The preprocessed DataFrame.
        '''
        # (1) Pre-process numeric columns.
        numeric_cols = ['runtimeMinutes', 'numVotes', 'startYear', 'endYear']
        df = DataUtils.preprocess_numeric_cols(df, numeric_cols)
        # (2) Inject median values for missing records: "runtimeMinutes" and "numVotes".
        df = DataUtils.inject_median_values(df, 'runtimeMinutes')
        df = DataUtils.inject_median_values(df, 'numVotes')
        # (3) Log-transform "numVotes" to reduce skewness.
        df = df.withColumn('numVotes', log1p(col('numVotes')))
        # (4) Ensure startYear ≤ endYear and handle missing values.
        df = df.withColumn('startYear', when(col('startYear').isNull(), col('endYear')).otherwise(col('startYear')))
        df = df.withColumn('endYear', when(col('endYear').isNull(), col('startYear')).otherwise(col('endYear')))
        df = df.withColumn('endYear', when(col('endYear') < col('startYear'), col('startYear')).otherwise(col('endYear')))
        # (5) Normalize text title fields and handle missing records.
        df = df.toPandas()
        df['primaryTitle'] = df['primaryTitle'].apply(DataUtils.preprocess_text)
        df['originalTitle'] = df['originalTitle'].apply(DataUtils.preprocess_text)
        df['primaryTitle'] = df.apply(
            lambda row: row['originalTitle'] if pd.isna(row['primaryTitle']) else row['primaryTitle'], axis=1
        )
        df['originalTitle'] = df.apply(
            lambda row: row['primaryTitle'] if pd.isna(row['originalTitle']) else row['originalTitle'], axis=1
        )
        # Back to Spark DataFrame.
        df = SparkSession.builder.getOrCreate().createDataFrame(df)
        return df

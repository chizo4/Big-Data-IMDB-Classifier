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
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p
import re
import unicodedata


class DataUtils:
    '''
    -------------------------
    DataLoader - A utility class for core data-related tasks; e.g., loading data
                 from CSV and JSON files into Spark DataFrames, atomic pre-processing
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









    @staticmethod
    def engineer_features(df: 'DataFrame', **kwargs) -> 'DataFrame':
        '''
        Performs feature engineering on the dataset. Scale numerical columns,
        encode categorical features, and merge metadata from directing/writing tables.

        PROCEDURE:
        (1) Merge directing and writing metadata.
        (2) Convert years to categorical features.
        (3) Scale numeric features.
        (4) Set up Spark pipeline.
        (5) Fit and transform the data.
        (6) Drop redundant columns.

            Parameters:
            -----------
            df : DataFrame
                The input Spark DataFrame (ideally - processed data).
            kwargs : dict
                Dictionary containing dataframes for JSON metadata.

            Returns:
            -----------
            df : DataFrame
                The transformed DataFrame with engineered features.
        '''
        # (1) Merge directing and writing metadata.
        writing_df = kwargs.get('writing_df')
        writing_df = writing_df.withColumnRenamed('movie', 'tconst').withColumnRenamed('writer', 'writer_id')
        df = df.join(writing_df, on='tconst', how='left')



        # directing_df = kwargs.get('directing_df')
        # directing_df.printSchema()
        # writing_df.printSchema()
        # df = df.join(directing_df, on='tconst', how='left')
        # "Unknown" for missing director/writer IDs.
        # df = df.fillna({'director_id': 'Unknown', 'writer_id': 'Unknown'})
        # (2) Convert "startYear" and "endYear" into categorical features to avoid misleading numerical relationships.
        start_year_indexer = StringIndexer(inputCol='startYear', outputCol='startYear_idx', handleInvalid='keep')
        end_year_indexer = StringIndexer(inputCol='endYear', outputCol='endYear_idx', handleInvalid='keep')
        start_year_encoder = OneHotEncoder(inputCol='startYear_idx', outputCol='startYear_encoded')
        end_year_encoder = OneHotEncoder(inputCol='endYear_idx', outputCol='endYear_encoded')
        # (3) Convert "runtimeMinutes" and "numVotes" into a single feature vector and apply standard scaling
        # to prevent larger values (numVotes) from dominating smaller ones (runtimeMinutes), ensuring
        # balanced feature contributions in ML models.
        assembler = VectorAssembler(inputCols=['runtimeMinutes', 'numVotes'], outputCol='raw_features')
        scaler = StandardScaler(inputCol='raw_features', outputCol='scaled_features')
        # (4) Set up Spark pipeline.
        pipeline = SparkPipeline(stages=[
            start_year_indexer,
            end_year_indexer,
            start_year_encoder,
            end_year_encoder,
            assembler,
            scaler
        ])
        # (5) Fit and transform the data.
        df = pipeline.fit(df).transform(df)
        # (6) Drop redundant columns, disregarded after feature engineering.
        df = df.drop("startYear", "endYear", "startYear_idx", "endYear_idx", "raw_features")
        return df

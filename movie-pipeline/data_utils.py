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
import os
from pyspark.sql import SparkSession


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
        ######
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
            'validation': val_df,
            'test': test_df,
            'directing': directing_df,
            'writing': writing_df
        }
        return data_dict

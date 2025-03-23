'''
--------------------------------------------------------------
FILE:
    movie_pipeline/data_utils.py

INFO:
    Utily class for data implementing atomic tools associated with
    loading data from CSV and JSON files into Spark DataFrames,
    pre-processing sub-steps, or feature engineering.
    Implemented to avoid making ClassifierPipeline too bulky.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import csv
import glob
import json
from logger import get_logger
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, row_number, desc
from pyspark.ml.feature import StringIndexer
from pyspark.sql.window import Window
import pandas as pd
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
        DataUtils.logger.info(f'Loading from: "{json_path}".')
        return spark.read.json(json_path)

    @staticmethod
    def merge_directing_json(spark: SparkSession, directing_json_path: str) -> 'DataFrame':
        '''
        Merge movie and director records for directing JSON, since
        the raw data is not properly setup for further merging.

            Parameters:
            -----------
            spark : SparkSession
                The active Spark session for loading data.
            directing_json_path : str
                The path to the JSON file containing movie and director records.

            Returns:
            -----------
            df: spark.DataFrame
                The merged Spark DataFrame.
        '''
        # Merge movie and director records for directing.
        with open(directing_json_path, 'r') as file:
            directing_data = json.load(file)
        movie_dict = directing_data.get('movie', {})
        director_dict = directing_data.get('director', {})
        director_pairs = []
        # Process in smaller chunks to avoid memory issues.
        common_keys = set(movie_dict.keys()) & set(director_dict.keys())
        # Create data pairs.
        for idx in common_keys:
            director_pairs.append({
                'movie': movie_dict[idx],
                'director': director_dict[idx]
            })
        # Convert to Spark DataFrame only after processing is complete
        df = spark.createDataFrame(director_pairs)
        return df

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
        DataUtils.logger.info(f'Loading from: "{csv_path}".')
        return spark.read.option('header', True).option('inferSchema', True).csv(csv_path)

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
        DataUtils.logger.info(f'Found {len(train_files)} TRAIN files:\n{train_files}')
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
        DataUtils.logger.info('TRAIN DATA SCHEMA:')
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
    def normalize_text_cols(df: 'DataFrame') -> 'DataFrame':
        '''
        Normalize text columns by cleaning the text strings.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.
            cols : list
                The list of text column names to preprocess.

            Returns:
            -----------
            df : DataFrame
                The preprocessed DataFrame.
        '''
        # Convert for efficient text processing.
        df = df.toPandas()
        df['primaryTitle'] = df['primaryTitle'].apply(DataUtils.preprocess_text)
        df['originalTitle'] = df['originalTitle'].apply(DataUtils.preprocess_text)
        # Fill missing values with the other column content.
        df['primaryTitle'] = df.apply(
            lambda row: row['originalTitle'] if pd.isna(row['primaryTitle']) else row['primaryTitle'], axis=1
        )
        df['originalTitle'] = df.apply(
            lambda row: row['primaryTitle'] if pd.isna(row['originalTitle']) else row['originalTitle'], axis=1
        )
        # (Back to Spark DataFrame.)
        df = SparkSession.builder.getOrCreate().createDataFrame(df)
        return df

    @staticmethod
    def preprocess_numeric_cols(df: 'DataFrame', cols: list, num_type: str = 'double') -> 'DataFrame':
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
            df = df.withColumn(col_name, when(col(col_name) == '\\N', None).otherwise(col(col_name)))
            # Convert the current column to the proper numeric type: double.
            df = df.withColumn(col_name, col(col_name).cast(num_type))
        return df

    @staticmethod
    def calc_mean_col(df: 'DataFrame', col_name: str) -> float:
        '''
        Calculate mean values per specified numeric column.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.
            col_name : str
                The name of the column to be processed.

            Returns:
            -----------
            col_mean_float : float
                The calculated mean value for the specified column.
        '''
        mean_value = df.select(col_name).agg({col_name: 'avg'}).collect()[0][0]
        col_mean_float = float(mean_value) if mean_value is not None else 1
        DataUtils.logger.info(f'Mean: {col_name} = {col_mean_float}')
        return col_mean_float

    @staticmethod
    def string_index_col(df: 'DataFrame', col_name: str, return_model: bool=False) -> tuple:
        '''
        Apply string indexing to a given column.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.
            col_name : str
                The name of the column to be processed.
            return_model : bool
                Whether to return the fitted StringIndexerModel.

            Returns:
            -----------
            tuple :
                - If return_model=False: (df, output_col_name)
                Tuple of the preprocessed DataFrame and the output column name.
                - If return_model=True: (df, output_col_name, indexer_model)
                Tuple with DataFrame, output column name, and fitted StringIndexerModel.
        '''
        output_col_name = f'{col_name}_index'
        indexer = StringIndexer(inputCol=col_name, outputCol=output_col_name).setHandleInvalid('keep')
        indexer_model = indexer.fit(df)
        df = indexer_model.transform(df)
        # Drop intermediate column.
        df = df.drop(col_name)
        # Return with or without the model based on param.
        if return_model:
            return (df, output_col_name, indexer_model)
        else:
            return (df, output_col_name)

    @staticmethod
    def count_entity(df: 'DataFrame', key_name: str) -> 'DataFrame':
        '''
        Count the number of occurences per metadata field,
        e.g., directors, writers, etc.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.

            Returns:
            -----------
            counted_df : DataFrame
                The DataFrame with the counted metadata field.
        '''
        # Count occurrences of an entity.
        entity_counts = df.groupBy(key_name).count()
        # Join the counts back to the original data
        counted_df = df.join(
            entity_counts,
            on=key_name,
            how='left'
        ).withColumnRenamed('count', f'{key_name}_count')
        return counted_df

    @staticmethod
    def get_top_count_entity(df: 'DataFrame', key_name: str) -> 'DataFrame':
        '''
        Get the top count of an entity, e.g., director, writer, etc.

            Parameters:
            -----------
            df : DataFrame
                The input DataFrame to preprocess.

            Returns:
            -----------
            entity_df : DataFrame
                The DataFrame with the top count per entity.
        '''
        entity_window = Window.partitionBy('movie').orderBy(desc(key_name))
        entity_ranked = df.withColumn('rank', row_number().over(entity_window))
        entity_df = entity_ranked.filter(col('rank') == 1).drop('rank', key_name)
        return entity_df

    @staticmethod
    def load_or_create_genre_predictions(
        spark: 'SparkSession',
        df: 'DataFrame',
        predictor: 'LLMGenrePredictor',
        csv_cache_path: str
    ) -> 'DataFrame':
        '''
        Load genre predictions from a simple CSV cache if available, or generate new predictions.
        Uses a lightweight approach that minimizes memory usage and processing overhead.

            Parameters:
            -----------
            spark : SparkSession
                The active Spark session for loading and processing data.
            df : DataFrame
                The input DataFrame containing movie information.
            predictor : LLMGenrePredictor
                The predictor object that will generate genre predictions if needed.
            csv_cache_path : str
                The CSV path for caching predictions.

            Returns:
            -----------
            genre_df : DataFrame
                A DataFrame containing movie IDs (tconst) and their predicted genres.
        '''
        predictions = {}
        # Extract the tconst values we need to predict.
        current_ids = set([row['tconst'] for row in df.select('tconst').collect()])
        DataUtils.logger.info(f'Need predictions for {len(current_ids)} movies')
        # Load existing predictions from cache.
        if os.path.exists(csv_cache_path):
            DataUtils.logger.info(f'Loading genre cache from: "{csv_cache_path}".')
            try:
                # Read the CSV directly with Python. More efficient in this case.
                with open(csv_cache_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        predictions[row['tconst']] = row['genre']
                DataUtils.logger.info(f'Loaded {len(predictions)} cached predictions.')
            except Exception as e:
                DataUtils.logger.error(f'Error reading cache: {str(e)}')
                DataUtils.logger.info('Will generate new predictions')
                predictions = {}
        # Find which movies need predictions.
        missing_ids = current_ids - set(predictions.keys())
        # Generate predictions for missing movies.
        if missing_ids:
            DataUtils.logger.info(f'Generating predictions for {len(missing_ids)} new movies.')
            missing_df = df.filter(df.tconst.isin(list(missing_ids)))
            new_predictions_df = predictor.predict_genres(missing_df)
            # Extract new predictions.
            for row in new_predictions_df.collect():
                try:
                    predictions[row['tconst']] = row['genre']
                except (KeyError, IndexError):
                    predictions[row['tconst']] = 'unknown'
            # Save updated cache.
            try:
                DataUtils.logger.info(f'Saving updated cache to {csv_cache_path}')
                with open(csv_cache_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['tconst', 'genre'])
                    writer.writeheader()
                    for tconst, genre in predictions.items():
                        writer.writerow({'tconst': tconst, 'genre': genre})
            except Exception as e:
                DataUtils.logger.error(f'Error saving cache: {str(e)}')
        result_rows = [{'tconst': tconst, 'genre': genre} for tconst, genre in predictions.items()
                    if tconst in current_ids]
        genre_df = spark.createDataFrame(result_rows)
        return genre_df

    @staticmethod
    def save_preds_txt(df: 'DataFrame', output_txt_path: str) -> None:
        '''
        Save predictions to a TXT file.

            Parameters:
            -----------
            df : DataFrame
                The DataFrame containing the predictions.
            output_txt_path : str
                The path to save the predictions.
        '''
        # Extract the predictions and convert numeric predictions to boolean strings.
        DataUtils.logger.info('Converting predictions to boolean strings...')
        # IMPORTANT: Sort by tconst to match the original order
        sorted_df = df.orderBy('tconst')
        pred_results = sorted_df.select('tconst', 'prediction').collect()
        # OPTIONAL: label distribution debugs.
        # DataUtils.logger.info("Prediction distribution:")
        # true_count = 0
        # false_count = 0
        # for row in pred_results:
        #     if row['prediction'] == 1.0:
        #         true_count += 1
        #     else:
        #         false_count += 1
        # total = true_count + false_count
        # true_percent = (true_count / total) * 100 if total > 0 else 0
        # false_percent = (false_count / total) * 100 if total > 0 else 0
        # DataUtils.logger.info(f"True predictions: {true_count} ({true_percent:.2f}%)")
        # DataUtils.logger.info(f"False predictions: {false_count} ({false_percent:.2f}%)")
        # Format predictions.
        pred_strings = []
        for row in pred_results:
            pred_value = 'True' if row['prediction'] == 1.0 else 'False'
            pred_strings.append(pred_value)
        # Write predictions to TXT file.
        DataUtils.logger.info(f'Writing {len(pred_strings)} predictions to: "{output_txt_path}".')
        with open(output_txt_path, 'w') as f:
            for pred in pred_strings:
                f.write(f"{pred}\n")

'''
--------------------------------------------------------------
FILE:
    movie_pipeline/classifier_pipeline.py

INFO:
    Main pipeline file orchestrating the full classifier pipeline
    workflow. From data loading to model predictions.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import argparse
from data_utils import DataUtils
from datetime import datetime
from llm_predictor import LLMGenrePredictor
from logger import get_logger
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, collect_list, concat_ws


class ClassifierPipeline:
    '''
    -------------------------
    ClassifierPipeline - Main class orchestrating the overall workflow, utilizing
                         DataUtils and Classifier classes:
                         (1) Initial setups: CLI args, access data paths, etc.
                         (2) Loading data (from CSV/JSON).
                         (3) Pre-processing data (missing values, data types, etc.).
                         (4) Feature engineering (metadata, genre predictions, etc.).
                         (5) Train the Random Forest Classifier model.
                         (6) Evaluates and save predictions.
    -------------------------
    '''

    # Class logger for Spark operations.
    logger = get_logger(__name__)
    # Base file name for results to be customized for use case.
    RESULT_FILE_BASE = r'{set_name}_prediction_{timestamp}.csv'
    # Standard cols to follow in data.
    NUMERIC_COLS = {'runtimeMinutes', 'numVotes', 'startYear', 'endYear'}

    def __init__(self: 'ClassifierPipeline') -> None:
        '''
        Initialize the ClassifierPipeline class.
        '''
        self.data_dict = {}
        self.median_dict = {
            'runtimeMinutes': None,
            'numVotes': None
        }
        # Initialize feature cols that will be further extended. Skip "endYear".
        self.feature_cols = self.NUMERIC_COLS - {'endYear'}
        # Initialize Spark session.
        self.spark = SparkSession.builder.appName('MoviePipeline').getOrCreate()
        # Initialize RF classifier and scaler.
        self.rf_classifier = RandomForestClassifier(
            featuresCol='scaled_features',
            labelCol='label',
            numTrees=100,
            seed=42
        )
        self.scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features',
            withStd=True,
            withMean=False
        )
        # LLLM genre predictor setup.
        self.genre_predictor_llm = LLMGenrePredictor(
            batch_size=20,
            model_name='gemma3:1b',
            spark=self.spark
        )

    @staticmethod
    def set_args() -> argparse.Namespace:
        '''
        Process the CLI arguments relating to data.

            Returns:
            -------------------------
            args : argparse.Namespace
                Parsed arguments for the script.
        '''
        parser = argparse.ArgumentParser(description='Data for movie pipeline.')
        parser.add_argument(
            '--data',
            type=str,
            required=True,
            help='Base path to access the task data.'
        )
        parser.add_argument(
            '--val',
            type=str,
            required=True,
            help='Name of the CSV validation file.'
        )
        parser.add_argument(
            '--test',
            type=str,
            required=True,
            help='Name of the CSV test file.'
        )
        parser.add_argument(
            '--directing',
            type=str,
            required=True,
            help='Name of the JSON directing file.'
        )
        parser.add_argument(
            '--writing',
            type=str,
            required=True,
            help='Name of the JSON writing file.'
        )
        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Path to store the trained model.'
        )
        parser.add_argument(
            '--results',
            type=str,
            required=True,
            help='Path to store the model ouputs (results).'
        )
        return parser.parse_args()

    @staticmethod
    def set_pred_file(base_path: str, set_name: str) -> str:
        '''
        Initialize the CSV prediction file for the classification task.
        Annotate it with the current timestamp and set name.

            Parameters:
            -------------------------
            base_path : str
                Base path to store results.
            set_name : str
                Name of the set to initialize the prediction file for.

            Returns:
            -------------------------
            pred_path : str
                Customized path name to the prediction file.
        '''
        # r'{set_name}_prediction_{timestamp}.csv'
        curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_filename = ClassifierPipeline.RESULT_FILE_BASE
        pred_filename = pred_filename.replace('{set_name}', set_name)
        pred_filename = pred_filename.replace('{timestamp}', curr_time)
        pred_path = f'{base_path}/{pred_filename}'
        return pred_path

    def set_files(self: 'ClassifierPipeline') -> None:
        '''
        Set the file paths for the data, model, and results.
        '''
        # Extract and assign data paths for the task.
        self.data_path = self.args.data
        self.cache_path = f'{self.data_path}/genre_predictions.parquet'
        self.train_csv_path = f'{self.data_path}/train-*.csv'
        self.val_csv_path = f'{self.data_path}/{self.args.val}'
        self.test_csv_path = f'{self.data_path}/{self.args.test}'
        self.directing_json_path = f'{self.data_path}/{self.args.directing}'
        self.writing_json_path = f'{self.data_path}/{self.args.writing}'
        # Set path for model storage.
        self.model_path = self.args.model
        # Initialize VAL and TEST results files.
        self.val_pred_path = self.set_pred_file(set_name='val', base_path=self.args.results)
        self.test_pred_path = self.set_pred_file(set_name='test', base_path=self.args.results)

    def load_data(self: 'ClassifierPipeline') -> None:
        '''
        Load data from CSV and JSON files into Spark DataFrames
        to build a full-data dictionary.
        '''
        # Load JSON files with metadata for directing and writing.
        writing_df = DataUtils.load_json(self.spark, self.writing_json_path)
        # For directing: merge movie and director dictionaries (since unstructured).
        directing_df = DataUtils.merge_directing_json(self.spark, self.directing_json_path)
        # Load CSV files.
        val_df = DataUtils.load_csv(self.spark, self.val_csv_path)
        test_df = DataUtils.load_csv(self.spark, self.test_csv_path)
        # For train data: detect all CSV files and load them accordingly.
        train_df = DataUtils.load_train_csv(self.spark, self.train_csv_path)
        self.data_dict = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'directing': directing_df,
            'writing': writing_df
        }

    def preprocess(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Preprocess the data by handling missing values, fixing data types,
        normalizing text fields, handling outliers, etc. Utilizes various
        atomic functions from DataUtils.

            PROCEDURE:
            (1) Pre-process numeric columns by converting them to integer type.
            (2) Inject median values for missing records: runtimeMinutes and numVotes.
            (3) Ensure startYear ≤ endYear and handle missing values.
            (4) Normalize text fields ("primaryTitle", "originalTitle").

            Parameters:
            -----------
            df : DataFrame
                The input dataframe to preprocess.
            train : bool (default=False)
                Flag to indicate if the input data is the TRAIN set.

            Returns:
            -----------
            (DataFrame, medians) : tuple
                Tuple with the preprocessed DataFrame and optionally medians
        '''
        # (1) Pre-process numeric columns.
        ClassifierPipeline.logger.info('Pre-processing numeric columns...')
        df = DataUtils.preprocess_numeric_cols(df, self.NUMERIC_COLS)
        # (2) Inject median values for missing records: "runtimeMinutes" and "numVotes".
        for col_name in ['runtimeMinutes', 'numVotes']:
            # From TRAIN: find medians for numeric columns (for further injection).
            # For other sets, assign pre-computed values, since TRAIN runs first.
            if train:
                self.median_dict[col_name] = DataUtils.calc_median_col(df, col_name)
            # Inject TRAIN median values into NULL fields.
            df = df.withColumn(
                col_name, when(col(col_name).isNull(), self.median_dict[col_name]).otherwise(col(col_name))
            )
        # (3) Ensure startYear ≤ endYear and handle missing values.
        df = df.withColumn('startYear', when(col('startYear').isNull(), col('endYear')).otherwise(col('startYear')))
        df = df.withColumn('endYear', when(col('endYear').isNull(), col('startYear')).otherwise(col('endYear')))
        df = df.withColumn('endYear', when(col('endYear') < col('startYear'), col('startYear')).otherwise(col('endYear')))
        # (4) Normalize text title fields and handle missing records.
        ClassifierPipeline.logger.info('Pre-processing textual columns...')
        df = DataUtils.normalize_text_cols(df)
        return df

    def merge_metadata_into_df(self: 'ClassifierPipeline', df: 'DataFrame') -> 'DataFrame':
        '''
        Merge movie metadata - writing and directing - into main DataFrame.
        For movies with multiple writers/directors, select the one with highest occurrence.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to merge the metadata into.

            Returns:
            -----------
            df : DataFrame
                DataFrame including directing and writing metadata.
        '''
        ClassifierPipeline.logger.info('Assigning most frequent writer/director per movie...')
        # Count occurrences of each writer and director across all movies.
        writing_df = DataUtils.count_entity(df=self.data_dict['writing'], key_name='writer')
        directing_df = DataUtils.count_entity(df=self.data_dict['directing'], key_name='director')
        # For each movie, select the writer/director with the highest count.
        writing_df = DataUtils.get_top_count_entity(df=writing_df, key_name='writer_count')
        directing_df = DataUtils.get_top_count_entity(df=directing_df, key_name='director_count')
        # Join the selected writer/director for each movie to the main dataset.
        df = df.join(writing_df, df['tconst'] == writing_df['movie'], how='left').drop('movie')
        df = df.join(directing_df, df['tconst'] == directing_df['movie'], how='left').drop('movie')
        # Handle NULL cols for writers and directors
        df = df.withColumn('writer', when(col('writer').isNull(), 'unknown').otherwise(col('writer')))
        df = df.withColumn('director', when(col('director').isNull(), 'unknown').otherwise(col('director')))
        return df

    def predict_genres_llm(self, df: 'DataFrame') -> 'DataFrame':
        '''
        Predict movie genres using LLM and add them to the DataFrame.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to predict genres for.

            Returns:
            -----------
            df : DataFrame
                The main DataFrame with added genre predictions.
        '''
        # Get genre predictions (from cache or by generating new ones).
        genre_df = DataUtils.load_or_create_genre_predictions(
            self.spark, df, self.genre_predictor_llm, self.cache_path
        )
        # Join the predictions with the main dataframe.
        df = df.join(genre_df, on='tconst', how='left')
        # Handle missing genres.
        df = df.withColumn('genre', when(col('genre').isNull(), 'unknown').otherwise(col('genre')))
        return df

    def engineer_features(self: 'ClassifierPipeline', df: 'DataFrame', train:bool=False) -> 'DataFrame':
        '''
        Apply feature engineering to the DataFrame, handling metadata.

            PROCEDURE:
            (1) Merging metadata (directors, writers) to include relevant categorical features.
            (2) For TRAIN: convert boolean string labels to binary numeric.
            (3) Utilize LLM to create new synthetic feature: "genre".
            (4) Handling categorical features via string indexing.
            (5) Handling missing values.
            (6) Assemble features into single vector.
            (7) Standardize numeric features.

            Parameters:
            -----------
            df : DataFrame
                The input dataframe to be transformed.
            train : bool (default=False)
                Flag to indicate if the input data is the TRAIN set.

            Returns:
            -----------
            df : DataFrame
                The transformed DataFrame ready for training.
        '''
        # (1) Join movie dataset with writing and directing metadata.
        ClassifierPipeline.logger.info('Merging METADATA JSON into main DataFrame...')
        df = self.merge_metadata_into_df(df)
        ClassifierPipeline.logger.info('Initial DataFrame after metadata merge:')
        df.show(10, False)
        # (2) Drop endYear column due to high percentage of missing values (around 90%).
        ClassifierPipeline.logger.info('Dropping "endYear" column...')
        if 'endYear' in df.columns:
            df = df.drop('endYear')
        # (2) For TRAIN: convert string labels to binary numeric, where: "True" -> 1.0, "False" -> 0.0.
        if train:
            ClassifierPipeline.logger.info('Converting boolean "label" field to binary for TRAIN...')
            df = df.withColumn('label', col('label').cast('double'))
        # (3) Apply LLM to introduce new synthetic feature: "genre".
        ClassifierPipeline.logger.info('Generating synthetic genre features via LLM...')
        df = self.predict_genres_llm(df)
        # (4) Handle categorical variables (metadata, genre) by applying StringIndexer.
        ClassifierPipeline.logger.info('String indexing categorical features...')
        for col_name in ['writer', 'director', 'genre']:
            df, output_col_name = DataUtils.string_index_col(df, col_name)
            self.feature_cols.add(output_col_name)
        ClassifierPipeline.logger.info(f'Feature column names: {self.feature_cols}')
        # (5) Handle missing values.
        df = df.fillna(0)
        # (6) Assemble all feature columns into a single vector.
        ClassifierPipeline.logger.info('Assembling features into a single vector...')
        assembler = VectorAssembler(inputCols=list(self.feature_cols), outputCol='features')
        df = assembler.transform(df)
        # (7) Standardize numeric features.
        ClassifierPipeline.logger.info('Standardizing numerical features')
        scaler_model = self.scaler.fit(df)
        df = scaler_model.transform(df)
        ClassifierPipeline.logger.info('Final DataFrame after all FE steps:')
        df.show(10, False)
        return df

    # def train_model(self: 'ClassifierPipeline', df: 'DataFrame') -> 'PipelineModel':
    def train_model(self: 'ClassifierPipeline', train_df: 'DataFrame') -> 'PipelineModel':
        '''
        Train the RandomForestClassifier model using the engineered features,
        and save it for further use.

            Parameters:
            -----------
            df : DataFrame
                The preprocessed and feature-engineered DataFrame for training.

            Returns:
            -----------
            model : PipelineModel
                The trained model.
        '''
        train_df, val_df = train_df.randomSplit([0.8, 0.2], seed=42) #TEMP!
        # Set Spark ML Pipeline to encapsulate training steps.
        spark_pipeline = SparkPipeline(stages=[self.rf_classifier])
        model = spark_pipeline.fit(train_df) #df
        # Save trained model.
        model.write().overwrite().save(self.model_path)
        ClassifierPipeline.logger.info(f'TRAINED RF model saved to: "{self.model_path}".')
        # TEMP: some eval debugs
        y_pred = model.transform(val_df)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(y_pred)
        print("Model Accuracy:", accuracy)
        return model

    def __call__(self: 'ClassifierPipeline') -> None:
        '''
        Main method to call the pipeline functionalities.

            PROCEDURE:
            (1) Load data.
            (2) Pre-process data.
            (3) Apply feature engineering.
            (4) Train RF classifier model.
            (5) Predict via trained model and save outputs.
        '''
        # (0) Process CLI args.
        self.args = ClassifierPipeline.set_args()
        self.set_files()
        # (1) Load data.
        ClassifierPipeline.logger.info('***(1) LOADING DATA...***')
        self.load_data()
        ClassifierPipeline.logger.info('***(1) DATA: LOADED!***')
        # (2) Pre-process data: TRAIN, VAL, TEST.
        ClassifierPipeline.logger.info('***(2) PRE-PROCESSING DATA...***')
        train_df = self.preprocess(self.data_dict['train'], train=True)
        val_df = self.preprocess(self.data_dict['val'])
        test_df = self.preprocess(self.data_dict['test'])
        ClassifierPipeline.logger.info('***(2) DATA PRE-PROCESSING: COMPLETE!***')
        # (3) Apply feature engineering procedures.
        ClassifierPipeline.logger.info('***(3) APPLYING FEATURE ENGINEERING...***')
        ClassifierPipeline.logger.info('***FE: TRAIN SET***')
        # train_df = self.engineer_features(train_df, train=True)
        # ClassifierPipeline.logger.info('***FE: VAL SET***')
        # val_df = self.engineer_features(val_df)
        # ClassifierPipeline.logger.info('***FE: TEST SET***')
        # test_df = self.engineer_features(test_df)
        ClassifierPipeline.logger.info('***(3) FEATURE ENGINEERING: COMPLETE!***')
        # (4) Train the RF model.
        ClassifierPipeline.logger.info('***(4) TRAINING RANDOM FOREST CLASSIFIER...***')
        # rf_model = self.train_model(train_df)
        ClassifierPipeline.logger.info('***(4) MODEL TRAINING: COMPLETE!***')


if __name__ == '__main__':
    classifier_pipe = ClassifierPipeline()
    classifier_pipe()

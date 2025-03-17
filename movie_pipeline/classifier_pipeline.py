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
                         (3) Train the Random Forest Classifier model.
                         (3) Evaluates and save predictions.
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
        # Set up CLI args.
        self.args = ClassifierPipeline.set_args()
        # Extract and assign data paths for the task.
        self.data_path = self.args.data
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
        # Initialize feature cols that will be further extended. Skip endYear.
        self.feature_cols = self.NUMERIC_COLS - {'endYear'}

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
        Merge movie metadata - wrting and directing - into main DataFrame.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to merge the metadata into.

            Returns:
            -----------
            df : DataFrame
                DataFrame including directing and writing metadata.
        '''
        # Join movie dataset with writing and directing metadata. Some movies have
        # multiple writers and/or directors, thus we need to first aggregate them.
        writing_df = self.data_dict['writing'].groupBy('movie') \
            .agg(collect_list('writer').alias('writers_array')) \
            .withColumn('writers', concat_ws(',', 'writers_array')) \
            .drop('writers_array')
        directing_df = self.data_dict['directing'].groupBy('movie') \
            .agg(collect_list('director').alias('directors_array')) \
            .withColumn('directors', concat_ws(',', 'directors_array')) \
            .drop('directors_array')
        df = df.join(writing_df, df['tconst'] == writing_df['movie'], how='left').drop('movie')
        df = df.join(directing_df, df['tconst'] == directing_df['movie'], how='left').drop('movie')
        # Handle NULL cols for writers and directors.
        df = df.withColumn('writers', when(col('writers').isNull(), 'unknown').otherwise(col('writers')))
        df = df.withColumn('directors', when(col('directors').isNull(), 'unknown').otherwise(col('directors')))
        return df

    def engineer_features(self: 'ClassifierPipeline', df: 'DataFrame', train:bool=False) -> 'DataFrame':
        '''
        Apply feature engineering to the DataFrame, handling metadata.

            PROCEDURE:
            (1) Merging metadata (directors, writers) to include relevant categorical features.
            (2) For TRAIN: convert boolean string labels to binary numeric.
            (3) Handling categorical features via tokenizing+hashing and indexing.
            (4) Handling missing values.
            (5) Assemble features into single vector.
            (6) Standardize numeric features.

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
        # (2) For TRAIN: convert string labels to binary numeric, where: "True" -> 1, "False" -> 0.
        if train:
            ClassifierPipeline.logger.info('Converting boolean labels to binary for TRAIN...')
            df = df.withColumn('label', col('label').cast('double'))
        # (3) Handle categorical variables.
        # For "writers" and "directors": use tokenization and hashing for better handling.
        ClassifierPipeline.logger.info('Tokenizing and hashing metadata for directors and writers...')
        for col_name in ['writers', 'directors']:
            # Tokenize and hash the current column and add it to the feature set.
            df, output_col_name = DataUtils.tokenize_and_hash_col(df, col_name)
            self.feature_cols.add(output_col_name)
        # For text titles: apply standard StringIndexer.
        ClassifierPipeline.logger.info('String indexing titles...')
        for col_name in ['primaryTitle', 'originalTitle']:
            df, output_col_name = DataUtils.string_index_col(df, col_name)
            self.feature_cols.add(output_col_name)
        ClassifierPipeline.logger.info(f'Feature column names: {self.feature_cols}')
        # (4) Handle missing values.
        df = df.fillna(0)
        # (5) Assemble all feature columns into a single vector
        ClassifierPipeline.logger.info('Assembling features into a single vector...')
        assembler = VectorAssembler(inputCols=list(self.feature_cols), outputCol='features')
        df = assembler.transform(df)
        # (6) Standardize numeric features.
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
        train_df = self.engineer_features(train_df, train=True)
        # ClassifierPipeline.logger.info('***FE: VAL SET***')
        # val_df = self.engineer_features(val_df)
        # ClassifierPipeline.logger.info('***FE: TEST SET***')
        # test_df = self.engineer_features(test_df)
        ClassifierPipeline.logger.info('***(3) FEATURE ENGINEERING: COMPLETE!***')
        # (4) Train the RF model.
        ClassifierPipeline.logger.info('***(4) TRAINING RANDOM FOREST CLASSIFIER...***')
        rf_model = self.train_model(train_df)
        ClassifierPipeline.logger.info('***(4) MODEL TRAINING: COMPLETE!***')


if __name__ == '__main__':
    classifier_pipe = ClassifierPipeline()
    classifier_pipe()

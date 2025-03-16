'''
--------------------------------------------------------------
FILE:
    movie-pipeline/pipeline.py

INFO:
    Main pipeline file orchestrating the full-pipeline workflow.
    From data loading to model predictions.

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
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


class Pipeline:
    '''
    -------------------------
    Pipeline - Main class orchestrating the overall workflow, utilizing DataUtils
               and Classifier classes:
               (1) Initial setups: CLI args, access data paths, etc.
               (2) Loading data (from CSV/JSON).
               (3) Process and train the model with Classifier.
               (3) Evaluates and save predictions.
    -------------------------
    '''

    # Class logger for Spark operations.
    logger = get_logger(__name__)
    # Base path for results to be customized for the task-specific data.
    RESULT_BASE_PATH = r'results/{data_path}/{set_name}_prediction_{timestamp}.csv'
    # Standard numeric cols in data.
    NUMERIC_COLS = ['runtimeMinutes', 'numVotes', 'startYear', 'endYear']

    def __init__(self: 'Pipeline') -> None:
        '''
        Initialize the Pipeline class.
        '''
        self.df = None
        self.median_dict = {
            'runtimeMinutes': None,
            'numVotes': None
        }
        # Set up CLI args.
        self.args = Pipeline.set_args()
        # Extract and assign data paths for the task.
        self.data_path = self.args.data
        self.train_csv_path = f'{self.data_path}/train-*.csv'
        self.val_csv_path = f'{self.data_path}/{self.args.val}'
        self.test_csv_path = f'{self.data_path}/{self.args.test}'
        self.directing_json_path = f'{self.data_path}/{self.args.directing}'
        self.writing_json_path = f'{self.data_path}/{self.args.writing}'
        # Initialize VAL and TEST results files.
        self.val_pred_path = self.set_pred_file(set_name='val', data_path=self.data_path)
        self.test_pred_path = self.set_pred_file(set_name='test', data_path=  self.data_path)
        # Initialize Spark session.
        self.spark = SparkSession.builder.appName('MoviePipeline').getOrCreate()

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
        return parser.parse_args()

    @staticmethod
    def set_pred_file(data_path: str, set_name: str) -> str:
        '''
        Initialize the CSV prediction file for the classification task.
        Annotate it with the current timestamp and set name.

            Parameters:
            -------------------------
            data_path : str
                Base path to access the task data.
            set_name : str
                Name of the set to initialize the prediction file for.

            Returns:
            -------------------------
            pred_path : str
                Customized path name to the prediction file.
        '''
        curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_path = Pipeline.RESULT_BASE_PATH
        pred_path = pred_path.replace('{data_path}', data_path)
        pred_path = pred_path.replace('{set_name}', set_name)
        pred_path = pred_path.replace('{timestamp}', curr_time)
        return pred_path

    def preprocess(self: 'Pipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Preprocess the data by handling missing values, fixing data types,
        normalizing text fields, handling outliers, etc. Utilizes various
        atomic functions from DataUtils.

            FULL PROCEDURE:
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
        df = df.toPandas()
        df['primaryTitle'] = df['primaryTitle'].apply(DataUtils.preprocess_text)
        df['originalTitle'] = df['originalTitle'].apply(DataUtils.preprocess_text)
        df['primaryTitle'] = df.apply(
            lambda row: row['originalTitle'] if pd.isna(row['primaryTitle']) else row['primaryTitle'], axis=1
        )
        df['originalTitle'] = df.apply(
            lambda row: row['primaryTitle'] if pd.isna(row['originalTitle']) else row['originalTitle'], axis=1
        )
        # (Back to Spark DataFrame.)
        df = SparkSession.builder.getOrCreate().createDataFrame(df)
        return df

    def __call__(self: 'Pipeline') -> None:
        '''
        Main method to call the pipeline functionalities.

        PROCEDURE:
        (1) Load data.
        (2) Pre-process data.
        (3) Apply feature engineering.
        (4) TODO:
        '''
        # (1) Load data.
        Pipeline.logger.info('LOADING DATA...')
        data = DataUtils.load_data(
            spark=self.spark,
            train_path=self.train_csv_path,
            val_path=self.val_csv_path,
            test_path=self.test_csv_path,
            directing_path=self.directing_json_path,
            writing_path=self.writing_json_path
        )
        Pipeline.logger.info('DATA LOADED!')
        # (2) Pre-process data: TRAIN, VAL, TEST.
        Pipeline.logger.info('PRE-PROCESSING DATA...')
        train_df = self.preprocess(data['train'], train=True)
        val_df = self.preprocess(data['val'])
        test_df = self.preprocess(data['test'])
        Pipeline.logger.info('DATA PRE-PROCESSING COMPLETE!')
        # (3) TODO: Apply feature engineering procedures.

        # Pipeline.logger.info('APPLYING FEATURE ENGINEERING...')
        # train_df = DataUtils.engineer_features(
        #     df=train_df, directing_df=data['directing'], writing_df=data['writing']
        # )
        # train_df.show(20, truncate=False)
        # val_df = DataUtils.engineer_features(
        #     df=val_df, directing_df=data['directing'], writing_df=data['writing']
        # )
        # test_df = DataUtils.engineer_features(
        #     df=test_df, directing_df=data['directing'], writing_df=data['writing']
        # )
        Pipeline.logger.info('FEATURE ENGINEERING COMPLETE!')


if __name__ == '__main__':
    pipe = Pipeline()
    pipe()

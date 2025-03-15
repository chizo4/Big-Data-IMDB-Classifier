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
from pyspark.sql import SparkSession


class Pipeline:
    '''
    -------------------------
    Pipeline - Main class orchestrating the overall workflow, utilizing DataUtils
               and Classifier classes:
               (1) Initial setups: CLI args, access data paths, etc.
               (2) Loading CSV/JSON data.
               (3) Process and train the model with Classifier.
               (3) Evaluates and save predictions.
    -------------------------
    '''

    # Class logger for Spark operations.
    logger = get_logger(__name__)
    # Base path for results to be customized for the task-specific data.
    RESULT_BASE_PATH = r'results/{data_path}/{set_name}_prediction_{timestamp}.csv'

    def __init__(self: 'Pipeline') -> None:
        '''
        Initialize the Pipeline class.
        '''
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

    def __call__(self: 'Pipeline') -> None:
        '''
        Main method to call the pipeline functionalities:
        (1) Load data.
        '''
        Pipeline.logger.info('LOADING DATA...')
        # Load data.
        data = DataUtils.load_data(
            spark=self.spark,
            train_path=self.train_csv_path,
            val_path=self.val_csv_path,
            test_path=self.test_csv_path,
            directing_path=self.directing_json_path,
            writing_path=self.writing_json_path
        )
        Pipeline.logger.info('DATA LOADED!')
        # Pre-process data.
        Pipeline.logger.info('PREPROCESSING DATA...')
        # data['train'].show(20, truncate=False)
        train_df = DataUtils.preprocess(data['train'])
        # train_df.show(20, truncate=False)
        val_df = DataUtils.preprocess(data['val'])
        test_df = DataUtils.preprocess(data['test'])
        # Apply feature engineering procedures.
        Pipeline.logger.info('APPLYING FEATURE ENGINEERING...')
        # train_df = DataUtils.feature_engineering(train_df)
        # val_df = DataUtils.feature_engineering(val_df)
        # test_df = DataUtils.feature_engineering(test_df)
        Pipeline.logger.info('DATA PREPARATION COMPLETE!')


if __name__ == '__main__':
    pipe = Pipeline()
    pipe()

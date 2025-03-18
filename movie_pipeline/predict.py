'''
--------------------------------------------------------------
FILE:
    movie_pipeline/predict.py

INFO:
    Entry point for model prediction pipeline that handles
    command-line arguments before passing them to ClassifierPipeline.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import argparse
from datetime import datetime
import sys
from classifier_pipeline import ClassifierPipeline
from logger import get_logger


class PredictRunner:
    '''
    -------------------------
    PredictRunner - A class to execute the prediction pipeline.
    -------------------------
    '''

    # Base file name for results to be customized for use case.
    RESULT_FILE_BASE = r'{set_name}_prediction_{timestamp}.txt'

    def __init__(self: 'PredictRunner') -> None:
        '''
        Initialize the PredictRunner class.
        '''
        self.logger = get_logger(__name__)
        self.args = self._parse_args()

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        '''
        Parse command line arguments.

            Returns:
            --------
            args : argparse.Namespace
                Parsed CLI arguments.
        '''
        parser = argparse.ArgumentParser(description='Data for prediction pipeline.')
        # Core args (i.e., required).
        parser.add_argument(
            '--data-path',
            type=str,
            required=True,
            help='Base path to access the task data.'
        )
        parser.add_argument(
            '--directing-json',
            type=str,
            required=True,
            help='Path to the directing metadata JSON file.'
        )
        parser.add_argument(
            '--writing-json',
            type=str,
            required=True,
            help='Path to the writing metadata JSON file.'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            required=True,
            help='Path from where the trained model will be reloaded.'
        )
        parser.add_argument(
            '--results-path',
            type=str,
            required=True,
            help='Path where the results will be saved.'
        )
        parser.add_argument(
            '--test-csv',
            type=str,
            required=True,
            help='Path to the prediction CSV file.'
        )
        # Optional args.
        parser.add_argument(
            '--model',
            type=str,
            default='gemma3:1b',
            help='Ollama LLM model name for genre predictor.'
        )
        return parser.parse_args()

    @staticmethod
    def _set_pred_file(base_path: str, set_name: str) -> str:
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
        curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_filename = PredictRunner.RESULT_FILE_BASE
        pred_filename = pred_filename.replace('{set_name}', set_name)
        pred_filename = pred_filename.replace('{timestamp}', curr_time)
        pred_path = f'{base_path}/{pred_filename}'
        return pred_path

    def _set_file_paths(self: 'PredictRunner') -> None:
        '''
        Set the paths for the prediction-related files.
        '''
        # Extract and assign data paths for the task.
        data_path = self.args.data_path
        # Extract set name identified from CSV name.
        set_name = self.args.test_csv.split('/')[-1].split('_')[0]
        self.cache_path = f'{data_path}/{set_name}_cache.csv'
        self.pred_path = self._set_pred_file(
            set_name=set_name,
            base_path=self.args.results_path
        )

    def __call__(self: 'PredictRunner') -> None:
        '''
        Run the prediction pipeline.
        '''
        self.logger.info('Creating PREDICTION files.')
        self._set_file_paths()
        self.logger.info('Running PREDICTION for movie classification.')
        pipeline = ClassifierPipeline(
            model_path=self.args.model_path,
            results_path=self.args.results_path,
            directing_json_path=self.args.directing_json,
            writing_json_path=self.args.writing_json,
            llm_cache_path=self.cache_path,
            llm_name=self.args.model,
            train=False
        )
        pipeline.run_predict(
            input_csv_path=self.args.test_csv,
            output_txt_path=self.pred_path
        )
        self.logger.info('***SUCCESS: PREDICTION COMPLETE!***')


if __name__ == '__main__':
    predict_runner = PredictRunner()
    try:
        predict_runner()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f'Error running prediction pipeline: {str(e)}')
        logger.exception('Stack trace:')
        sys.exit(1)

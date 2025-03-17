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
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
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
    # Base path for results to be customized for the task-specific data.
    RESULT_BASE_PATH = r'results/{data_path}/{set_name}_prediction_{timestamp}.csv'
    # Base path for trained model.
    MODEL_BASE_PATH = r'models/{data_path}/'
    # Standard cols to follow in data.
    NUMERIC_COLS = {'runtimeMinutes', 'numVotes', 'startYear', 'endYear'}
    CATEGORICAL_COLS = {'primaryTitle', 'originalTitle', 'writers', 'directors'}

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
        # Set up model store path.
        self.model_path = ClassifierPipeline.MODEL_BASE_PATH.replace('{data_path}', self.data_path)
        # Initialize VAL and TEST results files.
        self.val_pred_path = self.set_pred_file(set_name='val', data_path=self.data_path)
        self.test_pred_path = self.set_pred_file(set_name='test', data_path=  self.data_path)
        # Initialize Spark session.
        self.spark = SparkSession.builder.appName('MoviePipeline').getOrCreate()
        # Initialize RF classifier.
        self.rf_classifier = RandomForestClassifier(
            featuresCol='scaled_features',
            labelCol='label',
            numTrees=100,  # TODO: Number of trees in the forest.
            # maxDepth=10,   # TODO: Depth of each tree (to prevent overfitting).
            seed=42        # TODO: Random seed for reproducibility.
        )
        # Initialize feature cols that will be further extended.
        self.feature_cols = self.NUMERIC_COLS

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
        pred_path = ClassifierPipeline.RESULT_BASE_PATH
        pred_path = pred_path.replace('{data_path}', data_path)
        pred_path = pred_path.replace('{set_name}', set_name)
        pred_path = pred_path.replace('{timestamp}', curr_time)
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
        ClassifierPipeline.logger.info('MERGING METADATA INTO MAIN DATAFRAME...')
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
        ClassifierPipeline.logger.info('MERGING METADATA: COMPLETE!')
        return df

    def engineer_features(self: 'ClassifierPipeline', df: 'DataFrame', train:bool=False) -> 'DataFrame':
        '''
        Apply feature engineering to the DataFrame, handling metadata.

            PROCEDURE:
            (1) Merging metadata (directors, writers) to include relevant categorical features.
            (2) For TRAIN: convert booleam string labels to binary numeric.

            (2) Handling categorical features via indexing.
            (3) Assembling numerical features into a vector for RandomForestClassifier.

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
        df = self.merge_metadata_into_df(df)
        # (2) For TRAIN: convert string labels to binary numeric, where: "True" -> 1, "False" -> 0.
        if train:
            df = df.withColumn('label', col('label').cast('double'))
        df.show(10, False)
        # (2) Handle categorical variables by indexing them.
        for col_name in self.CATEGORICAL_COLS:
            output_col_name = f'{col_name}_index'
            indexer = StringIndexer(inputCol=col_name, outputCol=output_col_name).setHandleInvalid('keep')
            df = indexer.fit(df).transform(df)
            df = df.drop(col_name)
            # Append the new output column into set of features.
            self.feature_cols.add(output_col_name)
        print(self.feature_cols)
        df.show(10, False)

        ###### test from here

        # # TODO: (3)
        # feature_cols = [col_name for col_name in df.columns if col_name != "label"]
        # assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        # df = assembler.transform(df)
        # # Standardize numeric features
        # scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
        # scaler_model = scaler.fit(df)
        # df = scaler_model.transform(df)
        ###### to here
        # indexer_writers = StringIndexer(inputCol='writers', outputCol='writers_index', handleInvalid='keep')
        # indexer_directors = StringIndexer(inputCol='directors', outputCol='directors_index', handleInvalid='keep')
        # df = indexer_writers.fit(df).transform(df)
        # df = indexer_directors.fit(df).transform(df)
        # (3) Assemble all feature columns into a single feature vector.
        # assembler = VectorAssembler(inputCols=self.FEATURE_COLS, outputCol='features')
        # df = assembler.transform(df)
        # (4) For TRAIN: convert string labels to binary numeric, where: "True" -> 1, "False" -> 0.
        # if train:
        #     df = df.withColumn('label', when(col('label') == 'True', 1).otherwise(0))
        #     df = df.withColumn('label', col('label').cast(IntegerType()))
        # Drop unused columns (they were converted to indexed features).
        # df = df.drop('writers', 'directors')
        # df.show(20, False)
        return df

    def train_model(self: 'ClassifierPipeline', df: 'DataFrame') -> 'PipelineModel':
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
        # Set Spark ML Pipeline to encapsulate training steps.
        spark_pipeline = SparkPipeline(stages=[self.rf_classifier])
        model = spark_pipeline.fit(df)
        # Save trained model.
        model.write().overwrite().save(self.model_path)
        ClassifierPipeline.logger.info(f'TRAINED MODEL SAVED TO: "{self.model_path}".')
        return model

    def __call__(self: 'ClassifierPipeline') -> None:
        '''
        Main method to call the pipeline functionalities.

            PROCEDURE:
            (1) Load data.
            (2) Pre-process data.
            (3) Apply feature engineering.
            (4) Train RF classifier model.
            (5) TODO: Predict via model.
        '''
        # (1) Load data.
        ClassifierPipeline.logger.info('LOADING DATA...')
        self.load_data()
        ClassifierPipeline.logger.info('DATA: LOADED!')
        # (2) Pre-process data: TRAIN, VAL, TEST.
        ClassifierPipeline.logger.info('PRE-PROCESSING DATA...')
        train_df = self.preprocess(self.data_dict['train'], train=True)
        val_df = self.preprocess(self.data_dict['val'])
        test_df = self.preprocess(self.data_dict['test'])
        ClassifierPipeline.logger.info('DATA PRE-PROCESSING: COMPLETE!')
        # (3) Apply feature engineering procedures.
        ClassifierPipeline.logger.info('APPLYING FEATURE ENGINEERING...')
        train_df = self.engineer_features(train_df, train=True)
        # val_df = self.engineer_features(val_df)
        # test_df = self.engineer_features(test_df)
        ClassifierPipeline.logger.info('FEATURE ENGINEERING: COMPLETE!')
        # (4) TODO: Train the model.
        ClassifierPipeline.logger.info('TRAINING RANDOM FORREST MODEL...')
        # rf_model = self.train_model(train_df)
        ClassifierPipeline.logger.info('MODEL TRAINING: COMPLETE!')


if __name__ == '__main__':
    classifier_pipe = ClassifierPipeline()
    classifier_pipe()

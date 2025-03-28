'''
--------------------------------------------------------------
FILE:
    movie_pipeline/classifier_pipeline.py

INFO:
    Main pipeline file orchestrating the full classifier pipeline
    workflow. From data loading to model predictions. Can be applied
    either for both TRAINING and/or PREDICTING tasks.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


from classifier_model import ClassifierModel
from data_utils import DataUtils
from llm_predictor import LLMGenrePredictor
from logger import get_logger
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, floor, concat, lit


class ClassifierPipeline:
    '''
    -------------------------
    ClassifierPipeline - Main class orchestrating the overall workflow, utilizing
                         DataUtils and Classifier classes:
                         (1) Loading data (from CSV/JSON).
                         (2) Pre-processing data (missing values, data types, etc.).
                         (3) Feature engineering (metadata, genre predictions, etc.).
                         (4) Train the classifier model.
                         (5) Run predictions/evaluations.
    -------------------------
    '''

    # Class logger for Spark operations.
    logger = get_logger(__name__)
    # Standard cols to follow in data.
    NUMERIC_COLS = {'runtimeMinutes', 'numVotes', 'startYear', 'endYear'}
    CATEGORIC_COLS = {'writer', 'director', 'genre', 'decade'}

    def __init__(
            self: 'ClassifierPipeline',
            model_path: str,
            results_path: str,
            directing_json_path: str,
            writing_json_path: str,
            llm_cache_path_train: str,
            llm_cache_path_test: str,
            llm_name: str='gemma3:4b'
        ) -> None:
        '''
        Initialize the ClassifierPipeline class.

            Parameters:
            -----------
            model_path : str
                The path to save the trained model.
            results_path : str
                The path to save the results and evaluation metrics.
            directing_json_path : str
                The path to the directing metadata JSON file.
            writing_json_path : str
                The path to the writing metadata JSON file.
            llm_cache_path_train : str
                The path to the LLM cache for training.
            llm_cache_path_test : str
                The path to the LLM cache for predictions.
            llm_name : str
                The name of the LLM model for genre predictions.
        '''
        self.data_dict = {}
        # Assign standard paths.
        self.model_path = model_path
        self.results_path = results_path
        self.cache_path_train = llm_cache_path_train
        self.cache_path_test = llm_cache_path_test
        self.directing_json_path = directing_json_path
        self.writing_json_path = writing_json_path
        # Initialize feature cols that will be further extended.
        self.feature_cols = {'runtimeMinutes', 'numVotes', 'popularity', 'budget', 'revenue'}
        # Initialize Spark session.
        self.spark = SparkSession.builder \
            .appName('MoviePipeline') \
            .config('spark.driver.memory', '8g') \
            .config('spark.executor.memory', '8g') \
            .config('spark.memory.offHeap.enabled', 'true') \
            .config('spark.memory.offHeap.size', '2g') \
            .getOrCreate()
        # LLM genre predictor setup.
        self.genre_predictor_llm = LLMGenrePredictor(
            batch_size=20,
            model_name=llm_name,
            spark=self.spark
        )
        # Initialize feature scaler and classifier model.
        self.scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features',
            withStd=True,
            withMean=False
        )
        self.indexer_models = {}
        self.classifier_model = ClassifierModel(
            model_path=model_path,
            feature_cols=self.feature_cols
        )
        # Initialize the mean dictionary for numeric columns.
        self.mean_dict = {
            'runtimeMinutes': None,
            'numVotes': None,
            'popularity': None,
            'budget': None,
            'revenue': None
        }

    def _load_data(
            self: 'ClassifierPipeline',
            train_csv_path: str,
            test_csv_path: str,
            extra_csv_path: str=None
        ) -> None:
        '''
        Load data from CSV and JSON files into Spark DataFrames
        to build a full-data dictionary.

            Parameters:
            -----------
            train_csv_path : str
                The path to load train CSV data.
            extra_csv_path: str
                The path to load extra CSV data (optional).
            test_csv_path : str
                The path to load test CSV data.
        '''
        # Load JSON files with metadata for directing and writing.
        writing_df = DataUtils.load_json(self.spark, self.writing_json_path)
        # For directing: merge movie and director dictionaries (since unstructured).
        directing_df = DataUtils.merge_directing_json(self.spark, self.directing_json_path)
        # For train data: detect all CSV files and load them accordingly.
        train_df = DataUtils.load_train_csv(self.spark, train_csv_path)
        # Load extra data source.
        if extra_csv_path:
            extra_df = DataUtils.load_csv(self.spark, extra_csv_path)
        # For test data, load standard data; e.g. val/test CSV data and means.
        test_df = DataUtils.load_csv(self.spark, test_csv_path)
        self.data_dict = {
            'train_data': train_df,
            'test_data': test_df,
            'directing': directing_df,
            'writing': writing_df
        }
        if extra_csv_path:
            self.data_dict['extra_data'] = extra_df

    def _preprocess(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Preprocess the data by handling missing values, fixing data types,
        normalizing text fields, handling outliers, etc. Utilizes various
        atomic functions from DataUtils.

            PROCEDURE:
            (1) Pre-process numeric columns by converting them to double type.
            (2) Inject mean values for missing records: runtimeMinutes and numVotes.
            (3) Ensure startYear ≤ endYear and handle missing values.
            (4) Normalize text fields ("primaryTitle", "originalTitle").

            Parameters:
            -----------
            df : DataFrame
                The input dataframe to preprocess.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The preprocessed DataFrame
        '''
        # (1) Pre-process numeric columns.
        ClassifierPipeline.logger.info('Pre-processing numeric columns...')
        df = DataUtils.preprocess_numeric_cols(df, self.NUMERIC_COLS)
        # (2) Inject mean values for missing records: "runtimeMinutes" and "numVotes".
        for col_name in ['runtimeMinutes', 'numVotes']:
            # From TRAIN: find means for numeric columns (for further injection).
            # For other sets, assign pre-computed values, since TRAIN runs first.
            if train:
                # Only calculate if we have no pre-computed values.
                if self.mean_dict[col_name] is None:
                    self.mean_dict[col_name] = DataUtils.calc_mean_col(df, col_name)
            # Inject TRAIN mean values into NULL fields.
            df = df.withColumn(
                col_name, when(col(col_name).isNull(), self.mean_dict[col_name]).otherwise(col(col_name))
            )
        # (3) Ensure startYear ≤ endYear and handle missing values.
        df = df.withColumn('startYear', when(col('startYear').isNull(), col('endYear')).otherwise(col('startYear')))
        df = df.withColumn('endYear', when(col('endYear').isNull(), col('startYear')).otherwise(col('endYear')))
        df = df.withColumn('endYear', when(col('endYear') < col('startYear'), col('startYear')).otherwise(col('endYear')))
        # (4) Normalize text title fields and handle missing records.
        ClassifierPipeline.logger.info('Pre-processing textual columns...')
        df = DataUtils.normalize_text_cols(df)
        return df

    def _preprocess_extra(self: 'ClassifierPipeline', id_col: str, target_cols: list) -> None:
        '''
        Pre-process the extra data by filtering only the relevant columns.

            Parameters:
            -----------
            id_col : str
                Name of the original ID column to be renamed as "tconst".
            target_cols : list
                List of target columns to keep.
        '''
        df = self.data_dict['extra_data']
        ClassifierPipeline.logger.info(f'Preprocessing extra data, keeping only columns: {target_cols}...')
        # Select only the target columns that exist in the current DF.
        filter_cols = target_cols + [id_col]
        filtered_df = df.select(filter_cols)
        # Rename "imdb_id" column to "tconst" to match other dataframes
        filtered_df = filtered_df.withColumnRenamed(id_col, 'tconst')
        filtered_df.show(10, False)
        # Ensure numeric for target cols.
        filtered_df = DataUtils.preprocess_numeric_cols(
            filtered_df,
            cols=target_cols,
            num_type='double'
        )
        # Find means among the data.
        for col_name in target_cols:
            # Only calculate if we have no pre-computed values.
            if self.mean_dict[col_name] is None:
                # Filter for non-zero values and calculate mean for these.
                non_zero_df = filtered_df.filter(col(col_name) > 0.0)
                self.mean_dict[col_name] = DataUtils.calc_mean_col(non_zero_df, col_name)
        self.data_dict['extra_data'] = filtered_df

    def _merge_metadata_into_df(self: 'ClassifierPipeline', df: 'DataFrame') -> 'DataFrame':
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

    def merge_extra_data_into_df(self: 'ClassifierPipeline', df: 'DataFrame') -> 'DataFrame':
        '''
        Left-join extra data intom the main DataFrame.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to merge the extra data into.

            Returns:
            -----------
            df : DataFrame
                DataFrame including extra data columns.
        '''
        extra_df = self.data_dict['extra_data']
        df = df.join(extra_df, df['tconst'] == extra_df['tconst'], 'left')
        df = df.drop(extra_df['tconst'])
        return df

    def _predict_genres_llm(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Predict movie genres using LLM and add them to the DataFrame.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to predict genres for.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The main DataFrame with added genre predictions.
        '''
        # Get genre predictions (from cache or by generating new ones).
        llm_cache = self.cache_path_train if train else self.cache_path_test
        genre_df = DataUtils.load_or_create_genre_predictions(
            self.spark, df, self.genre_predictor_llm, llm_cache
        )
        # Join the predictions with the main dataframe.
        df = df.join(genre_df, on='tconst', how='left')
        # Handle missing genres.
        df = df.withColumn('genre', when(col('genre').isNull(), 'unknown').otherwise(col('genre')))
        return df

    def _engineer_features(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Apply feature engineering to the DataFrame, handling metadata.

            PROCEDURE:
            (1) Merge metadata (directors, writers) to include relevant categorical features.
            (2) Merge extra columns provided within extra data and inject their full-collcetion means.
            (3) For TRAIN: convert boolean string labels to binary numeric.
            (4) Utilize LLM to create new synthetic feature: "genre".
            (5) Handle "decade" feature by processing "startYear".
            (6) Drop initial year columns, due to redundancy.
            (7) Handling categorical features via string indexing.
            (8) Handling missing values.
            (9) Assemble features into single vector.
            (10) Standardize numeric features.

            Parameters:
            -----------
            df : DataFrame
                The input dataframe to be transformed.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The transformed DataFrame ready for training.
        '''
        # (1) Join movie dataset with writing and directing metadata.
        ClassifierPipeline.logger.info('Merging METADATA JSON into main DataFrame...')
        df = self._merge_metadata_into_df(df)
        # (2) Merge extra data columns. Join on "tconst" column from main dataframe and extra data.
        ClassifierPipeline.logger.info('Merging columns from EXTRA DATA into main DataFrame...')
        df = self.merge_extra_data_into_df(df)
        for col_name in ['popularity', 'budget', 'revenue']:
            # Apply mean values from the dictionary to replace nulls and zeros
            df = df.withColumn(
                col_name,
                when((col(col_name).isNull()) | (col(col_name) == 0.0),
                    self.mean_dict[col_name]).otherwise(col(col_name))
            )
        ClassifierPipeline.logger.info('Initial DataFrame after metadata merge:')
        df.show(20, False)
        # (3) For TRAIN: convert string labels to binary numeric, where: "True" -> 1.0, "False" -> 0.0.
        if train:
            ClassifierPipeline.logger.info('Converting boolean "label" field to binary for TRAIN...')
            df = df.withColumn('label', col('label').cast('double'))
        # (4) Apply LLM to introduce new synthetic feature: "genre".
        ClassifierPipeline.logger.info('Generating synthetic genre features via LLM...')
        df = self._predict_genres_llm(df, train=train)
        # (5) Handle "decade" feature by processing "startYear".
        ClassifierPipeline.logger.info('Adding "decade" feature based on "startYear" records...')
        df = df.withColumn('startYear', when(col('startYear').isNull(), 2000).otherwise(col('startYear')))
        df = df.withColumn('decade', concat(floor(col('startYear')/10).cast('int')*10, lit('s')))
        # ClassifierPipeline.logger.info('Decade distribution:')
        # df.groupBy('decade').count().orderBy('decade').show(20, False)
        # (6) Drop "startYear" and "endYear" cols. The first has been used for "decade",
        #     while the latter is removed due to high percentage of missing values (around 90%).
        ClassifierPipeline.logger.info('Dropping "startYear" and "endYear" columns...')
        for col_name in ['startYear', 'endYear']:
            if col_name in df.columns:
                df = df.drop(col_name)
        # (7) Handle categorical variables by applying StringIndexer.
        ClassifierPipeline.logger.info('String indexing categorical features...')
        for col_name in self.CATEGORIC_COLS:
            if train:
                # When training, fit new indexers and store them.
                df, output_col_name, indexer_model = DataUtils.string_index_col(
                    df, col_name, return_model=True
                )
                self.indexer_models[col_name] = indexer_model
            else:
                # When predicting, use stored indexers from training.
                output_col_name = f'{col_name}_index'
                df = self.indexer_models[col_name].transform(df)
                df = df.drop(col_name)
            self.feature_cols.add(output_col_name)
        ClassifierPipeline.logger.info(f'Feature column names: {self.feature_cols}')
        # (8) Handle missing values.
        df = df.fillna(0)
        # (9) Assemble all feature columns into a single vector.
        ClassifierPipeline.logger.info('Assembling features into a single vector...')
        assembler = VectorAssembler(inputCols=list(self.feature_cols), outputCol='features')
        df = assembler.transform(df)
        # (10) Standardize numeric features.
        ClassifierPipeline.logger.info('Standardizing numerical features...')
        scaler_model = self.scaler.fit(df)
        df = scaler_model.transform(df)
        ClassifierPipeline.logger.info('Final DataFrame after all FE steps:')
        df.show(20, False)
        return df

    def run(
            self: 'ClassifierPipeline',
            train_csv_path: str,
            extra_data_csv_path: str,
            test_csv_path: str,
            output_txt_path: str,
        ) -> None:
        '''
        Main method to execute full pipeline workflow: training + predicting.

            PROCEDURE:
            (1) Load data.
            (2) Pre-process data.
            (3) Apply feature engineering.
            (4) Train classifier model.
            (5) Run predictions on the in-memory model.

            Parameters:
            -----------
            train_csv_path : str
                The path to the training CSV data.
            test_csv_path : str
                The path to the test CSV data.
            extra_data_csv_path : str
                The path to the extra data CSV.
            output_txt_path : str
                The path to save the predictions.
        '''
        # (1) Load data: TRAIN and TEST.
        ClassifierPipeline.logger.info('***(1) LOADING DATA...***')
        self._load_data(
            train_csv_path=train_csv_path,
            extra_csv_path=extra_data_csv_path,
            test_csv_path=test_csv_path
        )
        train_df = self.data_dict['train_data']
        test_df = self.data_dict['test_data']
        ClassifierPipeline.logger.info('***(1) DATA: LOADED!***')
        # (2) Pre-process data.
        ClassifierPipeline.logger.info('***(2) PRE-PROCESSING DATA...***')
        # TRAIN and TEST data.
        ClassifierPipeline.logger.info('***PRE-PROCESS: TRAIN SET***')
        train_df = self._preprocess(train_df, train=True)
        ClassifierPipeline.logger.info('***PRE-PROCESS: TEST SET***')
        test_df = self._preprocess(test_df)
        # EXTRA data.
        ClassifierPipeline.logger.info('***PRE-PROCESS: EXTRA DATA SET***')
        EXTRA_DATA_TARGET_COLS = ['budget', 'revenue', 'popularity']
        self._preprocess_extra(id_col='imdb_id', target_cols=EXTRA_DATA_TARGET_COLS)
        ClassifierPipeline.logger.info('***(2) DATA PRE-PROCESSING: COMPLETE!***')
        # (3) Apply feature engineering procedures.
        ClassifierPipeline.logger.info('***(3) APPLYING FEATURE ENGINEERING...***')
        ClassifierPipeline.logger.info('***FE: TRAIN SET***')
        train_df = self._engineer_features(train_df, train=True)
        self.classifier_model.update_feature_cols(self.feature_cols)
        ClassifierPipeline.logger.info('***FE: TEST SET***')
        test_df = self._engineer_features(test_df)
        ClassifierPipeline.logger.info('***(3) FEATURE ENGINEERING: COMPLETE!***')
        # (4) Train the model and save.
        ClassifierPipeline.logger.info('***(4) TRAINING CLASSIFIER MODEL...***')
        self.classifier_model.train(train_df)
        ClassifierPipeline.logger.info('***(4) MODEL TRAINING: COMPLETE!***')
        # (5) Run predictions on the model.
        ClassifierPipeline.logger.info('***(5) RUNNING PREDICTIONS...***')
        predictions_df = self.classifier_model.predict(test_df)
        DataUtils.save_preds_txt(predictions_df, output_txt_path)
        ClassifierPipeline.logger.info('***(5) PREDICTIONS: COMPLETE!***')

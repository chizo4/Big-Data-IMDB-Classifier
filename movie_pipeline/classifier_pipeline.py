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


from data_utils import DataUtils
import json
from llm_predictor import LLMGenrePredictor
from logger import get_logger
import os
from pathlib import Path
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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
                         (4) Train the Random Forest Classifier model.
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
            llm_cache_path: str,
            llm_name: str='gemma3:4b',
            train: bool=False
        ) -> None:
        '''
        Initialize the ClassifierPipeline class.

            Parameters:
            -----------
            model_path : str
                The path to save the trained RF model.
            results_path : str
                The path to save the results and evaluation metrics.
            directing_json_path : str
                The path to the directing metadata JSON file.
            writing_json_path : str
                The path to the writing metadata JSON file.
            llm_cache_path : str
                The path to the LLM genre predictions cache.
            llm_name : str
                The name of the LLM model for genre predictions.
            train : bool
                Flag to determine the mode of the pipeline (train/predict).
        '''
        self.data_dict = {}
        # Default flag for train/predict mode.
        self.train = train
        # Assign standard paths.
        self.model_path = model_path
        self.results_path = results_path
        self.cache_path = llm_cache_path
        self.directing_json_path = directing_json_path
        self.writing_json_path = writing_json_path
        # Initialize feature cols that will be further extended.
        self.feature_cols = {'runtimeMinutes', 'numVotes'}
        # Initialize Spark session.
        self.spark = SparkSession.builder.appName('MoviePipeline').getOrCreate()
        # LLM genre predictor setup.
        self.genre_predictor_llm = LLMGenrePredictor(
            batch_size=20,
            model_name=llm_name,
            spark=self.spark
        )
        # Initialize RF classifier and related scaler.
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
        # Initialize the median dictionary for numeric columns.
        self.median_dict = {
            'runtimeMinutes': None,
            'numVotes': None
        }
        self.medians_file_path = Path(os.path.dirname(llm_cache_path)) / 'medians.json'

    def _load_data(self: 'ClassifierPipeline', csv_path: str) -> None:
        '''
        Load data from CSV and JSON files into Spark DataFrames
        to build a full-data dictionary.

            Parameters:
            -----------
            file_path : str
                The path to load CSV data.
        '''
        # Load JSON files with metadata for directing and writing.
        writing_df = DataUtils.load_json(self.spark, self.writing_json_path)
        # For directing: merge movie and director dictionaries (since unstructured).
        directing_df = DataUtils.merge_directing_json(self.spark, self.directing_json_path)
        # For train data: detect all CSV files and load them accordingly.
        if self.train:
            df = DataUtils.load_train_csv(self.spark, csv_path)
        else:
            # Otherwise - loading standard data; e.g. val/test CSV data and medians
            df = DataUtils.load_csv(self.spark, csv_path)
            self._load_medians()
        self.data_dict = {
            'data': df,
            'directing': directing_df,
            'writing': writing_df
        }

    def _load_medians(self: 'ClassifierPipeline') -> None:
        '''
        Load median values from JSON.
        '''
        try:
            with open(self.medians_file_path, 'r') as f:
                self.median_dict = json.load(f)
                ClassifierPipeline.logger.info(f'Loaded pre-computed medians: {self.median_dict}')
        except Exception as e:
            ClassifierPipeline.logger.warning(f'Failed to load medians: {e}')

    def _save_medians(self: 'ClassifierPipeline') -> None:
        '''
        Save median values into JSON. In training mode.
        '''
        try:
            with open(self.medians_file_path, 'w') as f:
                json.dump(self.median_dict, f)
                ClassifierPipeline.logger.info(f'Saved medians to "{self.medians_file_path}".')
        except Exception as e:
            ClassifierPipeline.logger.warning(f'Failed to save medians: {e}')

    def _preprocess(self: 'ClassifierPipeline', df: 'DataFrame') -> 'DataFrame':
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

            Returns:
            -----------
            df : DataFrame
                The preprocessed DataFrame
        '''
        # (1) Pre-process numeric columns.
        ClassifierPipeline.logger.info('Pre-processing numeric columns...')
        df = DataUtils.preprocess_numeric_cols(df, self.NUMERIC_COLS)
        # (2) Inject median values for missing records: "runtimeMinutes" and "numVotes".
        for col_name in ['runtimeMinutes', 'numVotes']:
            # From TRAIN: find medians for numeric columns (for further injection).
            # For other sets, assign pre-computed values, since TRAIN runs first.
            if self.train:
                self.median_dict[col_name] = DataUtils.calc_median_col(df, col_name)
            # Inject TRAIN median values into NULL fields.
            df = df.withColumn(
                col_name, when(col(col_name).isNull(), self.median_dict[col_name]).otherwise(col(col_name))
            )
        # Save medians for later prediction mode.
        if self.train:
            self._save_medians()
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

    def _engineer_features(self: 'ClassifierPipeline', df: 'DataFrame') -> 'DataFrame':
        '''
        Apply feature engineering to the DataFrame, handling metadata.

            PROCEDURE:
            (1) Merging metadata (directors, writers) to include relevant categorical features.
            (2) For TRAIN: convert boolean string labels to binary numeric.
            (3) Utilize LLM to create new synthetic feature: "genre".
            (4) Handle "decade" feature by processing "startYear".
            (5) Drop initial year columns, due to redundancy.
            (6) Handling categorical features via string indexing.
            (7) Handling missing values.
            (8) Assemble features into single vector.
            (9) Standardize numeric features.

            Parameters:
            -----------
            df : DataFrame
                The input dataframe to be transformed.

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
        # (2) For TRAIN: convert string labels to binary numeric, where: "True" -> 1.0, "False" -> 0.0.
        if self.train:
            ClassifierPipeline.logger.info('Converting boolean "label" field to binary for TRAIN...')
            df = df.withColumn('label', col('label').cast('double'))
        # (3) Apply LLM to introduce new synthetic feature: "genre".
        ClassifierPipeline.logger.info('Generating synthetic genre features via LLM...')
        df = self.predict_genres_llm(df)
        # (4) Handle "decade" feature by processing "startYear".
        ClassifierPipeline.logger.info('Adding "decade" feature based on "startYear" records...')
        df = df.withColumn('startYear', when(col('startYear').isNull(), 2000).otherwise(col('startYear')))
        df = df.withColumn('decade', concat(floor(col('startYear')/10).cast('int')*10, lit('s')))
        # ClassifierPipeline.logger.info('Decade distribution:')
        # df.groupBy('decade').count().orderBy('decade').show(20, False)
        # (5) Drop "startYear" and "endYear" cols. The first has been used for "decade",
        #     while the latter is removed due to high percentage of missing values (around 90%).
        ClassifierPipeline.logger.info('Dropping "startYear" and "endYear" columns...')
        for col_name in ['startYear', 'endYear']:
            if col_name in df.columns:
                df = df.drop(col_name)
        # (6) Handle categorical variables by applying StringIndexer.
        ClassifierPipeline.logger.info('String indexing categorical features...')
        for col_name in self.CATEGORIC_COLS:
            df, output_col_name = DataUtils.string_index_col(df, col_name)
            self.feature_cols.add(output_col_name)
        ClassifierPipeline.logger.info(f'Feature column names: {self.feature_cols}')
        # (7) Handle missing values.
        df = df.fillna(0)
        # (0) Assemble all feature columns into a single vector.
        ClassifierPipeline.logger.info('Assembling features into a single vector...')
        assembler = VectorAssembler(inputCols=list(self.feature_cols), outputCol='features')
        df = assembler.transform(df)
        # (9) Standardize numeric features.
        ClassifierPipeline.logger.info('Standardizing numerical features...')
        scaler_model = self.scaler.fit(df)
        df = scaler_model.transform(df)
        ClassifierPipeline.logger.info('Final DataFrame after all FE steps:')
        df.show(10, False)
        return df

    def _train_model(self: 'ClassifierPipeline', train_df: 'DataFrame') -> 'SparkPipeline':
        '''
        Train the RandomForestClassifier model using the engineered features,
        and save it for further use.

            Parameters:
            -----------
            df : DataFrame
                The preprocessed and feature-engineered DataFrame for training.

            Returns:
            -----------
            model : SparkPipeline
                The trained model.
        '''
        # train_df, val_df = train_df.randomSplit([0.8, 0.2], seed=42) #TEMP!
        # # Set Spark ML Pipeline to encapsulate training steps.
        # spark_pipeline = SparkPipeline(stages=[self.rf_classifier])
        # model = spark_pipeline.fit(train_df) #df
        # # Save trained model.
        # model.write().overwrite().save(self.model_path)
        # ClassifierPipeline.logger.info(f'TRAINED RF model saved to: "{self.model_path}".')
        # # TEMP: some eval debugs
        # y_pred = model.transform(val_df)
        # evaluator = MulticlassClassificationEvaluator(
        #     labelCol="label",
        #     predictionCol="prediction",
        #     metricName="accuracy"
        # )
        # accuracy = evaluator.evaluate(y_pred)
        # print("Model Accuracy:", accuracy)
        # return model
        # TODO: pick one model for now. then perform small hyperparam search.
        train_df, val_df = train_df.randomSplit([0.8, 0.2], seed=42)
        # Tune hyperparameters
        paramGrid = (ParamGridBuilder()
                    .addGrid(self.rf_classifier.numTrees, [100, 300, 500])
                    .addGrid(self.rf_classifier.maxDepth, [8, 12, 16])
                    # .addGrid(self.rf_classifier.featureSubsetStrategy, ['sqrt', 'log2'])
                    .build())

        evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
        # Use CrossValidator for tuning
        crossval = CrossValidator(estimator=self.rf_classifier,
                                estimatorParamMaps=paramGrid,
                                evaluator=evaluator,
                                numFolds=2, # 3-fold cross-validation
                                parallelism=4) #parallelism for faster compute. spark feature here.
        model = crossval.fit(train_df)
        best_model = model.bestModel
        # Save the best model
        best_model.write().overwrite().save(self.model_path)
        # Evaluate best model
        y_pred = best_model.transform(val_df)
        accuracy = evaluator.evaluate(y_pred)
        print(f"Best Model Accuracy: {accuracy}")
        return best_model

    def run_train(self: 'ClassifierPipeline', train_csv_path: str,) -> None:
        '''
        Main method to execute pipeline training.

            PROCEDURE:
            (1) Load data.
            (2) Pre-process data.
            (3) Apply feature engineering.
            (4) Train RF classifier model and save it.
        '''
        # (1) Load data.
        ClassifierPipeline.logger.info('***(1) LOADING DATA...***')
        self._load_data(csv_path=train_csv_path)
        train_df = self.data_dict['data']
        ClassifierPipeline.logger.info('***(1) DATA: LOADED!***')
        # (2) Pre-process data.
        ClassifierPipeline.logger.info('***(2) PRE-PROCESSING DATA...***')
        train_df = self._preprocess(train_df)
        ClassifierPipeline.logger.info('***(2) DATA PRE-PROCESSING: COMPLETE!***')
        # (3) Apply feature engineering procedures.
        ClassifierPipeline.logger.info('***(3) APPLYING FEATURE ENGINEERING...***')
        ClassifierPipeline.logger.info('***FE: TRAIN SET***')
        train_df = self._engineer_features(train_df)
        ClassifierPipeline.logger.info('***(3) FEATURE ENGINEERING: COMPLETE!***')
        # (4) Train the RF model and save.
        ClassifierPipeline.logger.info('***(4) TRAINING RANDOM FOREST CLASSIFIER...***')
        rf_model = self._train_model(train_df)
        ClassifierPipeline.logger.info('***(4) MODEL TRAINING: COMPLETE!***')

    def _predict_model(self: 'ClassifierPipeline', test_df: 'DataFrame', output_txt_path: str) -> None:
        '''
        Load the trained model, make predictions on the test data, and save predictions to a text file.

            Parameters:
            -----------
            test_df : DataFrame
                The preprocessed and feature-engineered test DataFrame.
            output_txt_path : str
                Path where the prediction results should be saved.
        '''
        # Load the trained model.
        ClassifierPipeline.logger.info(f'Loading model from: "{self.model_path}".')
        try:
            loaded_model = SparkPipeline.load(self.model_path)
        except Exception as e:
            ClassifierPipeline.logger.error(f'Error loading model: {str(e)}')
            raise RuntimeError(f'Could not load model from "{self.model_path}".')
        # Make predictions on the test data.
        ClassifierPipeline.logger.info('Running predictions on test data...')
        predictions_df = loaded_model.transform(test_df)
        # Convert and save predictions into TXT file.
        DataUtils.save_preds_txt(predictions_df, output_txt_path)

    def run_predict(self: 'ClassifierPipeline', input_csv_path: str, output_txt_path: str) -> None:
        '''
        Main method to execute pipeline predictions.

            PROCEDURE:
            (1) Load data.
            (2) Pre-process data.
            (3) Apply feature engineering.
            (4) Run predictions via loaded model.
        '''
        # (1) Load data.
        ClassifierPipeline.logger.info('***(1) LOADING DATA...***')
        self._load_data(csv_path=input_csv_path)
        test_df = self.data_dict['data']
        # (2) Pre-process data.
        ClassifierPipeline.logger.info('***(2) PRE-PROCESSING DATA...***')
        test_df = self._preprocess(test_df)
        ClassifierPipeline.logger.info('***(2) DATA PRE-PROCESSING: COMPLETE!***')
        # (3) Apply feature engineering procedures.
        ClassifierPipeline.logger.info('***(3) APPLYING FEATURE ENGINEERING...***')
        ClassifierPipeline.logger.info('***FE: TEST SET***')
        test_df = self._engineer_features(test_df)
        ClassifierPipeline.logger.info('***(3) FEATURE ENGINEERING: COMPLETE!***')
        # (4) Run predictions via loaded model.
        ClassifierPipeline.logger.info('***(4) RUNNING MODEL PREDICTIONS...***')
        self._predict_model(test_df, output_txt_path)
        ClassifierPipeline.logger.info('***(4) RUNNING MODEL PREDICTIONS: COMPLETE!***')

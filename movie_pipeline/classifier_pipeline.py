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
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, floor, concat, lit, count, avg, countDistinct, udf


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
        self.feature_cols = {'runtimeMinutes', 'numVotes'}
        # Initialize Spark session.
        self.spark = SparkSession.builder.appName('MoviePipeline').getOrCreate()
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
        # Initialize the median dictionary for numeric columns.
        self.median_dict = {
            'runtimeMinutes': None,
            'numVotes': None
        }

        # New: Initialize collaborative feature models
        self.historical_df = None
        self.dir_writer_collab_model = None
        self.director_success_model = None 
        self.writer_success_model = None
        self.team_ratings_model = None
        self.company_success_model = None
        self.stability_index_model = None
        self.genre_success_model = None

    def _load_data(self: 'ClassifierPipeline', train_csv_path: str, test_csv_path: str) -> None:
        '''
        Load data from CSV and JSON files into Spark DataFrames
        to build a full-data dictionary.

            Parameters:
            -----------
            train_csv_path : str
                The path to load train CSV data.
            test_csv_path : str
                The path to load test CSV data.
        '''
        # Load JSON files with metadata for directing and writing.
        writing_df = DataUtils.load_json(self.spark, self.writing_json_path)
        # For directing: merge movie and director dictionaries (since unstructured).
        directing_df = DataUtils.merge_directing_json(self.spark, self.directing_json_path)
        # For train data: detect all CSV files and load them accordingly.
        train_df = DataUtils.load_train_csv(self.spark, train_csv_path)
        # For test data, load standard data; e.g. val/test CSV data and medians.
        test_df = DataUtils.load_csv(self.spark, test_csv_path)
        self.data_dict = {
            'train_data': train_df,
            'test_data': test_df,
            'directing': directing_df,
            'writing': writing_df
        }

    def _preprocess(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
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
        # (2) Inject median values for missing records: "runtimeMinutes" and "numVotes".
        for col_name in ['runtimeMinutes', 'numVotes']:
            # From TRAIN: find medians for numeric columns (for further injection).
            # For other sets, assign pre-computed values, since TRAIN runs first.
            if train:
                # Only calculate if we have no pre-computed values.
                if self.median_dict[col_name] is None:
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

    def _predict_collab_count(self: 'ClassifierPipeline', df: 'DataFrame', train: bool = False) -> 'DataFrame':
        '''
        Compute or load the number of past collaborations between each director-writer pair.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to compute the director-writer collaboration count.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The main DataFrame with added collaboration count feature.
        '''
        if train:
            # Compute collaboration count from historical data
            collab_df = df.groupBy("director", "writer").agg(count("*").alias("collab_count"))
            # Store for later use
            self.dir_writer_collab_model = collab_df
        else:
            # Use stored model from training
            collab_df = self.dir_writer_collab_model

        # Join the computed collaboration count back to the main dataframe
        df = df.join(collab_df, on=["director", "writer"], how="left")

        # Fill missing values (new director-writer pairs that haven't collaborated before)
        df = df.withColumn("collab_count", when(col("collab_count").isNull(), 0).otherwise(col("collab_count")))

        return df

    def _predict_director_success(self: 'ClassifierPipeline', df: 'DataFrame', train: bool = False) -> 'DataFrame':
        '''
        Compute or load the director's past success rate and number of films directed.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to compute the director's success rate.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The main DataFrame with added director success rate feature.
        '''
        if train:
            # Compute director success metrics
            director_success = df.groupBy("director").agg(
                count("*").alias("director_film_count"),
                avg(when(col("label") == 1.0, 1.0).otherwise(0.0)).alias("director_success_rate")
            )
            # Store for later use
            self.director_success_model = director_success
        else:
            # Use stored model from training
            director_success = self.director_success_model

        # Join the computed success rate back to the main dataframe
        df = df.join(director_success, on="director", how="left")

        # Compute global average success rate in case some directors have missing values
        avg_success = director_success.agg(avg("director_success_rate")).first()[0]
        
        # Fill missing values with global average
        df = df.withColumn("director_success_rate", 
                        when(col("director_success_rate").isNull(), avg_success)
                        .otherwise(col("director_success_rate")))

        # If director has no prior films, set default to 1
        df = df.withColumn("director_film_count", 
                        when(col("director_film_count").isNull(), 1)
                        .otherwise(col("director_film_count")))

        return df

    def _predict_writer_success(self: 'ClassifierPipeline', df: 'DataFrame', train: bool = False) -> 'DataFrame':
        '''
        Compute or load the writer's past success rate and number of films written.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to compute the writer's success rate.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The main DataFrame with added writer success rate feature.
        '''
        if train:
            # Compute writer success metrics
            writer_success = df.groupBy("writer").agg(
                count("*").alias("writer_film_count"),
                avg(when(col("label") == 1.0, 1.0).otherwise(0.0)).alias("writer_success_rate")
            )
            # Store for later use
            self.writer_success_model = writer_success
        else:
            # Use stored model from training
            writer_success = self.writer_success_model

        # Join the computed success rate back to the main dataframe
        df = df.join(writer_success, on="writer", how="left")

        # Compute global average success rate in case some writers have missing values
        avg_success = writer_success.agg(avg("writer_success_rate")).first()[0]
        
        # Fill missing values with global average
        df = df.withColumn("writer_success_rate", 
                        when(col("writer_success_rate").isNull(), avg_success)
                        .otherwise(col("writer_success_rate")))

        # If writer has no prior films, set default to 1
        df = df.withColumn("writer_film_count", 
                        when(col("writer_film_count").isNull(), 1)
                        .otherwise(col("writer_film_count")))

        return df

    def _predict_team_success(self: 'ClassifierPipeline', df: 'DataFrame', train: bool = False) -> 'DataFrame':
        '''
        Compute or load the director-writer team's past success rate.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to compute the director-writer success rate.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The main DataFrame with added team success rate feature.
        '''
        if train:
            # Compute the success rate for each director-writer pair
            team_ratings = df.groupBy("director", "writer").agg(
                avg(when(col("label") == 1.0, 1.0).otherwise(0.0)).alias("team_success_rate")
            )
            # Store for later use
            self.team_ratings_model = team_ratings
        else:
            # Use stored model from training
            team_ratings = self.team_ratings_model

        # Join the computed team success rate back to the main dataframe
        df = df.join(team_ratings, on=["director", "writer"], how="left")

        # Compute global average success rate in case some teams have missing values
        avg_team_success = team_ratings.agg(avg("team_success_rate")).first()[0]

        # Fill missing values with global average
        df = df.withColumn("team_success_rate", 
                        when(col("team_success_rate").isNull(), avg_team_success)
                        .otherwise(col("team_success_rate")))

        return df

    def _predict_company_success(self: 'ClassifierPipeline', df: 'DataFrame', train: bool = False) -> 'DataFrame':
        '''
        Compute or load the production company's historical success rate.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame to compute the company success rate.
            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The main DataFrame with added company success rate feature.
        '''
        if train:
            # Compute the number of films and success rate for each production company
            company_success = df.groupBy("production_company").agg(
                count("*").alias("company_film_count"),
                avg(when(col("label") == 1.0, 1.0).otherwise(0.0)).alias("company_success_rate")
            )
            # Store for later use
            self.company_success_model = company_success
        else:
            # Use stored model from training
            company_success = self.company_success_model

        # Join the computed success rate back to the main dataframe
        df = df.join(company_success, on="production_company", how="left")

        # Compute global average success rate in case some companies have missing values
        avg_company_success = company_success.agg(avg("company_success_rate")).first()[0]

        # Fill missing values with global average
        df = df.withColumn("company_success_rate", 
                        when(col("company_success_rate").isNull(), avg_company_success)
                        .otherwise(col("company_success_rate")))

        df = df.withColumn("company_film_count", 
                        when(col("company_film_count").isNull(), 1)
                        .otherwise(col("company_film_count")))

        return df
    
    def _predict_director_experience(self: 'ClassifierPipeline', df: 'DataFrame') -> 'DataFrame':
        '''
        Create a categorical experience level for directors based on number of previous films.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame with `director_film_count` column.

            Returns:
            -----------
            df : DataFrame
                DataFrame with added 'director_experience' feature.
        '''
        if 'director_film_count' in df.columns:
            df = df.withColumn("director_experience", 
                when(col("director_film_count") > 10, 2.0)
                .when(col("director_film_count") > 3, 1.0)
                .otherwise(0.0)
            )
        return df

    def _predict_company_size(self: 'ClassifierPipeline', df: 'DataFrame') -> 'DataFrame':
        '''
        Categorize production companies based on the number of films they’ve made.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame with `company_film_count`.

            Returns:
            -----------
            df : DataFrame
                The DataFrame with added 'company_size' feature.
        '''
        if 'company_film_count' in df.columns:
            df = df.withColumn("company_size", 
                when(col("company_film_count") > 50, 2.0)
                .when(col("company_film_count") > 10, 1.0)
                .otherwise(0.0)
            )
        return df

    def _predict_team_stability(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Compute the ratio of unique writers per director to measure team stability.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame with `director` and `writer`.

            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The DataFrame with added 'director_team_variability' feature.
        '''
        if 'director' in df.columns:
            ClassifierPipeline.logger.info('Computing director-team stability index...')

            if train:
                stability_index = df.groupBy("director").agg(
                    (countDistinct("writer") / count("*")).alias("director_team_variability")
                )
                self.stability_index_model = stability_index
            else:
                stability_index = self.stability_index_model

            df = df.join(stability_index, on="director", how="left")
            avg_variability = stability_index.agg(avg("director_team_variability")).first()[0]

            df = df.withColumn("director_team_variability", 
                when(col("director_team_variability").isNull(), avg_variability)
                .otherwise(col("director_team_variability"))
            )

        return df

    def _predict_genre_specialization(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Compute director's specialization in a genre based on past performance.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame with 'director' and 'genre'.

            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The DataFrame with 'genre_success_rate' and 'genre_count' features.
        '''
        if 'genre' in df.columns and 'director' in df.columns:
            ClassifierPipeline.logger.info('Computing genre specialization...')

            if train:
                genre_success = df.groupBy("director", "genre").agg(
                    count("*").alias("genre_count"),
                    avg(when(col("label") == 1.0, 1.0).otherwise(0.0)).alias("genre_success_rate")
                )
                self.genre_success_model = genre_success
            else:
                genre_success = self.genre_success_model

            df = df.join(genre_success, on=["director", "genre"], how="left")
            avg_genre_success = genre_success.agg(avg("genre_success_rate")).first()[0]

            df = df.withColumn("genre_success_rate", 
                when(col("genre_success_rate").isNull(), avg_genre_success)
                .otherwise(col("genre_success_rate"))
            )

            df = df.withColumn("genre_count", 
                when(col("genre_count").isNull(), 1)
                .otherwise(col("genre_count"))
            )

        return df

    def _predict_lead_actor_success(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Compute lead actor's historical success rate.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame with 'lead_actor'.

            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The DataFrame with 'lead_actor_success_rate'.
        '''
        if 'lead_actor' in df.columns:
            ClassifierPipeline.logger.info('Computing lead actor success rate...')

            if train:
                actor_success = df.groupBy("lead_actor").agg(
                    count("*").alias("actor_film_count"),
                    avg(when(col("label") == 1.0, 1.0).otherwise(0.0)).alias("lead_actor_success_rate")
                )
                self.actor_success_model = actor_success
            else:
                actor_success = self.actor_success_model

            df = df.join(actor_success, on="lead_actor", how="left")
            avg_actor_success = actor_success.agg(avg("lead_actor_success_rate")).first()[0]

            df = df.withColumn("lead_actor_success_rate", 
                when(col("lead_actor_success_rate").isNull(), avg_actor_success)
                .otherwise(col("lead_actor_success_rate"))
            )

        return df

    def _predict_genre_count_by_decade(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
        '''
        Compute the number of movies per genre per decade.

            Parameters:
            -----------
            df : DataFrame
                The main DataFrame with 'genre' and 'decade'.

            train : bool
                Flag indicating if the data is for training.

            Returns:
            -----------
            df : DataFrame
                The DataFrame with 'genre_count_by_decade'.
        '''
        if 'genre' in df.columns and 'decade' in df.columns:
            ClassifierPipeline.logger.info('Computing genre count by decade...')

            if train:
                genre_trends = df.groupBy("genre", "decade").agg(
                    count("*").alias("genre_count_by_decade")
                )
                self.genre_trends_model = genre_trends
            else:
                genre_trends = self.genre_trends_model

            df = df.join(genre_trends, on=["genre", "decade"], how="left")
            df = df.withColumn("genre_count_by_decade", 
                when(col("genre_count_by_decade").isNull(), 1)
                .otherwise(col("genre_count_by_decade"))
            )

        return df

    def _categorize_runtime(self: 'ClassifierPipeline', df: 'DataFrame', train: bool = False) -> 'DataFrame':
        """
        Categorize movies based on their runtime into short, medium, long, very_long,
        and convert to numerical index using StringIndexer.

        Parameters:
        -----------
        df : DataFrame
            The DataFrame containing 'runtimeMinutes'.

        train : bool
            Whether the DataFrame is from the training set.

        Returns:
        --------
        df : DataFrame
            Updated DataFrame with 'runtime_category_index'.
        """
        from pyspark.ml.feature import StringIndexer

        ClassifierPipeline.logger.info('Creating runtime category feature...')

        # Step 1: Assign runtime category
        df = df.withColumn("runtime_category", 
            when(col("runtimeMinutes") < 90, "short")
            .when(col("runtimeMinutes") < 120, "medium")
            .when(col("runtimeMinutes") < 150, "long")
            .otherwise("very_long")
        )

        # Step 2: Index the category into a numeric feature
        if train:
            self.runtime_indexer = StringIndexer(
                inputCol="runtime_category", 
                outputCol="runtime_category_index", 
                handleInvalid="keep"
            ).fit(df)
        df = self.runtime_indexer.transform(df)

        # Step 3: Add to feature columns
        self.feature_cols.add("runtime_category_index")

        return df

    def _engineer_features(self: 'ClassifierPipeline', df: 'DataFrame', train: bool=False) -> 'DataFrame':
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
        ClassifierPipeline.logger.info('Initial DataFrame after metadata merge:')
        df.show(10, False)
        # (2) For TRAIN: convert string labels to binary numeric, where: "True" -> 1.0, "False" -> 0.0.
        if train:
            ClassifierPipeline.logger.info('Converting boolean "label" field to binary for TRAIN...')
            df = df.withColumn('label', col('label').cast('double'))
        # (3) Apply LLM to introduce new synthetic feature: "genre".
        ClassifierPipeline.logger.info('Generating synthetic genre features via LLM...')
        df = self._predict_genres_llm(df, train=train)
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

        # Create historical dataframe for reference if in training mode
        if train:
            self.historical_df = df.select("*")  # Deep copy to avoid issues
        
        # NEW STEPS: Add collaborative features
        ClassifierPipeline.logger.info('Adding collaborative features...')
        
        # 1. Director-Writer Collaboration Count
        # Apply LLM-based synthetic feature generation for collab_count
        ClassifierPipeline.logger.info('Generating synthetic collaboration count feature...')
        df = self._predict_collab_count(df, train=train)

        # 2. Director Success Rate
        # Apply synthetic feature generation for director success rate
        # ClassifierPipeline.logger.info('Generating synthetic director success rate feature...')
        # df = self._predict_director_success(df, train=train)

        # Add new features to the feature column list
        # self.feature_cols.add("director_success_rate")
        # self.feature_cols.add("director_film_count")

        # 3. Writer Success Rate
        # Apply synthetic feature generation for writer success rate
        # ClassifierPipeline.logger.info('Generating synthetic writer success rate feature...')
        # df = self._predict_writer_success(df, train=train)

        # Add new features to the feature column list
        # self.feature_cols.add("writer_success_rate")
        # self.feature_cols.add("writer_film_count")

        # 4. Director-Writer Team Rating
        # Apply synthetic feature generation for director-writer team success rate
        # ClassifierPipeline.logger.info('Generating synthetic team success rate feature...')
        # df = self._predict_team_success(df, train=train)

        # Add new feature to the feature column list
        # self.feature_cols.add("team_success_rate")

        # 5. Production Company Track Record (if available)
        # Apply synthetic feature generation for production company success rate
        # ClassifierPipeline.logger.info('Generating synthetic company success rate feature...')
        # df = self._predict_company_success(df, train=train)

        # Add new features to the feature column list
        # self.feature_cols.add("company_success_rate")
        # self.feature_cols.add("company_film_count")

        # 6. Repeat Collaboration Indicator
        # if 'collab_count' in df.columns:
            # ClassifierPipeline.logger.info('Creating repeat collaboration indicator...')
            # df = df.withColumn("is_repeat_collaboration", 
                            # when(col("collab_count") > 0, 1.0).otherwise(0.0))
            
            # self.feature_cols.add("is_repeat_collaboration")

        # 7. Director Experience Level
        # ClassifierPipeline.logger.info('Generating synthetic director experience level...')
        # df = self._predict_director_experience(df)
        # self.feature_cols.add("director_experience")

        # 8. Production Company Size (if available)
        # ClassifierPipeline.logger.info('Generating synthetic company size...')
        # df = self._predict_company_size(df)
        # self.feature_cols.add("company_size")

        # 9. Team Stability Index
        ClassifierPipeline.logger.info('Generating synthetic director-team stability...')
        df = self._predict_team_stability(df, train=train)
        self.feature_cols.add("director_team_variability")

        # 10. Genre Specialization (using the LLM-generated genre)
        # Apply synthetic features
        # ClassifierPipeline.logger.info('Generating synthetic genre specialization...')
        # df = self._predict_genre_specialization(df, train=train)
        # self.feature_cols.add("genre_success_rate")
        # self.feature_cols.add("genre_count")

        # ClassifierPipeline.logger.info('Generating synthetic lead actor success rate...')
        # df = self._predict_lead_actor_success(df, train=train)
        # self.feature_cols.add("lead_actor_success_rate")

        ClassifierPipeline.logger.info('Generating synthetic genre count by decade...')
        df = self._predict_genre_count_by_decade(df, train=train)
        self.feature_cols.add("genre_count_by_decade")

        # 11. RunTime Category
        # Apply synthetic feature: runtime category
        ClassifierPipeline.logger.info('Generating synthetic runtime category feature...')
        df = self._categorize_runtime(df, train=train)
        self.feature_cols.add("runtime_category_index")

        # 12. Revenue
        # Apply revenue feature
        # ClassifierPipeline.logger.info('Generating synthetic revenue feature...')
        # df = self._predict_revenue_tmdb(df, train=train)
        # self.feature_cols.add("revenue")

        # ORIGINAL CODE CONTINUES FROM HERE
        # (6) Handle categorical variables by applying StringIndexer.
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

        # (7) Handle missing values.
        df = df.fillna(0)
        # (8) Assemble all feature columns into a single vector.
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

    def run(
            self: 'ClassifierPipeline',
            train_csv_path: str,
            test_csv_path: str,
            output_txt_path: str
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
            output_txt_path : str
                The path to save the predictions.
        '''
        # (1) Load data: TRAIN and TEST.
        ClassifierPipeline.logger.info('***(1) LOADING DATA...***')
        self._load_data(train_csv_path=train_csv_path, test_csv_path=test_csv_path)
        train_df = self.data_dict['train_data']
        test_df = self.data_dict['test_data']
        ClassifierPipeline.logger.info('***(1) DATA: LOADED!***')
        # (2) Pre-process data: TRAIN and TEST.
        ClassifierPipeline.logger.info('***(2) PRE-PROCESSING DATA...***')
        ClassifierPipeline.logger.info('***PRE-PROCESS: TRAIN SET***')
        train_df = self._preprocess(train_df, train=True)
        ClassifierPipeline.logger.info('***PRE-PROCESS: TEST SET***')
        test_df = self._preprocess(test_df)
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

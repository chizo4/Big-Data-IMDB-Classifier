'''
--------------------------------------------------------------
FILE:
    movie_pipeline/classifier_model.py

INFO:
    Implementation of a Random Forest classifier for movie prediction.
    Handles model training, evaluation and prediction functionality.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


from logger import get_logger
import pyspark.ml as ml
from pyspark.ml.classification import RandomForestClassifier


class ClassifierModel:
    '''
    -------------------------
    ClassifierModel - Class for Random Forest model training and prediction.
    -------------------------
    '''

    # Class logger for Spark operations.
    logger = get_logger(__name__)

    def __init__(self: 'ClassifierModel', model_path: str, feature_cols: set=None) -> None:
        '''
        Initialize the ClassifierModel with Random Forest configuration.

            Parameters:
            -----------
            model_path : str
                Path where model will be saved/loaded from
            feature_cols : set
                Set of feature column names (will be updated by pipeline)
        '''
        self.model_path = model_path
        self.feature_cols = feature_cols if feature_cols else set()
        # Initialize Random Forest classifier with optimized parameters.
        self.classifier = RandomForestClassifier(
            featuresCol='scaled_features',
            labelCol='label',
            numTrees=200,
            maxDepth=16,
            minInstancesPerNode=5,
            impurity='gini',
            bootstrap=True,
            seed=42
        )

    def update_feature_cols(self: 'ClassifierModel', feature_cols: set) -> None:
        '''
        Update the feature columns used by the model.

            Parameters:
            -----------
            feature_cols : set
                Updated set of feature column names
        '''
        self.feature_cols = feature_cols
        ClassifierModel.logger.info(f'Updated feature columns: {self.feature_cols}')

    def train(self: 'ClassifierModel', train_df: 'DataFrame') -> None:
        '''
        Train the Random Forest classifier using the engineered features.

            Parameters:
            -----------
            train_df : DataFrame
                The preprocessed and feature-engineered DataFrame for training.
        '''
        # Run training.
        ClassifierModel.logger.info('Training model on TRAINING set...')
        spark_pipeline = ml.Pipeline(stages=[self.classifier])
        model = spark_pipeline.fit(train_df)
        # Save trained model.
        model.write().overwrite().save(self.model_path)
        ClassifierModel.logger.info(f'TRAINED model saved to: "{self.model_path}"')
        self._analyze_feature_importance(model)

    def _analyze_feature_importance(self: 'ClassifierModel', model: 'ml.Pipeline') -> None:
        '''
        Analyze and print feature importances from the model.

            Parameters:
            -----------
            model : ml.Pipeline
                Trained model pipeline
        '''
        if hasattr(model.stages[-1], 'featureImportances'):
            importances = model.stages[-1].featureImportances
            feature_names = list(self.feature_cols)
            feature_importance_pairs = [
                (feature_names[i], float(importances[i])) for i in range(min(len(feature_names), len(importances)))
            ]
            sorted_importances = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
            ClassifierModel.logger.info('Top 5 features by importance:')
            for i, (feature, importance) in enumerate(sorted_importances[:5]):
                ClassifierModel.logger.info(f"{i+1}. {feature:<20}: {importance:.6f}")

    def predict(self: 'ClassifierModel', test_df: 'DataFrame') -> 'DataFrame':
        '''
        Load the trained model and make predictions on test data.

            Parameters:
            -----------
            test_df : DataFrame
                The preprocessed and feature-engineered test DataFrame.

            Returns:
            -----------
            predictions_df : DataFrame
                DataFrame with predictions added.
        '''
        # Load the trained model.
        ClassifierModel.logger.info(f'Loading model from: "{self.model_path}"')
        try:
            loaded_model = ml.PipelineModel.load(self.model_path)
        except Exception as e:
            ClassifierModel.logger.error(f'Error loading model: {str(e)}')
            raise RuntimeError(f'Could not load model from "{self.model_path}"')
        # Make predictions on TEST data.
        ClassifierModel.logger.info('Running predictions on test data...')
        predictions_df = loaded_model.transform(test_df)
        return predictions_df

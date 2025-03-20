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
        self.model = None
        self.model_path = model_path
        self.feature_cols = feature_cols if feature_cols else set()
        # Initialize Random Forest classifier with optimized parameters.
        self.classifier = RandomForestClassifier(
            featuresCol='scaled_features',
            labelCol='label',
            numTrees=200,
            maxDepth=10,
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

    def _analyze_feature_importance(self: 'ClassifierModel') -> None:
        '''
        Analyze and print feature importances from the model.

            Parameters:
            -----------
            model : ml.Pipeline
                Trained model pipeline
        '''
        if hasattr(self.model.stages[-1], 'featureImportances'):
            importances = self.model.stages[-1].featureImportances
            feature_names = list(self.feature_cols)
            feature_importance_pairs = [
                (feature_names[i], float(importances[i])) for i in range(min(len(feature_names), len(importances)))
            ]
            sorted_importances = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
            ClassifierModel.logger.info('Top 5 features by importance:')
            for i, (feature, importance) in enumerate(sorted_importances[:5]):
                ClassifierModel.logger.info(f"{i+1}. {feature:<20}: {importance:.6f}")

    def train(self: 'ClassifierModel', train_df: 'DataFrame') -> None:
        '''
        Train the Random Forest classifier using the engineered features.

            Parameters:
            -----------
            train_df : DataFrame
                The preprocessed and feature-engineered DataFrame for training.
        '''
        # Train model.
        ClassifierModel.logger.info('Training model on TRAINING set...')
        spark_pipeline = ml.Pipeline(stages=[self.classifier])
        self.model = spark_pipeline.fit(train_df)
        # Save model.
        self.model.write().overwrite().save(self.model_path)
        ClassifierModel.logger.info(f'TRAINED model saved to: "{self.model_path}"')
        # Analyze feature impacts.
        self._analyze_feature_importance()

    def predict(self: 'ClassifierModel', test_df: 'DataFrame') -> 'DataFrame':
        '''
        Run model predictions.

            Parameters:
            -----------
            test_df : DataFrame
                The preprocessed and feature-engineered DataFrame for testing.

            Returns:
            -----------
            predictions_df : DataFrame
                DataFrame with predictions added.
        '''
        ClassifierModel.logger.info('Running predictions on TEST data...')
        predictions_df = self.model.transform(test_df)
        return predictions_df

'''
--------------------------------------------------------------
FILE:
    movie_pipeline/llm_predictor.py

INFO:
    A module implementing LLM-based genre prediction for movies.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import pandas as pd
import ollama
from pyspark.sql import DataFrame, SparkSession
from logger import get_logger


class LLMGenrePredictor:
    '''
    -------------------------
    LLMGenrePredictor - Class to predict movie genres using an LLM-based model.
                        Powered via the Ollama API.
    -------------------------
    '''

    # Pre-defined movie genres (for LLM prompt).
    MOVIE_GENRES = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
        'History', 'Horror', 'Music', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War'
    ]

    def __init__(
            self: 'LLMGenrePredictor',
            batch_size: int=20,
            model_name: str='gemma3:4b',
            spark: 'SparkSession'=None
        ) -> None:
        '''
        Initialize the LLMGenrePredictor class.

            Parameters:
            ----------
            batch_size : int
                Number of movies to process in each batch.
            model_name : str
                Name of the Ollama model to use for prediction.
            spark : SparkSession
                Spark session to use for creating DataFrames.
        '''
        self.batch_size = batch_size
        self.model_name = model_name
        self.spark = spark
        self.logger = get_logger(__name__)
        # Create an Ollama client.
        self.client = ollama.Client(host='http://localhost:11434')

    def predict_genres(self, df: 'DataFrame') -> 'DataFrame':
        '''
        Predict genres for a DataFrame of movies and return as a new DataFrame.

            Parameters:
            ----------
            df : DataFrame
                A DataFrame containing movie data.

            Returns:
            ----------
            DataFrame
                A DataFrame containing movie IDs and predicted genres.
        '''
        pdf = df.select(
            'tconst',
            'primaryTitle',
            'originalTitle',
            'startYear',
            'runtimeMinutes',
            'numVotes'
        ).toPandas()
        results = []
        total_batches = (len(pdf) + self.batch_size - 1) // self.batch_size
        # Process the movies in batches.
        for i in range(0, len(pdf), self.batch_size):
            batch = pdf.iloc[i:i+self.batch_size]
            self.logger.info(f'Processing batch {i//self.batch_size + 1}/{total_batches} ({len(batch)} movies)')
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        genre_df = pd.DataFrame(results)
        # Use the provided Spark session or extract it from the DataFrame
        if self.spark:
            spark_session = self.spark
        else:
            # Fallback to extracting from DataFrame
            spark_session = df._sc._jvm.org.apache.spark.sql.SparkSession.getActiveSession().get()
        return spark_session.createDataFrame(genre_df)

    def _process_batch(self: 'LLMGenrePredictor', batch: pd.DataFrame) -> list:
        '''
        Process a batch of movies through the LLM.

            Parameters:
            ----------
            batch : pandas.DataFrame
                A batch of movies to process.

            Returns:
            ----------
            list
                A list of dictionaries containing tconst and genre predictions.
        '''
        results = []
        for _, movie in batch.iterrows():
            prompt = self._create_prompt(movie)
            try:
                # Use the Ollama client to generate a response.
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=False
                )
                # Extract the response text.
                response_text = response['response']
                # Parse the JSON from the response.
                genre_data = self._parse_response(response_text, movie['tconst'])
                results.append(genre_data)
            except Exception as e:
                self.logger.error(f'Error processing movie {movie["tconst"]}: {str(e)}')
                # Fallback to a default genre.
                results.append({'tconst': movie['tconst'], 'genre': 'unknown'})
        return results

    def _create_prompt(self: 'LLMGenrePredictor', movie_data: dict) -> str:
        '''
        Create a structured prompt for genre prediction.

            Parameters:
            ----------
            movie_data : dict
                A dictionary containing movie information.

            Returns:
            ----------
            str
                A structured prompt for genre prediction.
        '''
        prompt = f'''You are an IMDB expert. Given this movie data:

        - Title: {movie_data['primaryTitle']}
        - Original Title: {movie_data['originalTitle']}
        - Year: {movie_data['startYear']}
        - Runtime: {movie_data['runtimeMinutes']} min
        - Votes: {movie_data['numVotes']}

        Predict ONE genre from this list:
        {", ".join(self.MOVIE_GENRES)}

        Respond ONLY with the genre name.'''
        return prompt

    def _parse_response(self: 'LLMGenrePredictor', response: str, tconst: str) -> dict:
        '''
        Parse the LLM response to extract the genre.

            Parameters:
            ----------
            response : str
                The response text from the LLM.
            tconst : str
                The movie ID to use as fallback if parsing fails.

            Returns:
            ----------
            dict
                A dictionary containing tconst and genre.
        '''
        try:
            # Strip whitespace and any quotes.
            genre = response.strip().strip('"\'')
            # Check if genre is in our predefined list (case-insensitive).
            for valid_genre in self.MOVIE_GENRES:
                if valid_genre.lower() == genre.lower():
                    return {'tconst': tconst, 'genre': valid_genre}
            # If the response doesn't match any valid genre.
            self.logger.warning(f'Unexpected genre "{genre}" for movie {tconst}')
            return {'tconst': tconst, 'genre': 'unknown'}
        except Exception as e:
            self.logger.error(f'Error parsing response for movie {tconst}: {str(e)}')
            return {'tconst': tconst, 'genre': 'unknown'}

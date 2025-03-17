'''
--------------------------------------------------------------
FILE:
    movie_pipeline/logger.py

INFO:
    Logger configuration for Spark debugs, since normal print()
    statements are not visible within Spark logs.

AUTHOR:
    @chizo4 (Filip J. Cierkosz)

VERSION:
    03/2025
--------------------------------------------------------------
'''


import logging


# Root logger configuration.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


def get_logger(name: str) -> logging.Logger:
    '''
    Get a logger instance for the given name.

        Parameters:
        -------------------------
        name : str
            The name of the logger.

        Returns:
        -------------------------
        logger : logging.Logger
            The logger instance for the given name.
    '''
    return logging.getLogger(name)

"""
This is a doc that contants constants that are used across python variables

This file should not need to be edited regularly

For configuration constants, edit either the project_settings.config or user_settings.config file
both of which this config file then imports
"""

import configparser
import functools
import logging
import os
from pathlib import Path
import time
import yaml

from sqlalchemy import create_engine, event

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_PATH =  BASE_DIR / 'data' / 'raw'
DATA_INTERIM_PATH = BASE_DIR / 'data' / 'interim'
DATA_PROCESSED_PATH = BASE_DIR / 'data' / 'processed'
LOGS_PATH = BASE_DIR / 'logs'

if os.name =='nt': # windows correct file path
    HOME_DIR = Path(os.environ['USERPROFILE'])
else: # linux correct file path
    HOME_DIR = Path('~')

# Imports both user and project config files and merges them
# Project configs overwrite user!!
config = configparser.ConfigParser()
config.read([HOME_DIR / 'user_settings.config', BASE_DIR / 'project_settings.config'])

S3_BUCKET_PATH = f"s3://{config['DEFAULT']['AWS_S3_BUCKET']}"
S3_PROJECT_PATH = f"{S3_BUCKET_PATH}/{config['DEFAULT']['AWS_S3_PROJ_PATH']}"

S3_DATA_PATH = f"{S3_BUCKET_PATH}/{config['DEFAULT']['AWS_S3_DATA_PATH']}"
S3_DATA_RAW_PATH = f"{S3_DATA_PATH}/raw"
S3_DATA_INTERIM_PATH = f"{S3_DATA_PATH}/interim"
S3_DATA_PROCESSED_PATH = f"{S3_DATA_PATH}/processed"

###########################################################
######### PROJECT SPECIFIC GLOBAL CONTENT HERE ############
###########################################################



###########################################################

# Database Import Functions
def sqlalchemy_conn_string(db_type='mssql',
                           uid=config['DEFAULT']['DB_USERNAME'],
                           pwd=config['DEFAULT']['DB_PASSWORD'],
                           server=config['DEFAULT']['DB_ENDPOINT'],
                           database=config['DEFAULT']['DB_DATABASE'],
                           port=config['DEFAULT']['DB_PORT'],
                           driver=config['DEFAULT']['DB_DRIVER']):
    if db_type == 'mssql':
        conn_str = f"mssql+pyodbc://{uid}:{pwd}@{server}:{port}/{database}?driver={driver}"
    else:
        conn_str = ""
        logging.warning(f"{db_type} not supported")
    logging.debug(f"createed conn_string of {conn_str}")
    return conn_str


def get_sqlalchemy_engine(conn_string, echo=False):
    engine = create_engine(conn_string, echo=echo)

    # This implements the execute many function when pd.to_sql is called to speed up exports
    @event.listens_for(engine, 'before_cursor_execute')
    def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
        if executemany:
            cursor.fast_executemany = True

    return engine


# This function does not need to be used but can be a reference as to how to push to the database
def df_to_db(df, table, engine, if_exists='fail', chunksize=100000, dtype=None):
    s = time.time()
    df.to_sql(table, engine, if_exists=if_exists, index=False, chunksize=chunksize, dtype=dtype)
    logging.info(f"{time.time() - s}s to run df.to_sql with a df length of {len(df)}")


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def setup_logging(default_path, default_level=logging.WARNING):
    """Setup logging configuration

    """
    path = default_path
    if path.exists():
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning('logging.yaml not imported')



def main_wrapper(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            setup_logging(BASE_DIR / 'logging_config.yaml', logging.DEBUG)  # todo fix logging
            logging.info("__main__ start")

            value = func(*args, **kwargs)

            logging.info("__main__ end successfully")
        except Exception as e:
            logging.error("__main__ end with error")
            raise
        finally:
            end_time = time.perf_counter()  # 2
            run_time = end_time - start_time  # 3
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer
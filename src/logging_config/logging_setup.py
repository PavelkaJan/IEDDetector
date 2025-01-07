import logging.config

def setup_logging():
    logging.config.fileConfig('src/logging_config/logging.ini')

import logging as lg


def get_logger(logger_name=''):
    """Get a default logger that includes a timestamp."""
    logger = lg.getLogger(logger_name)
    logger.handlers = []
    ch = lg.StreamHandler()
    str_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = lg.Formatter(str_fmt, datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger

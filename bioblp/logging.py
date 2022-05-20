import logging.config


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False
        },
        'epilp': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)


def get_logger(name: str):
    return logging.getLogger(name)
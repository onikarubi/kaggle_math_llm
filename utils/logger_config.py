import logging

def setup_logger(
    module_name: str,
    handler_class,
    log_level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    *handler_args,
    **handler_kwargs,
):
    logger = logging.getLogger(module_name)
    handler: logging.Handler = handler_class(*handler_args, **handler_kwargs)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger

# info_stream_logger = setup_logger(
#     module_name='info_stream_logger',
#     handler_class=logging.StreamHandler,
#     log_level=logging.INFO
# )

# debug_stream_logger = setup_logger(
#     module_name='debug_stream_logger',
#     handler_class=logging.StreamHandler,
#     log_level=logging.DEBUG
# )

# danger_stream_logger = setup_logger(
#     module_name='danger_stream_logger',
#     handler_class=logging.StreamHandler,
#     log_level=logging.WARNING
# )


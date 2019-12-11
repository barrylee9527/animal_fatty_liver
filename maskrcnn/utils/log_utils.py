# @Time    : 2018.10.09
# @Author  : kawa Yeung
# @Licence : bio-totem


import os
import yaml
import logging
from logging import config


def init_logger(log_config='log.yaml', log_dir=None):
    """
    initialize logging config setting
    :param log_config: logging config by file, see the `log.yaml` file
    :param log_dir: log dir to save logging, absolute path
    :return:
    """

    current_path = os.path.dirname(__file__)
    config_file = os.path.join(current_path, "..", log_config)
    with open(config_file, 'r', encoding='utf-8') as f:
        conf = yaml.load(f)
        if log_dir is None:
            # default path in the project root dir to save the logging
            log_dir = os.path.join(current_path, "..", "logs")
        if not os.path.isdir(log_dir): os.makedirs(log_dir)
        conf["handlers"]["file"]["filename"] = os.path.join(log_dir, conf["handlers"]["file"]["filename"])
        conf["handlers"]["error"]["filename"] = os.path.join(log_dir, conf["handlers"]["error"]["filename"])
        config.dictConfig(conf)


def get_logger(log_dir=None, level='info'):
    """
    get logger
    :param level: loggers level with "info", "error", mapping to the log.yaml file
    :return: logger
    """

    assert level in ["info", "error"], "wrong logger level setting, must be 'info' or 'error' "

    init_logger(log_dir=log_dir)

    return logging.getLogger(level)

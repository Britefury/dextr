import os
import sys
from configparser import RawConfigParser



_CONFIG_PATH = './dextr.cfg'

_config__ = None

def get_config():  # pragma: no cover
    global _config__
    if _config__ is None:
        if os.path.exists(_CONFIG_PATH):
            try:
                _config__ = RawConfigParser()
                _config__.read(_CONFIG_PATH)
            except Exception as e:
                print('WARNING: error {} trying to open config '
                      'file from {}'.format(e, _CONFIG_PATH))
                _config__ = RawConfigParser()
        else:
            _config__ = RawConfigParser()
    return _config__


def get_data_path(config_name, exists=True):  # pragma: no cover
    dir_path = get_config().get('paths', config_name)
    if exists:
        if not os.path.exists(dir_path):
            raise RuntimeError(
                'semantic_segmentation.settings: the directory path {} does not exist'.format(dir_path))
    return dir_path

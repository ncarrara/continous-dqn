import os

import logging

logger = logging.getLogger(__name__)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        logger.warning("Can't create \"{}\", folder exists".format(path))
import torch
import utils
import logging
import numpy as np

# logging.basicConfig(level=logging.ERROR)
logging.basicConfig(level=logging.INFO)

np.set_printoptions(precision=2)

logger = logging.getLogger(__name__)

logger.info("Pytorch version : {}".format(torch.__version__))
if str(torch.__version__) == "0.4.1.":
    logger.warn("0.4.1. is bugged regarding mse loss")

utils.set_device()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

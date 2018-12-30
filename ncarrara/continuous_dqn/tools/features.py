import numpy as np
import pprint

from ncarrara.bftq_pydial.tools.features import feature_0


def identity(s,e):
    return s.tolist()

def str_to_feature(feature_str):
    if feature_str == "identity":
        return identity
    elif feature_str=="feature_0":
        return feature_0
    else:
        raise Exception("Unknown feature : {}".format(feature_str))

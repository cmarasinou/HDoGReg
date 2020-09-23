import torch
import numpy as np
import argparse


def seed_everything(seed=0):
    """Fixes the random seeds for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def arg_parsing(config_dict):
    """ For parsing command-line arguments defined with a dictionary
    Args:
        config_dict (dict): keys are command-line arguments, values are the default values
    Returns:
        config_dict: updated with command-line argument values
    """
    arg_parser = argparse.ArgumentParser()
    for key in config_dict.keys():
        arg_parser.add_argument('-{}'.format(key))
    args = vars(arg_parser.parse_args())
    for key in config_dict.keys():
        if args[key] is not None:
            config_dict[key] = args[key]
    return config_dict
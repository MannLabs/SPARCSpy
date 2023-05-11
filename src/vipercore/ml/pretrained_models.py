"""
Collection of functions to load pretrained models to use in the SPARCSpy environment.
"""

from vipercore.ml.plmodels import MultilabelSupervisedModel
import os

def _load_multilabelSupervised(checkpoint_path, hparam_path, type, eval = True):
    """
    Load a pretrained model uploaded to the github repository.

    Parameters
    ----------
    checkpoint_path : str
        The path of the checkpoint file to load the pretrained model from.
    hparam_path : str
        The path of the hparams file containing the hyperparameters used in training the model.
    type : str
        The type of the model, e.g., 'VGG1' or 'VGG2'.
    eval : bool, optional
        If True then the model will be returned in eval mode. Default is set to True.

    Returns
    -------
    model : MultilabelSupervisedModel
        The pretrained multilabel classification model loaded from the checkpoint, and moved to the appropriate device.

    Examples
    --------
    >>> model = _load_multilabelSupervised(“path/to/checkpoint.ckpt”, “path/to/hparams.yaml”, “resnet50”)
    >>> print(model)
    MultilabelSupervisedModel(…)

    """

    #load model
    model = MultilabelSupervisedModel.load_from_checkpoint(checkpoint_path, hparams_file=hparam_path, type = type)
    model.eval();

    return(model)

def _get_data_dir():
    """
    Helper Function to get path to data that was packaged with SPARCSpy

    Returns
    -------
        str: path to data directory
    """
    src_code_dir, _ = os.path.split(__file__)
    data_dir = src_code_dir.replace("vipercore/ml", "pretrained_models/")
    return (data_dir)

def autophagy_classifier1():
    """
    Load binary autophagy classification model published as Model 1 in original SPARCSpy publication.
    """

    data_dir = _get_data_dir()

    checkpoint_path = os.path.join(data_dir, "autophagy/autophagy1/VGG1_autophagy_classifier1.cpkt")
    hparam_path = os.path.join(data_dir, "autophagy/autophagy1/hparams.yaml")

    model = _load_multilabelSupervised(checkpoint_path, hparam_path, type = "VGG1_old")
    return(model)

def autophagy_classifier2_1():
    """
    Load binary autophagy classification model published as Model 2.1 in original SPARCSpy publication.
    """

    data_dir = _get_data_dir()

    checkpoint_path = os.path.join(data_dir, "autophagy/autophagy2.1/VGG2_autophagy_classifier2.1.cpkt")
    hparam_path = os.path.join(data_dir, "autophagy/autophagy2.1/hparams.yaml")

    model = _load_multilabelSupervised(checkpoint_path, hparam_path, type = "VGG2_old")
    return(model)

def autophagy_classifier_2_2():
    """
    Load binary autophagy classification model published as Model 2.2 in original SPARCSpy publication.
    """

    data_dir = _get_data_dir()

    checkpoint_path = os.path.join(data_dir, "autophagy/autophagy2.2/VGG2_autophagy_classifier2.2.cpkt")
    hparam_path = os.path.join(data_dir, "autophagy/autophagy2.2/hparams.yaml")

    model = _load_multilabelSupervised(checkpoint_path, hparam_path, type = "VGG2_old")
    return(model)
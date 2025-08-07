"""
Core modules for coinformer experiments.
"""
from .models import ModelConfig, create_coinformer_model, DEFAULT_CONFIG
from .config import ExperimentConfig
from .training import train_coinformer_model, save_model_with_config, load_model_from_config
from .samplers import generate_data, generate_data_with_p, generate_data_with_p_list
from . import plotting

__all__ = [
    'ModelConfig',
    'ExperimentConfig', 
    'create_coinformer_model',
    'train_coinformer_model',
    'save_model_with_config',
    'load_model_from_config',
    'generate_data',
    'generate_data_with_p',
    'generate_data_with_p_list',
    'plotting',
    'DEFAULT_CONFIG'
]
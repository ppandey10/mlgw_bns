from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import IO, ClassVar, Optional, Type, Union

import h5py
import joblib  # type: ignore
import numpy as np
import pkg_resources
import yaml
from dacite import from_dict
from numba import njit  # type: ignore

from .data_management import (
    DownsamplingIndices,
    FDWaveforms,
    ParameterRanges,
    PrincipalComponentData,
    Residuals,
    SavableData,
)
from .dataset_generation import (
    BarePostNewtonianGenerator,
    Dataset,
    ParameterGenerator,
    ParameterSet,
    TEOBResumSGenerator,
    WaveformGenerator,
    WaveformParameters,
)
from .higher_order_modes import (
    BarePostNewtonianModeGenerator,
    Mode,
    ModeGenerator,
    ModeGeneratorFactory,
    spherical_harmonic_spin_2,
    spherical_harmonic_spin_2_conjugate, 
    teob_mode_generator_factory,
)
from .downsampling_interpolation import DownsamplingTraining, GreedyDownsamplingTraining
from .neural_network import Hyperparameters, NeuralNetwork, SklearnNetwork
from .principal_component_analysis import (
    PrincipalComponentAnalysisModel,
    PrincipalComponentTraining,
)
from .model import Model

from .taylorf2 import SUN_MASS_SECONDS, smoothing_func

PRETRAINED_MODEL_FOLDER = "data/"
MODELS_AVAILABLE = ["default", "fast"]

PRETRAINED_MODES_MODEL_FOLDER = "data/HOM/"
MODES_MODELS_AVAILABLE = ["pp_small_default_l2_m2", "pp_large_default_l2_m2"] 

DEFAULT_DATASET_BASENAME = "data/default"

class ModesModel:
    def __init__(
        self, 
        modes: list[Mode], 
        generator_factory: ModeGeneratorFactory = teob_mode_generator_factory,
        **model_kwargs
        ):
        
        self.modes = modes

        self._base_filename = model_kwargs.pop('filename', '')

        self.models = {}

        for mode in modes:

            self.models[mode] = Model(
                filename = self.mode_filename(mode),
                waveform_generator = generator_factory(mode),
                mode = mode,
                **model_kwargs
                )

    
    def mode_filename(self, mode: Mode) -> str:
        return f'{self.base_filename}_l{mode[0]}_m{mode[1]}'
    

    @property
    def base_filename(self) -> str:
        return self._base_filename


    @base_filename.setter
    def base_filename(self, value: str):
        self._base_filename = value
        for mode, model in self.models.items():
            model.filename = self.mode_filename(mode)
    

    def generate(self, *args, **kwargs) -> None:
        for model in self.models.values():
            model.generate(*args, **kwargs)


    def set_hyper_and_train_nn(self, *args, **kwargs) -> None:
        for model in self.models.values():
            model.set_hyper_and_train_nn(*args, **kwargs)


    def save(self, *args, **kwargs) -> None:
        for model in self.models.values():
            model.save(*args, **kwargs)

 
    def load(self, *args, **kwargs) -> None:
        for model in self.models.values():
            model.load(*args, **kwargs)
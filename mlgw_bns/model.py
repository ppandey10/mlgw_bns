import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import IO, ClassVar, Optional, Union

import h5py
import joblib  # type: ignore
import numpy as np
import optuna
import pkg_resources
from numba import njit  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from .data_management import (
    DownsamplingIndices,
    FDWaveforms,
    PrincipalComponentData,
    Residuals,
    SavableData,
)
from .dataset_generation import (
    BarePostNewtonianGenerator,
    Dataset,
    ParameterSet,
    TEOBResumSGenerator,
    WaveformGenerator,
    WaveformParameters,
)
from .downsampling_interpolation import DownsamplingTraining, GreedyDownsamplingTraining
from .principal_component_analysis import (
    PrincipalComponentAnalysisModel,
    PrincipalComponentTraining,
)
from .taylorf2 import SUN_MASS_SECONDS

TRIALS_FILE = "data/best_trials.pkl"
DEFAULT_DATASET_BASENAME = "data/default_dataset"


@dataclass
class Hyperparameters:
    r"""Dataclass containing the parameters which are passed to
    the neural network for training, as well as a few more.

    Parameters
    ----------
    pc_exponent: float
            Exponent to be used in the normalization of the
            principal components: the network learns to reconstruct
            :math:`x_i \lambda_i^\alpha`, where
            :math:`x_i` are the principal-component
            representation of a waveform, while
            :math:`\lambda_i` are the eigenvalues of the PCA
            and finally :math:`\alpha` is this parameter.
    n_train: float
            Number of waveforms to use in the training.
    hidden_layer_sizes: tuple[int, ...]
            Sizes of the layers in the neural network.
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    activation: str
            Activation function.
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    alpha: float
            Regularization parameter.
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    batch_size: int
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    learning_rate_init: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    tol: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    validation_fraction: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    n_iter_no_change: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    """

    # controls how much weight is give to higher principal components
    pc_exponent: float

    # number of training data points to use
    n_train: int

    # parameters for the sklearn neural network
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    alpha: float
    batch_size: int
    learning_rate_init: float
    tol: float
    validation_fraction: float
    n_iter_no_change: float

    max_iter: int = field(default=1000)

    @property
    def nn_params(self) -> dict[str, Union[int, float, str, bool, tuple[int, ...]]]:
        """Return a dictionary which can be readily unpacked
        and used to initialize a :class:`MLPRegressor`.
        """

        return {
            "max_iter": self.max_iter,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate_init": self.learning_rate_init,
            "tol": self.tol,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "early_stopping": True,
            "shuffle": True,
        }

    @classmethod
    def from_trial(cls, trial: optuna.Trial, n_train_max: int):
        """Generate the hyperparameter set starting from an
        :class:`optuna.Trial`.

        Parameters
        ----------
        trial : optuna.Trial
                Used to generate the parameters.
        n_train_max : int
                Upper bound for the attribute :attr:`n_train`.
        """

        n_layers = trial.suggest_int("n_layers", 2, 4)

        layers = tuple(
            trial.suggest_int(f"size_layer_{i}", 10, 100) for i in range(n_layers)
        )

        return cls(
            hidden_layer_sizes=layers,
            activation=str(
                trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
            ),
            alpha=trial.suggest_loguniform("alpha", 1e-6, 1e-1),  # Default: 1e-4
            batch_size=trial.suggest_int("batch_size", 100, 200),  # Default: 200
            learning_rate_init=trial.suggest_loguniform(
                "learning_rate_init", 2e-4, 5e-2
            ),  # Default: 1e-3
            tol=trial.suggest_loguniform("tol", 1e-15, 1e-7),  # default: 1e-4
            validation_fraction=trial.suggest_uniform("validation_fraction", 0.05, 0.2),
            n_iter_no_change=trial.suggest_int(
                "n_iter_no_change", 40, 100, log=True
            ),  # default: 10
            pc_exponent=trial.suggest_loguniform("pc_exponent", 1e-3, 1),
            n_train=trial.suggest_int("n_train", 50, n_train_max),
        )

    @classmethod
    def from_frozen_trial(cls, frozen_trial: optuna.trial.FrozenTrial):

        params = frozen_trial.params
        n_layers = params.pop("n_layers")

        layers = [params.pop(f"size_layer_{i}") for i in range(n_layers)]
        params["hidden_layer_sizes"] = tuple(layers)

        return cls(**params)

    @classmethod
    def default(cls, training_waveform_number: Optional[int] = None):

        try:
            if training_waveform_number is not None:
                best_trials = retrieve_best_trials_list()
                return best_trial_under_n(best_trials, training_waveform_number)
        except (FileNotFoundError, IndexError):
            pass

        return cls(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            alpha=1e-4,
            batch_size=200,
            learning_rate_init=1e-3,
            tol=1e-9,
            n_iter_no_change=50,
            validation_fraction=0.1,
            pc_exponent=0.2,
            n_train=200,
        )


@dataclass
class ParametersWithExtrinsic:
    r"""Parameters for the generation of a single waveform,
    including extrinsic parameters.

    Parameters
    ----------
    mass_ratio : float
            Mass ratio of the system, :math:`q = m_1 / m_2`,
            where :math:`m_1 \geq m_2`, so :math:`q \geq 1`.
    lambda_1 : float
            Tidal polarizability of the larger star.
            In papers it is typically denoted as :math:`\Lambda_1`;
            for a definition see for example section D of
            `this paper <http://arxiv.org/abs/1805.11579>`_.
    lambda_2 : float
            Tidal polarizability of the smaller star.
    chi_1 : float
            Aligned dimensionless spin component of the larger star.
            The dimensionless spin is defined as
            :math:`\chi_i = S_i / m_i^2` in
            :math:`c = G = 1` natural units, where
            :math:`S_i` is the :math:`z` component
            of the dimensionful spin vector.
            The :math:`z` axis is defined as the one which is
            parallel to the orbital angular momentum of the binary.
    chi_2 : float
            Aligned spin component of the smaller star.
    distance_mpc : float
            Distance to the binary system, in Megaparsecs.
    inclination : float
            Inclination --- angle between the binary system's
            angular momentum and the observation direction, in radians.
    total_mass : float
            Total mass of the binary system, in solar masses.
    reference_phase : float
            This will be set as the phase of the first point of the waveform.
            Defaults to 0.
    time_shift : float
            The waveform will be shifted in the time domain
            by this amount (measured in seconds).
            In the frequency domain, this means adding a linear
            term to the phase.
            Defaults to 0, which by convention means a configuration
            in which the merger happens at the right edge of the
            timeseries. This also means that, in the frequency domain,
            the phase at high frequencies is roughly constant.

    """

    mass_ratio: float
    lambda_1: float
    lambda_2: float
    chi_1: float
    chi_2: float
    distance_mpc: float
    inclination: float
    total_mass: float
    reference_phase: float = 0.0
    time_shift: float = 0.0

    def intrinsic(self, dataset: Dataset) -> WaveformParameters:
        return WaveformParameters(
            mass_ratio=self.mass_ratio,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            chi_1=self.chi_1,
            chi_2=self.chi_2,
            dataset=dataset,
        )

    @property
    def mass_sum_seconds(self) -> float:
        return self.total_mass * SUN_MASS_SECONDS

    def teobresums_dict(self, dataset: Dataset) -> dict[str, Union[float, int]]:
        """Parameter dictionary in a format compatible with
        TEOBResumS.

        The parameters are all converted to natural units.
        """
        base_dict = self.intrinsic(dataset).teobresums
        frequency_rescaling = dataset.total_mass / self.total_mass
        base_dict["initial_frequency"] *= frequency_rescaling
        base_dict["srate_interp"] *= frequency_rescaling
        base_dict["df"] *= frequency_rescaling

        return {
            **base_dict,
            **{
                "M": self.total_mass,
                "distance": self.distance_mpc,
                "inclination": self.inclination,
            },
        }
        # TODO figure out if it is possible to also pass
        # the phase and the time shift to TEOB.


class Model:
    """``mlgw_bns`` model.
    This class incorporates all the functionality required to
    compute the downsampling indices, train a PCA model,
    train a neural network and predict new waveforms.


    Parameters
    ----------
    filename : str
            Name for the model. Saved data will be saved under this name.
    initial_frequency_hz : float, optional
            Initial frequency for the waveforms, by default 20.0
    srate_hz : float, optional
            Time-domain signal rate for the waveforms,
            which is twice the maximum frequency of
            their frequency-domain version.
            By default 4096.0
    pca_components_number : int, optional
            Number of PCA components to use when reducing
            the dimensionality of the dataset.
            By default 30, which is high enough to reach extremely good
            reconstruction accuracy (mismatches smaller than :math:`10^{-8}`).
    waveform_generator : WaveformGenerator, optional
            Generator for the waveforms to be used in the training;
            by default None, in which case the system attempts to import
            the Python wrapper for TEOBResumS, failing which a :class:`BareBarePostNewtonianGenerator`
            is used, which is unable to generate effective-one-body waveforms.
    downsampling_training : DownsamplingTraining, optional
            Training algorithm for the downsampling;
            by default None, which means the greedy algorithm
            implemented in :class:`GreedyDownsamplingTraining` is used.
    """

    def __init__(
        self,
        filename: str,
        initial_frequency_hz: float = 20.0,
        srate_hz: float = 4096.0,
        pca_components_number: int = 30,
        waveform_generator: Optional[WaveformGenerator] = None,
        downsampling_training: Optional[DownsamplingTraining] = None,
    ):

        self.filename = filename

        if waveform_generator is None:
            try:
                from EOBRun_module import EOBRunPy  # type: ignore

                self.waveform_generator: WaveformGenerator = TEOBResumSGenerator(
                    EOBRunPy
                )
            except ModuleNotFoundError:
                self.waveform_generator = BarePostNewtonianGenerator()
        else:
            self.waveform_generator = waveform_generator

        self.dataset = Dataset(
            initial_frequency_hz, srate_hz, waveform_generator=self.waveform_generator
        )

        if downsampling_training is None:
            self.downsampling_training: DownsamplingTraining = (
                GreedyDownsamplingTraining(self.dataset)
            )
        else:
            self.downsampling_training = downsampling_training

        self.pca_components_number = pca_components_number

        self.nn: Optional[MLPRegressor] = None
        self.hyper: Hyperparameters = Hyperparameters.default()

        self.training_dataset: Optional[Residuals] = None
        self.training_parameters: Optional[ParameterSet] = None

        self.pca_data: Optional[PrincipalComponentData] = None
        self.downsampling_indices: Optional[DownsamplingIndices] = None

    @classmethod
    def default(cls, filename: str):
        model = cls(DEFAULT_DATASET_BASENAME)

        stream_arrays = pkg_resources.resource_stream(__name__, model.filename_arrays)
        stream_hyper = pkg_resources.resource_stream(__name__, model.filename_hyper)
        stream_nn = pkg_resources.resource_stream(__name__, model.filename_nn)

        model.load(streams=(stream_arrays, stream_hyper, stream_nn))

        model.filename = filename

        return model

    @property
    def auxiliary_data_available(self) -> bool:
        return self.pca_data is not None and self.downsampling_indices is not None

    @property
    def nn_available(self) -> bool:
        return self.nn is not None and self.auxiliary_data_available

    @property
    def training_dataset_available(self) -> bool:
        return (
            self.training_dataset is not None and self.training_parameters is not None
        )

    @property
    def filename_arrays(self) -> str:
        return f"{self.filename}_arrays.h5"

    @property
    def file_arrays(self) -> h5py.File:
        """File object in which to save datasets.

        Returns
        -------
        file : h5py.File
            To be used as a context manager.
        """
        return h5py.File(self.filename_arrays, mode="a")

    @property
    def filename_nn(self) -> str:
        """File name in which to save the neural network."""
        return f"{self.filename}_nn.pkl"

    @property
    def filename_hyper(self) -> str:
        """File name in which to save the hyperparameters."""
        return f"{self.filename}_hyper.pkl"

    def generate(
        self,
        training_downsampling_dataset_size: Optional[int] = 64,
        training_pca_dataset_size: Optional[int] = 256,
        training_nn_dataset_size: Optional[int] = 256,
    ) -> None:
        """Generate a new model from scratch.

        The parameters are the sizes of the three datasets to be used when training,
        if they are set to None they are not computed and the pre-existing values are used instead.

        Raises
        ------
        AssertionError
                If one of the parameters is set to None but no
                pre-existing data is availabele for it.


        Parameters
        ----------
        training_downsampling_dataset_size : int, optional
                By default 64.
        training_pca_dataset_size : int, optional
                By default 256.
        training_nn_dataset_size : int, optional
                By default 256.

        """

        if training_downsampling_dataset_size is not None:
            self.downsampling_indices = self.downsampling_training.train(
                training_downsampling_dataset_size
            )
        else:
            assert self.downsampling_indices is not None

        if training_pca_dataset_size is not None:
            self.pca_training = PrincipalComponentTraining(
                self.dataset, self.downsampling_indices, self.pca_components_number
            )

            self.pca_data = self.pca_training.train(training_pca_dataset_size)
        else:
            assert self.pca_data is not None

        if training_nn_dataset_size is not None:
            _, parameters, residuals = self.dataset.generate_residuals(
                training_nn_dataset_size, self.downsampling_indices
            )

            self.training_dataset = residuals
            self.training_parameters = parameters
        else:
            assert self.training_dataset is not None
            assert self.training_parameters is not None

        self.train_parameter_scaler()

    def save_arrays(self, include_training_data: bool = True) -> None:
        """Save all big arrays contained in this object to the file
        defined as ``{filename}.h5``.
        """

        assert self.pca_data is not None
        assert self.downsampling_indices is not None
        assert self.training_parameters is not None

        arr_list: list[SavableData] = [
            self.downsampling_indices,
            self.pca_data,
            self.training_parameters,
        ]

        if include_training_data:
            assert self.training_dataset is not None

            arr_list += [
                self.training_dataset,
            ]

        for arr in arr_list:
            arr.save_to_file(self.file_arrays)

    def save(self, include_training_data: bool = True) -> None:
        self.save_arrays(include_training_data)
        joblib.dump(self.hyper, self.filename_hyper)
        if self.nn is not None:
            joblib.dump(self.nn, self.filename_nn)

    def load(
        self, streams: Optional[tuple[IO[bytes], IO[bytes], IO[bytes]]] = None
    ) -> None:
        """Load model from the files present in the current folder.

        Parameters
        ----------
        streams: tuple[IO[bytes], IO[bytes], IO[bytes]], optional
                For internal use (specifically, loading the default model).
                Defaults to None (look in the current folder).
        """

        if streams is not None:
            filename_arrays: Union[IO[bytes], str]
            filename_hyper: Union[IO[bytes], str]
            filename_nn: Union[IO[bytes], str]

            filename_arrays, filename_hyper, filename_nn = streams
            file_arrays = h5py.File(filename_arrays, mode="r")
            ignore_warnings = True
        else:
            file_arrays = self.file_arrays
            filename_hyper = self.filename_hyper
            filename_nn = self.filename_nn
            ignore_warnings = False

        self.downsampling_indices = DownsamplingIndices.from_file(file_arrays)
        self.pca_data = PrincipalComponentData.from_file(file_arrays)
        self.training_parameters = ParameterSet.from_file(file_arrays)

        if (
            self.downsampling_indices is None
            or self.pca_data is None
            or self.training_parameters is None
        ):
            raise FileNotFoundError

        self.training_dataset = Residuals.from_file(
            file_arrays, ignore_warnings=ignore_warnings
        )

        try:
            self.hyper = joblib.load(filename_hyper)
            self.nn = joblib.load(filename_nn)
        except FileNotFoundError:
            logging.warn("No trained network or hyperparmeters found.")

        self.train_parameter_scaler()

    def train_parameter_scaler(self) -> None:
        """Train the parameter scaler, which takes the
        waveform parameters and makes their magnitudes similar
        in order to aid the training of the network.

        The parameter scaler is always trained on all the available dataset.
        """
        assert self.training_parameters is not None

        self.param_scaler: StandardScaler = StandardScaler().fit(
            self.training_parameters.parameter_array
        )

    @property
    def reduced_residuals(self) -> np.ndarray:
        """Reduced-dimensionality residuals
        --- in other words, PCA components ---
        corresponding to the :attr:`training_dataset`.

        This attribute is cached.
        """

        assert self.training_dataset is not None

        return self._reduced_residuals(self.training_dataset)

    @lru_cache(maxsize=1)
    def _reduced_residuals(self, residuals: Residuals):

        assert self.pca_data is not None

        return self.pca_model.reduce_data(residuals.combined, self.pca_data)

    @property
    def pca_model(self) -> PrincipalComponentAnalysisModel:
        """PCA model to be used for dimensionality reduction.

        Returns
        -------
        PrincipalComponentAnalysisModel
        """
        return PrincipalComponentAnalysisModel(self.pca_components_number)

    def train_nn(
        self, hyper: Hyperparameters, indices: Union[list[int], slice] = slice(None)
    ) -> MLPRegressor:
        """Train a

        Parameters
        ----------
        hyper : Hyperparameters
            Hyperparameters to be used in the initialization
            of the network.
        indices : Union[list[int], slice], optional
            Indices used to perform a selection of a subsection
            of the training data; by default ``slice(None)``
            which means all available training data is used.

        Returns
        -------
        MLPRegressor
            Trained network.
        """
        assert self.training_parameters is not None
        assert self.pca_data is not None

        scaled_params: np.ndarray = self.param_scaler.transform(
            self.training_parameters.parameter_array[indices]
        )

        training_residuals = (
            self.reduced_residuals
            * (self.pca_data.eigenvalues ** hyper.pc_exponent)[np.newaxis, :]
        )

        return MLPRegressor(**hyper.nn_params).fit(
            scaled_params, training_residuals[indices]
        )

    def set_hyper_and_train_nn(self, hyper: Optional[Hyperparameters] = None) -> None:
        """Train the network according to the hyperparameters given,
        and set it as a class attribute

        Parameters
        ----------
        hyper : Hyperparameters, optional
            Hyperparameters to use when training the network, by default None.
            If not given, the default is to fall back to the standard set of hyperparameters
            provided with the module.
        """

        if hyper is None:
            assert self.training_dataset is not None
            hyper = Hyperparameters.default(len(self.training_dataset))

        # increase the number of maximum iterations by a lot:
        # here we do not want to stop the training early.
        hyper.max_iter *= 10

        self.nn = self.train_nn(hyper)
        self.hyper = hyper

    def predict_residuals_bulk(
        self, params: ParameterSet, nn: MLPRegressor, hyper: Hyperparameters
    ) -> Residuals:
        """Make a prediction for a set of different parameters,
        using a network provided as a parameter.

        Parameters
        ----------
        params : ParameterSet
            Parameters of the residuals to reconstruct.
        nn : MLPRegressor
            Neural network to use for the reconstruction
        hyper : Hyperparameters
            Used just for the :attr:`Hyperparameters.pc_exponent` attribute.

        Returns
        -------
        Residuals
            Prediction through the model plus PCA.
        """

        assert self.pca_data is not None
        assert self.downsampling_indices is not None

        scaled_params = self.param_scaler.transform(params.parameter_array)

        scaled_pca_components = nn.predict(scaled_params)

        combined_residuals = self.pca_model.reconstruct_data(
            scaled_pca_components / (self.pca_data.eigenvalues ** hyper.pc_exponent),
            self.pca_data,
        )

        return Residuals.from_combined_residuals(
            combined_residuals, self.downsampling_indices.numbers_of_points
        )

    def predict_waveforms_bulk(
        self,
        params: ParameterSet,
        nn: Optional[MLPRegressor] = None,
        hyper: Optional[Hyperparameters] = None,
    ) -> FDWaveforms:

        if nn is None:
            nn = self.nn

        if hyper is None:
            hyper = self.hyper

        residuals = self.predict_residuals_bulk(params, nn, hyper)

        return self.dataset.recompose_residuals(
            residuals, params, self.downsampling_indices
        )

    def predict(self, frequencies: np.ndarray, params: ParametersWithExtrinsic):
        """Calculate the waveforms in the plus and cross polarizations,
        accounting for extrinsic parameters

        Parameters
        ----------
        frequencies : np.ndarray
                Frequencies where to compute the waveform, in Hz.
        params : ParametersWithExtrinsic
                Parameters for the waveform, both intrinsic and extrinsic.

        Returns
        -------
        hp, hc (complex np.ndarray)
                Cartesian plus and cross-polarized waveforms, computed
                at the given frequencies, measured in 1/Hz.

        """
        assert self.downsampling_indices is not None

        intrinsic_params = params.intrinsic(self.dataset)
        waveforms = self.predict_waveforms_bulk(
            ParameterSet.from_list_of_waveform_parameters([intrinsic_params])
        )

        phase_shift = (
            params.reference_phase
            + (2 * np.pi * params.time_shift)
            * self.dataset.frequencies_hz[self.downsampling_indices.phase_indices]
        )

        waveforms.phases[0] += phase_shift

        cartesian_waveforms = cartesian_waveforms_at_frequencies(
            waveforms,
            frequencies * params.mass_sum_seconds,
            self.dataset,
            self.downsampling_training,
            self.downsampling_indices,
        )

        pre = self.dataset.mlgw_bns_prefactor(intrinsic_params.eta, params.total_mass)
        cosi = np.cos(params.inclination)
        pre_plus = (1 + cosi ** 2) / 2 * pre / params.distance_mpc
        pre_cross = cosi * pre * (-1j) / params.distance_mpc

        return compute_polarizations(cartesian_waveforms[0], pre_plus, pre_cross)


@njit
def compute_polarizations(
    waveform: np.ndarray,
    pre_plus: Union[complex, float],
    pre_cross: Union[complex, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the two polarizations of the waveform,
    assuming they are the same but for a differerent prefactor
    (which is the case for compact binary coalescences).

    This function is separated out so that it can be decorated with
    `numba.njit <https://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html>`_
    which allows it to be compiled --- this can speed up the computation somewhat.

    Parameters
    ----------
    waveform : np.ndarray
        Cartesian complex-valued waveform.
    pre_plus : complex
        Complex-valued prefactor for the plus polarization of the waveform.
    pre_cross : complex
        Complex-valued prefactor for the cross polarization of the waveform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Plus and cross polarizations: complex-valued arrays.
    """
    hp = pre_plus * waveform
    hc = pre_cross * waveform
    return hp, hc


def retrieve_best_trials_list() -> list[optuna.trial.FrozenTrial]:
    """Read the list of best trials which is provided
    with the package.

    This list's location can be found at the location
    defined by ``TRIALS_FILE``;
    if one wishes to modify it, a new one can be generated
    with the method :meth:`save_best_trials_to_file` of the
    class :class:`mlgw_bns.hyperparameter_optimization.HyperparameterOptimization`
    after an optimization job has been run.

    The current trials provided are the result of about 30 hours
    of optimization on a laptop.

    Returns
    -------
    list[optuna.trial.FrozenTrial]
        List of the trials in the Pareto front of an optimization.
    """

    stream = pkg_resources.resource_stream(__name__, TRIALS_FILE)
    return joblib.load(stream)


def best_trial_under_n(
    best_trials: list[optuna.trial.FrozenTrial], training_number: int
) -> Hyperparameters:
    """Utility function to retrieve
    a set of hyperparameters starting from a list of optimization trials.
    The best trial in terms of accuracy is returned.

    Parameters
    ----------
    best_trials : list[optuna.trial.FrozenTrial]
        List of trials in the Pareto front of an optimization run.
        A default value for such a list is the one provided by
        :func:`retrieve_best_trials_list`.
    training_number : int
        Return the best trial obtained while using less than this number
        of training waveforms.

    Returns
    -------
    Hyperparameters
        Hyperparameters corresponding to the best trial found.
    """

    accuracy = lambda trial: trial.values[0]

    # take the most accurate trial
    # which used less training data than the given
    # training number
    best_trial = sorted(
        [trial for trial in best_trials if trial.params["n_train"] <= training_number],
        key=accuracy,
    )[0]

    return Hyperparameters.from_frozen_trial(best_trial)


def cartesian_waveforms_at_frequencies(
    waveforms: FDWaveforms,
    new_frequencies: np.ndarray,
    dataset: Dataset,
    downsampling_training: DownsamplingTraining,
    downsampling_indices: DownsamplingIndices,
) -> np.ndarray:
    """Starting from an array of downsampled waveforms decomposed into
    amplitude and phase, interpolate them to a new frequency grid and
    put them in Cartesian form.

    Parameters
    ----------
    waveforms : FDWaveforms
        Waveforms to put in Cartesian form.
    new_frequencies : np.ndarray
        Frequencies to resample to, in natural units.
    dataset : Dataset
        Reference dataset.
    downsampling_training : DownsamplingTraining
        Training model for the downsampling, contains metadata needed for the resampling.
    downsampling_indices : DownsamplingIndices
        Downsampling indices, needed for the resampling.

    Returns
    -------
    np.ndarray
        Cartesian waveforms: complex array with shape
        ``(n_waveforms, len(new_frequencies))``.
    """

    amp_frequencies = dataset.frequencies[downsampling_indices.amplitude_indices]
    phi_frequencies = dataset.frequencies[downsampling_indices.phase_indices]

    amps = np.array(
        [
            downsampling_training.resample(amp_frequencies, new_frequencies, amp)
            for amp in waveforms.amplitudes
        ]
    )

    phis = np.array(
        [
            downsampling_training.resample(phi_frequencies, new_frequencies, phi)
            for phi in waveforms.phases
        ]
    )

    return combine_amplitude_phase(amps, phis)


@njit
def combine_amplitude_phase(amp: np.ndarray, phi: np.ndarray) -> np.ndarray:
    r"""Starting from arrays of amplitude :math:`A` and phase :math:`\phi`,
    return the cartesian waveform :math:`A e^{i \phi}`.

    Parameters
    ----------
    amp : np.ndarray
        Amplitude array.
    phi : np.ndarray
        Phase array.

    Returns
    -------
    np.ndarray
        Cartesian waveform.
    """
    return amp * np.exp(1j * phi)

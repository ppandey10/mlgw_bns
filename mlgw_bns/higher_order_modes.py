r""" This module deals with the generation of waveforms coming from a single mode 
of a spherical harmonic decomposition of the GW emission from a binary system.

As described in `the TEOBResumS documentation <https://bitbucket.org/eob_ihes/teobresums/wiki/Conventions,%20parameters%20and%20output>`_,
the full waveform is expressed as 

:math:`h_+ - i h_\times = \sum_{\ell m} A_{\ell m} e^{-i \phi_{\ell m}} Y_{\ell m}(\iota, \varphi)`

where the pair :math:`\ell, m`, with :math:`\ell \geq m`, is known as a *mode*, and
it is implemented as the namedtuple :class:`Mode`.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Callable, NamedTuple, Optional

import numpy as np

from .dataset_generation import (
    WaveformGenerator,
    WaveformParameters,
    amplitude_3h_post_newtonian,
    phase_5h_post_newtonian_tidal,
)

WaveformCallable = Callable[[WaveformParameters, np.ndarray], np.ndarray]

# Mode = namedtuple("Mode", ["l", "m"])
class Mode(NamedTuple):
    """A mode in the harmonic decomposition of the GW emission from a system."""

    l: int
    m: int


# TODO fix these, but it's not so bad now -
# these are only wrong by a constant scaling

_post_newtonian_amplitudes_by_mode: dict[Mode, WaveformCallable] = {
    Mode(2, 2): amplitude_3h_post_newtonian
}
_post_newtonian_phases_by_mode: dict[Mode, WaveformCallable] = {
    Mode(2, 2): phase_5h_post_newtonian_tidal
}


class ModeGenerator(WaveformGenerator):
    """Generic generator of a single mode for a waveform."""

    def __init__(self, mode: Mode, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        # see (https://github.com/python/mypy/issues/5887) for typing problem
        self.mode = mode

        # TODO improve the way these are handled
        self.supported_modes = list(_post_newtonian_amplitudes_by_mode.keys())
        if self.mode not in self.supported_modes:
            raise NotImplementedError(f"{self.mode} is not supported yet!")


class BarePostNewtonianModeGenerator(ModeGenerator):
    def post_newtonian_amplitude(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        return _post_newtonian_amplitudes_by_mode[self.mode](params, frequencies)

    def post_newtonian_phase(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        return _post_newtonian_phases_by_mode[self.mode](params, frequencies)

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ):
        raise NotImplementedError(
            "This generator does not include the possibility "
            "to generate effective one body waveforms"
        )


class TEOBResumSModeGenerator(BarePostNewtonianModeGenerator):
    def __init__(self, eobrun_callable: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eobrun_callable = eobrun_callable

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ):
        par_dict: dict = params.teobresums()

        # tweak initial frequency backward by a few samples
        # this is needed because of a bug in TEOBResumS
        # causing the phase evolution not to behave properly
        # at the beginning of integration
        # TODO remove this once the TEOB bug is fixed

        n_additional = 256
        f_0 = par_dict["initial_frequency"]
        delta_f = par_dict["df"]
        new_f0 = f_0 - delta_f * n_additional
        par_dict["initial_frequency"] = new_f0

        to_slice = (
            slice(-len(frequencies), None)
            if frequencies is not None
            else slice(n_additional, None)
        )

        if frequencies is not None:
            frequencies_list = list(
                np.insert(
                    frequencies,
                    0,
                    np.arange(f_0 - delta_f * n_additional, f_0, step=delta_f),
                )
            )
            par_dict.pop("df")
            par_dict["interp_freqs"] = "yes"
            par_dict["freqs"] = frequencies_list

        par_dict["arg_out"] = "yes"
        par_dict["output_multipoles"] = "yes"
        par_dict["use_mode_lm"] = [mode_to_k(self.mode)]
        par_dict["output_lm"] = [mode_to_k(self.mode)]

        f_spa, _, _, _, _, hflm, _, _ = self.eobrun_callable(par_dict)

        amplitude = hflm[str(mode_to_k(self.mode))][0]
        phase = hflm[str(mode_to_k(self.mode))][1]

        return (f_spa, amplitude, phase)


def mode_to_k(mode: Mode) -> int:
    """
    Map a mode to a unique integer identifier; needed because of
    TEOBResumS conventions.
    """
    return int(mode.l * (mode.l - 1) / 2 + mode.m - 2)

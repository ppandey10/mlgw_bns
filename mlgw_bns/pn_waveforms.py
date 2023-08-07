r"""Post-Newtonian modes. 

Appendix E of http://arxiv.org/abs/2001.10914

A good approximation for the phases is (eq. 4.8)
:math:`\phi_{\ell m} (f) \approx \frac{m}{2} \phi_{22} (2f / m)`

The convensions are defined in http://arxiv.org/abs/1601.05588:
we need 

:math:`\delta = \frac{m_1 - m_2}{M} = \frac{q-1}{q+1}`

as well as 

:math:`\chi_a^z = \frac{1}{2} (\vec{chi_1} - \vec{\chi_2}) \cdot \hat{L}_N = \frac{1}{2} (\chi_1 - \chi_2)`

and similarly for the symmetric component :math:`\chi_s^z`, with a plus sign.
"""

from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
from numba import njit 

from .taylorf2 import phase_5h_post_newtonian_tidal, smoothly_connect_with_zero

from .dataset_generation import (
    WaveformGenerator,
    WaveformParameters
)

if TYPE_CHECKING:
    from .dataset_generation import WaveformParameters


H_callable = Callable[[np.ndarray, float, float, float, float], np.ndarray]

Callable_Waveform = Callable[[WaveformParameters, np.ndarray], np.ndarray]

# In Python, a dictionary is a collection that allows us to store data in key: value pairs.
# _post_newtonian_amplitudes_by_mode: dict[Mode, Callable_Waveform] = {
#     Mode(2, 2): amp_lm(H_22, Mode(2, 2)),
# }
# _post_newtonian_phases_by_mode: dict[Mode, Callable_Waveform] = {
#     Mode(2, 2): phi_lm(Mode(2, 2)),
# }


class Mode(NamedTuple):
    """A mode in the harmonic decomposition of the GW emission from a system."""

    l: int
    m: int

    def opposite(self):
        return self.__class__(self.l, -self.m)


def H_22(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:

    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v6 = v3 * v3

    v2_coefficient = 451 * eta / 168 - 323 / 224

    v3_coefficient = (
        27 * delta * chi_a_z / 8 - 11 * eta * chi_s_z / 6 + 27 * chi_s_z / 8
    )

    v4_coefficient = (
        -49 * delta * chi_a_z * chi_s_z / 16
        + 105271 * eta ** 2 / 24192
        + 6 * eta * chi_a_z ** 2
        + eta * chi_s_z ** 2 / 8
        - 1975055 * eta / 338688
        - 49 * chi_a_z ** 2 / 32
        - 49 * chi_s_z ** 2 / 32
        - 27312085 / 8128512
    )

    # v6_coefficient = (
    #     107291 * delta * eta * chi_a_z * chi_s_z / 2688
    #     - 875047 * delta * chi_a_z * chi_s_z / 32256
    #     + 31 * np.pi * delta * chi_a_z / 12
    #     + 34473079 * eta**3 / 6386688
    #     + 491 * eta**2 * chi_a_z**2 / 84
    #     - 51329 * eta**2 * chi_s_z**2 / 4032
    #     - 3248849057 * eta**2 / 178827264
    #     + 129367 * eta * chi_a_z**2 / 2304
    #     + 8517 * eta * chi_s_z**2 / 224
    #     - 7 * np.pi * eta * chi_s_z / 3
    #     - 205 * np.pi**2 * eta / 48
    #     + 545384828789 * eta / 5007163392
    #     - 875047 * chi_a_z**2 / 64512
    #     - 875047 * chi_s_z**2 / 64512
    #     + 31 * np.pi * chi_s_z / 12
    #     + 428 * 1j * np.pi / 105
    #     - 177520268561 / 8583708672
    # )

    return (
        1 
        + v2 * v2_coefficient 
        + v3 * v3_coefficient 
        + v4 * v4_coefficient 
        )

def H_33(
    v: np.ndarray, 
    eta: float, 
    delta: float, 
    chi_a_z: float, 
    chi_s_z: float,
    ) -> np.ndarray:
    
    i = 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2

    H33_coefficient = -3/4 * i * np.sqrt(5/4)

    v1_coefficient = delta

    v3_coefficient = delta * (27 * eta / 8 - 1945 / 672)

    v4_coefficient = (
        -2 * delta * eta * chi_s_z / 3 
        + 65 * delta * chi_s_z / 24 
        + np.pi * delta - 21 * i * delta / 5 
        + 6 * i * delta * np.log(3 / 2)
        - 28 * eta * chi_a_z / 3 
        + 65 * chi_a_z / 24
        )

    return H33_coefficient * (
        v1_coefficient * v1 
        + v3_coefficient * v3 
        + v4_coefficient * v4
        )


def amp_lm(H_lm_callable: H_callable, mode: Mode):
    def function(params: "WaveformParameters", frequencies: np.ndarray) -> np.ndarray:

        v = np.abs(2 * np.pi * frequencies / mode.m) ** (1.0 / 3.0)

        delta = (params.mass_ratio - 1) / (params.mass_ratio + 1)
        chi_a_z = (params.chi_1 - params.chi_2) / 2
        chi_s_z = (params.chi_1 + params.chi_2) / 2
        
        return (
            np.pi 
            * np.sqrt(2 * params.eta / 3)
            * v ** (-7 / 2)
            * H_lm_callable(v, params.eta, delta, chi_a_z, chi_s_z)
        )

    return function


def phi_lm(mode: Mode):
    def function(params: "WaveformParameters", frequencies: np.ndarray) -> np.ndarray:
        return (
            phase_5h_post_newtonian_tidal(params, frequencies) * (mode.m / 2)
        )
    
    return function


def psi_lm(mode: Mode):
    def function(params: "WaveformParameters", frequencies: np.ndarray) -> np.ndarray:
        mode_freq = 2 * frequencies / mode.m
        return (
            phase_5h_post_newtonian_tidal(params, mode_freq) * (mode.m / 2) 
        )
    return function

# In Python, a dictionary is a collection that allows us to store data in key: value pairs.
_post_newtonian_amplitudes_by_mode: dict[Mode, Callable_Waveform] = {
    Mode(2, 2): amp_lm(H_22, Mode(2, 2)),
}
_post_newtonian_phases_by_mode: dict[Mode, Callable_Waveform] = {
    Mode(2, 2): phi_lm(Mode(2, 2)),
}
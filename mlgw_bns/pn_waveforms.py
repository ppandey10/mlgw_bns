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

from .taylorf2 import phase_5h_post_newtonian_tidal, smoothly_connect_with_zero, SUN_MASS_SECONDS

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

    v2_coefficient = (
        451 * eta / 168 
        - 323 / 224
    )

    v3_coefficient = (
        27 * delta * chi_a_z / 8 
        - 11 * eta * chi_s_z / 6 
        + 27 * chi_s_z / 8
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

    v6_coefficient = (
        107291 * delta * eta * chi_a_z * chi_s_z / 2688
        - 875047 * delta * chi_a_z * chi_s_z / 32256
        + 31 * np.pi * delta * chi_a_z / 12
        + 34473079 * eta**3 / 6386688
        + 491 * eta**2 * chi_a_z**2 / 84
        - 51329 * eta**2 * chi_s_z**2 / 4032
        - 3248849057 * eta**2 / 178827264
        + 129367 * eta * chi_a_z**2 / 2304
        + 8517 * eta * chi_s_z**2 / 224
        - 7 * np.pi * eta * chi_s_z / 3
        - 205 * np.pi**2 * eta / 48
        + 545384828789 * eta / 5007163392
        - 875047 * chi_a_z**2 / 64512
        - 875047 * chi_s_z**2 / 64512
        + 31 * np.pi * chi_s_z / 12
        + 428 * 1j * np.pi / 105
        - 177520268561 / 8583708672
    )

    return (
        1 
        + v2 * v2_coefficient 
        + v3 * v3_coefficient 
        + v4 * v4_coefficient 
        + v6 * v6_coefficient
        )

def H_21(
    v: np.ndarray, 
    eta: float, 
    delta: float, 
    chi_a_z: float, 
    chi_s_z: float,
    ) -> np.ndarray:

    i = 0 + 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H21_coef = i * np.sqrt(2) * 1/3

    v1_coef = delta

    v2_coef = (
        - 3/2 * delta * chi_s_z
        - 3/2 * chi_a_z
        )

    v3_coef = (
        117/56 * delta * eta
        + 335/672 * delta 
        )

    v4_coef = (
        - 965 / 336 * delta * eta * chi_s_z
        + 3427 / 1344 * delta * chi_s_z
        - np.pi * delta
        - i / 2 * delta
        - i / 2 * delta * np.log(16)
        - 2101 / 336 * eta * chi_a_z
        + 3427 / 1344 * chi_a_z
        )

    v5_coef = (
        21365 * delta * eta ** 2 / 8064
        + 10 * delta * eta * chi_a_z ** 2
        + 39 * delta * eta * chi_s_z ** 2 / 8
        - 36529 * delta * eta / 12544
        - 307 * delta * chi_a_z ** 2 / 32
        - 307 * delta * chi_s_z ** 2 / 32
        + 3 * np.pi * delta * chi_s_z
        - 964357 * delta / 8128512
        + 213 * eta * chi_a_z * chi_s_z / 4
        - 307 * chi_a_z * chi_s_z / 16
        + 3 * np.pi * chi_a_z
        )

    v6_coef = (
        - 547 * delta * eta ** 2 * chi_s_z / 768
        - 15 * delta * eta * chi_a_z ** 2 * chi_s_z
        - 3 * delta * eta * chi_s_z ** 3 / 16
        - 7049629 * delta * eta * chi_s_z / 225792
        + 417 * np.pi * delta * eta / 112
        - 1489 * i * delta *  eta / 112
        - 89 * i * delta * eta * np.log(2) / 28
        + 729 * delta * chi_a_z ** 2 * chi_s_z / 64
        + 243 * delta * chi_s_z ** 3 / 64
        + 143063173 * delta * chi_s_z / 5419008
        - 2455 * np.pi * delta / 1344
        - 335 * i * delta / 1344
        - 335 * i * delta * np.log(2) / 336
        + 42617 * eta ** 2 * chi_a_z / 1792
        - 15 * eta * chi_a_z ** 3 
        - 489 * eta * chi_a_z * chi_s_z ** 2 / 16
        - 22758317 * eta * chi_a_z / 225792
        + 243 * chi_a_z ** 3 / 64 
        + 729 * chi_a_z * chi_s_z ** 2 / 64
        + 143063173 * chi_a_z / 5419008
        )

    return H21_coef * ( 
        v1 * v1_coef
        + v2 * v2_coef
        + v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef 
        )

def H_31(
    v: np.ndarray, 
    eta: float, 
    delta: float, 
    chi_a_z: float, 
    chi_s_z: float,
    ) -> np.ndarray:

    i = 0 + 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H31_coef = i / (12 * np.sqrt(7))

    v1_coef = delta

    v3_coef = (
        17 * delta * eta / 24
        - 1049 * delta / 672
        )

    v4_coef = (
        10 * delta * eta * chi_s_z / 3
        + 65 * delta * chi_s_z / 24
        - np.pi * delta
        - 7 * i * delta / 5
        - i * delta * np.log(1024) / 5
        - 40 * eta * chi_a_z / 3
        + 65 * chi_a_z / 24
        )

    v5_coef = (
        - 4085 * delta * eta**2 / 4224
        + 10 * delta * eta * chi_a_z**2
        + delta * eta * chi_s_z**2 / 8
        - 272311 * delta * eta / 59136
        - 81 * delta * chi_a_z**2 / 32
        - 81 * delta * chi_s_z**2 / 32
        + 90411961 * delta / 89413632
        + 81 * eta * chi_a_z * chi_s_z / 4
        - 81 * chi_a_z * chi_s_z / 16
        )

    v6_coef = (
        803 * delta * eta**2 * chi_s_z / 72
        - 36187 * delta * eta * chi_s_z / 1344
        + 245 * np.pi * delta * eta / 48
        - 239 * i * delta * eta / 120
        - 5 * i * delta * eta * np.log(2) / 12
        + 264269 * delta * chi_s_z / 16128
        + 313 * np.pi * delta / 1344
        + 1049 * i * delta / 480
        + 1049 * i * delta * np.log(2) / 336
        + 2809 * eta**2 * chi_a_z / 72
        - 318205 * eta * chi_a_z / 4032
        + 264269 * chi_a_z / 16128
        )

    return H31_coef * (
        v1 * v1_coef
        + v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
        )

def H_32(
    v: np.ndarray, 
    eta: float, 
    delta: float, 
    chi_a_z: float, 
    chi_s_z: float,
    ) -> np.ndarray:

    i = 0 + 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H32_coef = 1 / 3 * np.sqrt(5 / 7)

    v2_coef = (
        1 - 3 * eta
        )

    v3_coef = (
        4 * eta * chi_s_z
        ) 

    v4_coef = (
        - 589 * eta**2 / 72
        + 12325 * eta / 2016 
        - 10471 / 10080
        )

    v5_coef = (
        eta * (113 * delta * chi_a_z / 8 + 1081 * chi_s_z / 84 - 66 * i / 5)
        + (-113 * delta * chi_a_z - 113 * chi_s_z + 72 * i) / 24
        - 115 * eta**2 * chi_s_z
        ) 

    v6_coef = (
        eta * (- 1633 * delta * chi_a_z * chi_s_z / 48 
        - 563 * chi_a_z**2 / 32 
        - 2549 * chi_s_z**2 / 96 
        + 8 * np.pi * chi_s_z 
        - 8689883 / 149022720)
        + 81 * delta * chi_a_z * chi_s_z / 16
        + 837223 * eta**3 / 63360 
        + eta**2 * (30 * chi_a_z**2
        + 313 * chi_s_z**2 / 24
        - 78584047 / 2661120)
        + 81 * chi_a_z**2 /32
        + 81 * chi_s_z**2 / 32
        + 824173699 / 447068160
        )

    return H32_coef * (
        v2 * v2_coef
        + v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
        ) 

def H_33(
    v: np.ndarray, 
    eta: float, 
    delta: float, 
    chi_a_z: float, 
    chi_s_z: float,
    ) -> np.ndarray:
    
    i = 0 + 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H33_coefficient = -3/4 * i * np.sqrt(5/7)

    v1_coefficient = delta

    v3_coefficient = delta * (27 * eta / 8  - 1945 / 672)

    v4_coefficient = (
        -2 * delta * eta * chi_s_z / 3 
        + 65 * delta * chi_s_z / 24 
        + np.pi * delta 
        - 21 * i * delta / 5 
        + 6 * i * delta * np.log(3 / 2)
        - 28 * eta * chi_a_z / 3 
        + 65 * chi_a_z / 24
        )
    
    v5_coefficient = (
        420389 * delta * eta ** 2 / 63360
        + 10 * delta * eta * chi_a_z ** 2
        + delta * eta * chi_s_z ** 2 / 8
        - 11758073 * delta * eta / 887040
        - 81 * eta * chi_a_z ** 2 / 32
        - 81 * eta * chi_s_z ** 2 / 32
        - 1077664867 * delta / 447068160
        + 81 * eta * chi_a_z * chi_s_z / 4
        - 81 * chi_a_z * chi_s_z / 16
    )

    v6_coefficient = (
        - 67 * delta * eta ** 2 * chi_s_z / 72
        - 58745 * delta * eta * chi_s_z / 4032
        + 131 * np.pi * delta * eta / 16
        - 440957 * i * delta * eta / 9720
        + 69 * i * delta * eta * np.log(3 / 2) / 4
        + 163021 * delta * chi_s_z / 16128
        - 5675 * np.pi * delta / 1344
        + 389 * i * delta / 32
        - 1945 * i * delta * np.log(3/2) / 112
        - 137 * eta ** 2 * chi_a_z / 24
        - 148501 * eta * chi_a_z / 4032
        + 163021 * chi_a_z / 16128
    )

    h33 = (
        v1_coefficient * v1  
        + v3_coefficient * v3 
        + v4_coefficient * v4 
        + v5_coefficient * v5 
        + v6_coefficient * v6
        )

    return (H33_coefficient * h33)

def H_43(
    v: np.ndarray, 
    eta: float, 
    delta: float, 
    chi_a_z: float, 
    chi_s_z: float,
    ) -> np.ndarray:

    i = 0 + 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H43_coef = 3 * i * np.sqrt(3/35) / 4

    v3_coef = (
        2 * delta * eta
        - delta
        )

    v4_coef = (
        5 * eta * chi_a_z / 2
        - 5 * delta * eta * chi_s_z / 2
        )

    v5_coef = (
        887 * delta * eta**2 / 132
        - 10795 * delta * eta / 1232
        + 18035 * delta / 7392
        )

    v6_coef = (
        - 469 * delta * eta**2 * chi_s_z / 48
        + 4399 * delta * eta * chi_s_z / 448
        + 2 * np.pi * delta * eta 
        - 16301 * i * delta * eta / 810
        + 12 * i * delta * eta * np.log(3/2)
        - 113 * delta * chi_s_z / 24
        - np.pi * delta
        + 32 * i * delta / 5
        - 6 * i * delta * np.log(3/2)
        - 1642 * eta**2 * chi_a_z / 48 
        + 41683 * eta * chi_a_z / 1344
        - 113 * chi_a_z / 24
        )

    return H43_coef * (
        v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
        )

def H_44(
    v: np.ndarray, 
    eta: float, 
    delta: float, 
    chi_a_z: float, 
    chi_s_z: float,
    ) -> np.ndarray:

    i = 0 + 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H44_coef = np.sqrt(10/7) * 4 / 9

    v2_coef = (
        3 * eta - 1
        )

    v4_coef = (
        1063 * eta**2 / 88
        - 128221 * eta / 7392
        + 158383 / 36960
        )

    v5_coef = (
        np.pi * (2 - 6 * eta)
        - eta * (1695 * eta * chi_a_z
        + 2075 * chi_s_z - 3579 * i
        + 2880 * i * np.log(2)) / 120
        + (565 * delta * chi_a_z
        + 1140 * eta**2 * chi_s_z 
        + 565 * chi_s_z - 1008 * i
        + 960 * i * np.log(2)) / 120
        )

    v6_coef = (
        eta * (243 * delta * chi_a_z * chi_s_z / 16
        + 563 * chi_a_z**2 / 32 
        + 247 * chi_s_z**2 / 32 
        - 22580029007 / 880588800)
        - 81 * delta * chi_a_z * chi_s_z / 16
        - 7606537 * eta**3 / 274560
        + eta**2 * (-30 * chi_a_z**2 
        - 3 * chi_s_z**2 / 8
        + 901461137 / 11531520)
        - 81 * chi_a_z**2 / 32
        - 81 * chi_s_z**2 / 32
        + 7888301437 / 29059430400
        )

    return H44_coef * (
        v2 * v2_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
        )



def amp_lm(H_lm_callable: H_callable, mode: Mode):
    def function(params: "WaveformParameters", frequencies: np.ndarray) -> np.ndarray:

        v = np.abs(2 * np.pi * frequencies / mode.m) ** (1.0 / 3.0)
        delta = (params.mass_ratio - 1) / (params.mass_ratio + 1)
        chi_a_z = (params.chi_1 - params.chi_2) / 2
        chi_s_z = (params.chi_1 + params.chi_2) / 2
        
        return (
            np.abs(
                np.pi 
                * np.sqrt(2 * params.eta / 3)
                * v ** (-7 / 2)
                * H_lm_callable(v, params.eta, delta, chi_a_z, chi_s_z)
                )
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
    Mode(2, 1): amp_lm(H_21, Mode(2, 1)),
    Mode(2, 2): amp_lm(H_22, Mode(2, 2)),
    Mode(3, 1): amp_lm(H_31, Mode(3, 1)),
    Mode(3, 2): amp_lm(H_32, Mode(3, 2)),
    Mode(3, 3): amp_lm(H_33, Mode(3, 3)),
    Mode(4, 3): amp_lm(H_43, Mode(4, 3)),
    Mode(4, 4): amp_lm(H_44, Mode(4, 4)),
}
_post_newtonian_phases_by_mode: dict[Mode, Callable_Waveform] = {
    Mode(2, 1): psi_lm(Mode(2, 1)),
    Mode(2, 2): psi_lm(Mode(2, 2)),  
    Mode(3, 1): psi_lm(Mode(3, 1)),
    Mode(3, 2): psi_lm(Mode(3, 2)),     
    Mode(3, 3): psi_lm(Mode(3, 3)),
    Mode(4, 3): psi_lm(Mode(4, 3)),
    Mode(4, 4): psi_lm(Mode(4, 4)),
}
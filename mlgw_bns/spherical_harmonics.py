import numpy as np

from .dataset_generation import (
    WaveformGenerator,
    WaveformParameters
)

from .pn_waveforms import (
    Mode
)

# Define spin-weighted spherical harmonics for the modes
def Y_2_pos_2(
    inclination: float, 
    phi_c: float # phase at merger
):
    return (
        1/2 * np.sqrt(5/np.pi) * np.cos(inclination/2) ** 4 * np.exp(2*1j*phi_c)
    )

def Y_2_neg_2(
    inclination: float, 
    phi_c: float 
):
    return (
        1/2 * np.sqrt(5/np.pi) * np.sin(inclination/2) ** 4 * np.exp(-2*1j*phi_c)
    )

def Y_2_pos_1(
    inclination: float, 
    phi_c: float 
):
    return (
        1/2 * np.sqrt(5/np.pi) * (np.cos(inclination/2) ** 2) * np.sin(inclination) * np.exp(1j*phi_c)
    )

def Y_2_neg_1(
    inclination: float, 
    phi_c: float 
):
    return (
        1/2 * np.sqrt(5/np.pi) * (np.sin(inclination/2) ** 2) * np.sin(inclination) * np.exp(-1j*phi_c)
    )

def Y_3_pos_3(
    inclination: float, 
    phi_c: float 
):
    return (
        1/2 * np.sqrt(21/2*np.pi) * (np.cos(inclination/2) ** 4) * np.sin(inclination) * (-np.exp(3j*phi_c))
    )

def Y_3_neg_3(
    inclination: float, 
    phi_c: float 
):
    return (
        1/2 * np.sqrt(21/2*np.pi) * (np.sin(inclination/2) ** 4) * np.sin(inclination) * np.exp(-3j*phi_c)
    )

def Y_4_pos_4(
    inclination: float, 
    phi_c: float 
):
    return (
        3/4 * np.sqrt(7/np.pi) * (np.cos(inclination/2) ** 4) * (np.sin(inclination) ** 2) * np.exp(4j*phi_c)
    )

def Y_4_neg_4(
    inclination: float, 
    phi_c: float 
):
    return (
        3/4 * np.sqrt(7/np.pi) * (np.sin(inclination/2) ** 4) * (np.sin(inclination) ** 2) * np.exp(-4j*phi_c)
    )




    

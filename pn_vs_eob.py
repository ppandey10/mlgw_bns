from mlgw_bns.model import *

from mlgw_bns.higher_order_modes import (
    BarePostNewtonianModeGenerator,
    TEOBResumSModeGenerator,
    Mode,
    ModeGenerator,
    ModeGeneratorFactory,
    spherical_harmonic_spin_2,
    spherical_harmonic_spin_2_conjugate, 
    teob_mode_generator_factory,
)

from mlgw_bns.pn_waveforms import (
    Mode,
   _post_newtonian_amplitudes_by_mode,
   _post_newtonian_phases_by_mode 
)

import matplotlib.pyplot as plt

f = np.linspace(20, 2048, num=1<<20)

d = Dataset(20., 4096.)
f_natural = d.hz_to_natural_units(f)

p = ParametersWithExtrinsic.gw170817()
p_teob = p.intrinsic(d)

m = Mode(2, 2)
waveform_generator = teob_mode_generator_factory(m)
print(waveform_generator)

pn_amp = _post_newtonian_amplitudes_by_mode[m](p_teob, f_natural)
_, teob_amp, teob_phase = waveform_generator.effective_one_body_waveform(p_teob, f_natural)

print(teob_amp[-1] - pn_amp[-1])
# plt.loglog(f_natural, pn_amp)
# plt.loglog(f, teob_amp - pn_amp)
# plt.show()


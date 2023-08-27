from mlgw_bns.model import *

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

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

from mlgw_bns.data_management import (
    Residuals
)

model = Model.default_for_testing()

modes_model = ModesModel(filename = "pp_small_default", modes=[Mode(2,2)])
modes_model.load()
mmodel = modes_model.models[Mode(2, 2)]
f = np.linspace(20, 1024, num=1<<20)

d = Dataset(20., 4096.)
f_natural = d.hz_to_natural_units(f)
print(f_natural[1:4])

p = ParametersWithExtrinsic.gw170817()
p_teob = p.intrinsic(d)

m = Mode(2, 2)
waveform_generator = teob_mode_generator_factory(m)
print(waveform_generator)

pn_amp = _post_newtonian_amplitudes_by_mode[m](p_teob, f_natural)
pn_phase = _post_newtonian_phases_by_mode[m](p_teob, f_natural)
# _, teob_amp, teob_phase = model.waveform_generator.effective_one_body_waveform(p_teob, f_natural)
_, teob_amp, teob_phase = mmodel.waveform_generator.effective_one_body_waveform(p_teob, f_natural)

log_teob_pn_amp = np.log(np.abs(teob_amp) / np.abs(pn_amp))
phase_residuals = teob_phase - pn_phase

# res = Residuals(log_teob_pn_amp[np.newaxis, :], phase_residuals[np.newaxis, :])
res = Residuals(teob_amp[np.newaxis, :], teob_phase[np.newaxis, :])
res.flatten_phase(f_natural)

# plt.loglog(f, pn_amp, label = 'PN')
# plt.loglog(f, teob_amp, label = 'EOB')
# plt.semilogx(f, log_teob_pn_amp)
plt.plot(f_natural, pn_phase, label = 'PN')
plt.plot(f_natural, teob_phase, label = 'EOB')
# plt.plot(f_natural, res.phase_residuals[0], label = 'EOB')
plt.legend()
plt.ylabel(r'$A_{lm}$', fontsize = 15)
plt.xlabel(r'$f (Hz)$', fontsize = 15)
plt.tight_layout()
# plt.savefig('amp_pn_teob.pdf', dpi = 800)
plt.show()


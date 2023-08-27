from mlgw_bns.model import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.coolwarm
import matplotlib as mpl
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
from mlgw_bns.data_management import Residuals

f = np.linspace(20, 1024, num=1<<20)

modes_model = ModesModel(filename = "pp_small_default", modes=[Mode(2,2)])
modes_model.load()
mmodel = modes_model.models[Mode(2, 2)]

d = Dataset(20., 4096.)
f_natural = d.hz_to_natural_units(f)

p = ParametersWithExtrinsic.gw170817()
p_teob = p.intrinsic(d)

m = Mode(2, 2)
waveform_generator = teob_mode_generator_factory(m)

# lam = np.linspace(10, 2000, num=20)

# for i in range(len(lam)):

#     p = ParametersWithExtrinsic(1.2, lam[i], lam[i], 0, 0, 40, 0, 2.8)
#     p_teob = p.intrinsic(d)

#     amp, phase = mmodel.predict_amplitude_phase(f, p)
#     _, amp_teob, phase_teob = mmodel.waveform_generator.effective_one_body_waveform(p_teob, f_natural)
#     log_teob_pn_amp = np.log(np.abs(amp) / np.abs(amp_teob))
#     phase_diff = phase - phase_teob
#     res = Residuals(log_teob_pn_amp[np.newaxis, :], phase_diff[np.newaxis, :])
#     res.flatten_phase(f_natural)

#     # plt.semilogx(f, log_teob_pn_amp, color=cmap(i / len(lam)))
#     plt.semilogx(f_natural, res.phase_residuals[0], color=cmap(i / len(lam)))
# sm = plt.cm.ScalarMappable(cmap=cmap)
# sm.set_array(lam)
# plt.colorbar(sm)
# plt.ylabel('$\phi_P - \phi_E$', fontsize = 15)
# plt.xlabel('$Mf$', fontsize = 15)
# plt.tight_layout()
# plt.savefig('phi_residuals.pdf', dpi = 800)
# plt.show()

pn_phase = _post_newtonian_phases_by_mode[m](p_teob, f_natural)
amp, phase = mmodel.predict_amplitude_phase(f, p)
_, amp_teob, phase_teob = mmodel.waveform_generator.effective_one_body_waveform(p_teob, f_natural)

log_teob_pn_amp = np.log(np.abs(amp) / np.abs(amp_teob))

res_teob = Residuals(amp_teob[np.newaxis, :], phase_teob[np.newaxis, :])
res_pred = Residuals(amp[np.newaxis, :], phase[np.newaxis, :])
res_teob.flatten_phase(f_natural)
res_pred.flatten_phase(f_natural)


residual_phase =  res_teob.phase_residuals[0] - res_pred.phase_residuals[0]

# hp_m, hc_m = mmodel.predict(f, p)
# hp_teob, hc_teob = mmodel.waveform_generator.generate_full_teob_waveform(p_teob, f_natural)
# plt.plot(f_natural, hp_teob.real)
# plt.show()

# plt.loglog(f, amp)
# plt.loglog(f, amp_teob)
# plt.plot(f, log_teob_pn_amp)
# plt.semilogx(f_natural, res_teob.phase_residuals[0] - res_pred.phase_residuals[0])
plt.plot(f_natural, res_teob.phase_residuals[0], label = "EOB")
plt.plot(f_natural, res_pred.phase_residuals[0], label = "Predicted")
# plt.plot(f_natural, res.phase_residuals[0])
plt.ylabel('$\phi_P - \phi_E$', fontsize = 15)
plt.xlabel('$Mf$', fontsize = 15)
plt.tight_layout()
plt.legend()
# plt.savefig('phi_residuals.pdf', dpi = 800)
plt.show()

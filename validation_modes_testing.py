from mlgw_bns.model import *
import numpy as np
import matplotlib.pyplot as plt
from mlgw_bns.data_management import Residuals
from mlgw_bns.model_validation import *
from mlgw_bns.resample_residuals import *

f = np.linspace(20, 1024, num=1<<20)

d = Dataset(20., 4096.)
f_natural = d.hz_to_natural_units(f)
print(f_natural[1:4])

p = ParametersWithExtrinsic.gw170817()
p_teob = p.intrinsic(d)

main_model = ModesModel(filename = "pp_small_default", modes=[Mode(2,2)])
main_model.load()
mmode = main_model.models[Mode(2, 2)]
print(mmode.dataset.waveform_generator)

main_v_model = ValidateModel(mmode)

teob_wave, pred_wave = main_v_model.teob_and_pred_wavforms(1)

# plt.plot(aligned_teob, label = 'EOB')
# plt.plot(aligned_pred, label = 'Predicted')
# plt.show()

# print(teob_wave[0].real)'
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize  = (9, 7))
ax1.plot(teob_wave[0].real, label = 'EOB')
ax1.plot(pred_wave[0].real, label = 'Predicted')
ax1.set_ylabel(r'Re($h_{22}$)', fontsize = 12)
ax1.set_xlabel(r'f(Hz)', fontsize = 12)
ax1.legend()

ax2.plot(teob_wave[0].imag, label = 'EOB')
ax2.plot(pred_wave[0].imag, label = 'Predicted')
ax2.set_xlim([400, 600])
ax2.set_ylim([-4000, 4000])
ax2.set_ylabel(r'Re($h_{22}$)', fontsize = 12)
ax2.set_xlabel(r'f(Hz)', fontsize = 12)
ax2.legend()

ax3.plot(teob_wave[0].imag, label = 'EOB')
ax3.plot(pred_wave[0].imag, label = 'Predicted')
ax3.set_xlim([1200, 1400])
ax3.set_ylim([-400, 400])
ax3.set_ylabel(r'Re($h_{22}$)', fontsize = 12)
ax3.set_xlabel(r'f(Hz)', fontsize = 12)
ax3.legend()

fig.tight_layout()
# plt.savefig('cartesian_waveforms_eob_pred.pdf', dpi = 800)
plt.show()

# print(main_v_model.frequencies)
# main_mismatches = main_v_model.validation_mismatches(100)
# print(np.average(main_mismatches))
# plt.plot(main_mismatches)



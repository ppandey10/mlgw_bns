from mlgw_bns.model import *
import numpy as np
import matplotlib.pyplot as plt
from mlgw_bns.data_management import Residuals

f = np.linspace(20, 2048, num=1<<20)

modes_model = ModesModel(filename = "pp_small_default", modes=[Mode(2,2)])
modes_model.load()
mmodel = modes_model.models[Mode(2, 2)]

# mmodel.generate(100, 10000, 10000)
# mmodel.set_hyper_and_train_nn()

model = Model.default_for_testing()
print(mmodel.waveform_generator)

# for lam in np.linspace(10, 1000, num=20):
#     p_ext = ParametersWithExtrinsic(1.2, lam, lam, 0, 0, 40, 0, 2.8)
#     p_int = p_ext.intrinsic(Dataset(20., 4096.))
#     amp_mode, phase_mode = mmodel.predict_amplitude_phase(f, p_ext)
#     # _, amp_mode_teob, phase_mode_teob = mmodel.waveform_generator.effective_one_body_waveform(p_int, f)
#     # hpn, hcn = model.predict(f, p)

#     # res = Residuals(
#     #     np.log(abs(hcm) / abs(hcn))[np.newaxis, :],
#     #     (np.unwrap(np.angle(hcm)) - np.unwrap(np.angle(hcn)))[np.newaxis, :]
#     # )
    
#     # res.flatten_phase(f)
#     # plt.loglog(f, amp_mode)
#     # plt.loglog(f, amp_mode_teob)
#     # plt.loglog(f, phase_mode)
#     # plt.plot(f, amp_mode_teob)
# plt.show()

p = ParametersWithExtrinsic(1, 400, 400, 0, 0, 40, 0, 2.8)
p_teob = p.intrinsic(Dataset(20., 4096.))
# plt.semilogy(f, model.predict_amplitude_phase(f, p)[1])
# plt.loglog(f, mmodel.predict_amplitude_phase(f, p)[0])
plt.loglog(f, model.waveform_generator.effective_one_body_waveform(p_teob, f)[2])
# plt.semilogy(f, mmodel.predict_amplitude_phase(f, p)[1])
plt.show()
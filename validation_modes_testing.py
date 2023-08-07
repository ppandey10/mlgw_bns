from mlgw_bns.model import *
import numpy as np
import matplotlib.pyplot as plt
from mlgw_bns.data_management import Residuals
from mlgw_bns.model_validation import *


f = np.linspace(20, 2048, num=1<<20)

# mmodel = Model(filename = 'pp')
# mmodel.load()
# print(mmodel.dataset.waveform_generator) 
# v_model = ValidateModel(mmodel)
# mismatches = v_model.validation_mismatches(10)
# print(mismatches)

main_model = ModesModel(filename = "pp_small_default", modes=[Mode(2,2)])
mmode = main_model.models[Mode(2, 2)]
mmode.load()
# mmode = mode_dict[Mode(l=2, m=2)]
print(mmode.dataset.waveform_generator)
print(mmode.nn)

main_v_model = ValidateModel(mmode)
main_mismatches = main_v_model.validation_mismatches(10)
print(main_mismatches)



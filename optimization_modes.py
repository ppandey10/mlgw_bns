from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization
from mlgw_bns.model import *

def hyper_opt_fun():
    main_model = ModesModel(filename = "pp_small_default", modes=[Mode(2,2)])
    main_model.load()
    mmodel = main_model.models[Mode(2, 2)]
    ho = HyperparameterOptimization(mmodel)
    ho.optimize_and_save(0.1)


if __name__ == "__main__":
    hyper_opt_fun()
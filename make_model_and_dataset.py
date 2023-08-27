from mlgw_bns.model import *
import matplotlib.pyplot as plt


def modes_main():
    modes_model = ModesModel(filename = "pp_small_default", modes=[Mode(2,2)])
    modes_model.generate()
    modes_model.set_hyper_and_train_nn()
    modes_model.save()

    mmode = modes_model.models[Mode(2, 2)]
    loss_over_epochs = mmode.nn.get_loss_over_epochs()
    plt.plot(loss_over_epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss_over_epochs.pdf', dpi = 800)
    plt.show()

    # import os; os.replace...

def pp_main():
    model = Model(filename = "pp")
    model.generate()
    model.set_hyper_and_train_nn()
    model.save(False)


if __name__ == "__main__":
    #pp_main()
    modes_main()
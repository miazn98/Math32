from create_dataset import create_dataset
from train import train
from EulNet import eulerNet, cnn
from torch import random, save
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torchsummary import summary
from torch import cuda
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from cl_create_dataset import create_dataset as cd

from torch import nn

# seed the random number generator. Remove the line below if you want to try different initializations






def euler(model_type="base",
                       data_path="image_categorization_dataset.pt",
                       contrast_normalization=False, whiten=False):
    """
    Invokes the dataset creation, the model construction and training functions

    Arguments
    --------
    model_type: (string), the type of model to train. Use 'base' for the base model and 'improved for the improved model. Default: base
    data_path: (string), the path to the dataset. This argument will be passed to the dataset creation function
    contrast_normalization: (boolean), specifies whether or not to do contrast normalization
    whiten: (boolean), specifies whether or not to whiten the data.

    """
    # Do not change the output path
    # but you can uncomment the exp_dir if you do not want to save the model checkpoints
    output_path = "{}_euler_dataset.pt".format(model_type)
    exp_dir = "./{}_models".format(model_type)

    model_type = "reg"
    imp_outputPath = "{}_euler_dataset.pt".format(model_type)
    imp_expDir = "./{}_models".format(model_type)
    print(output_path)

    train_ds, val_ds, weights = create_dataset()
    # train_ds, val_ds = cd("image_categorization_dataset.pt")


    # specify the network architecture and the training policy of the models under
    # the respective blocks
    if model_type == "reg":
        train_opts = {}
        train_opts["momentum"] = 0.95
        train_opts["lr"] = 0.13
        train_opts["weight_decay"] = 0.0001
        train_opts["batch_size"] = 256
        train_opts["num_epochs"] = 35
        train_opts["step_size"] = 30
        train_opts["gamma"] = 0.1
        train_opts["criterion"] = CrossEntropyLoss(weight=weights)
        # train_opts["criterion"] = CrossEntropyLoss()
        model = eulerNet()
        print(model)

    elif model_type == "cnn":
        # create netspec_opts
        netspec_opts = {}

        netspec_opts["kernel_size"] = [3,3,2,
        3,3,
        2,
        3, 3,
        3]
        netspec_opts["stride"] = [1, 1, 2,
         1,1,
         2,
         1,1,
         2]
        netspec_opts["filters"] = [16,16, 0,
         32,32,
         0,
         64, 64,
         8]
        netspec_opts["layer_type"] = ["conv", "conv", "pool",
        "conv", "conv",
        "pool",
        "conv", "conv",
        "pred"]

        netspec_opts = {}

        netspec_opts["kernel_size"] = [3,3,3,2,
        3,3,2,
        3, -2,3,-2,3, 3,
        1]
        netspec_opts["stride"] = [1, 1, 1, 2,
         1,1,2,
         1,-2,1,-2,1,1,
         1]
        netspec_opts["filters"] = [16,16,32, 0,
         32,32,0,
         64, -2, 64,-2, 64, 0,
         8]
        netspec_opts["layer_type"] = ["conv","conv","conv","pool",
        "conv", "conv", "pool",
         "conv","dropout", "conv", "dropout", "conv", "pool",
        "pred"]

        # netspec_opts["kernel_size"] = [3,3,2,
        # 3,
        # 2,
        # 3,
        # 3]
        # netspec_opts["stride"] = [1, 1, 2,
        #  1,
        #  2,
        #  1,
        #  2]
        # netspec_opts["filters"] = [8,8, 0,
        #  16,
        #  0,
        #  32,
        #  8]
        # netspec_opts["layer_type"] = ["conv", "conv", "pool",
        # "conv",
        # "pool",
        # "conv",
        # "pred"]





        # netspec_opts["kernel_size"] = [3,3,3,2,
        # 3,3,2,
        # 3, -2,3,-2,3, 3,
        # 1]
        # netspec_opts["stride"] = [1, 1, 1, 2,
        #  1,1,2,
        #  1,-2,1,-2,1,1,
        #  1]
        # netspec_opts["filters"] = [8,8,16, 0,
        #  16,16,0,
        #  32, -2, 32,-2, 32,
        #  22]
        # netspec_opts["layer_type"] = ["conv","conv","conv","pool",
        # "conv", "conv", "pool",
        #  "conv","dropout", "conv", "dropout", "conv",
        # "pred"]

        netspec_opts["dropout"] = 0.2
        train_opts = {}
        train_opts["momentum"] = 0.9
        train_opts["lr"] = 0.1
        # train_opts["lr"] = 0.001
        train_opts["weight_decay"] = 0.001
        train_opts["batch_size"] = 256
        train_opts["num_epochs"] = 40
        train_opts["step_size"] = 30
        train_opts["gamma"] = 0.1
        train_opts["criterion"] = CrossEntropyLoss(weight=weights)
        # train_opts["criterion"] = CrossEntropyLoss()

        # create improved model

        model = cnn(netspec_opts)
        print(model)

    else:
        raise ValueError(f"Error: unknown model type {model_type}")
    print(train_opts.keys())
    # uncomment the line below if you wish to resume training of a saved model
    # model.load_state_dict(load(PATH to state))
    print("training " + model_type + "......")
    if cuda.is_available():
            print('activating cuda"')

            model.cuda()
    # train the model
    train(model, train_ds, val_ds, train_opts, exp_dir)
    # save model's state and architecture to the base directory
    model = {"state": model.state_dict(), "specs": netspec_opts}
    save(model, "{}-model.pt".format(model_type))

    plt.savefig(f"{model_type}-euler.png")
    plt.show()


if __name__ == '__main__':
    # Change the default values for the various parameters to your preferred values
    # Alternatively, you can specify different values from the command line
    # For example, to change model type from base to improved
    # type <cnn_categorization.py --model_type improved> at a command line and press enter
    args = ArgumentParser()
    args.add_argument("--model_type", type=str, default="base", required=False,
                      help="The model type must be either base or improved")
    args.add_argument("--data_path", type=str, default="image_categorization_dataset.pt",
                      required=False, help="Specify the path to the dataset")
    args.add_argument("--contrast_normalization", type=bool, default=False, required=False,
                      help="Specify whether or not to do contrast_normalization")
    args.add_argument("--whiten", type=bool, default=False, required=False,
                      help="Specify whether or not to whiten value")

    args, _ = args.parse_known_args()
    euler(**args.__dict__)

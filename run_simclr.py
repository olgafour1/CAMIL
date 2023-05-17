import pandas as pd
from args import parse_args, set_seed
from training.SIMCLR import SIMCLR
import os
from flushed_print import print
import numpy as np
from sklearn.model_selection import train_test_split

def load_encoder(check_dir):
    """
    Loads the appropriate siamese model using the information of the fold of k-cross
    fold validation and the id of experiment
    Parameters
    ----------
    check_dir  : directory where the weights of the pretrained siamese network. Weight files are stored in the format:
    weights-irun:d-ifold:d.hdf5
    irun       : int referring to the id of the experiment
    ifold      : int referring to the fold from the k-cross fold validation

    Returns
    -------
    returns  a Keras model instance of the pre-trained siamese net
    """

    encoder = SIMCLR(args).encoder
    try:
        file_path = os.path.join(check_dir, "encoder/sim_clr.ckpt")
        encoder.load_weights(file_path)
        return encoder
    except:
        print("no weight file found")
        return None

def load_projection_head(check_dir):
    """
    Loads the appropriate siamese model using the information of the fold of k-cross
    fold validation and the id of experiment
    Parameters
    ----------
    check_dir  : directory where the weights of the pretrained siamese network. Weight files are stored in the format:
    weights-irun:d-ifold:d.hdf5
    irun       : int referring to the id of the experiment
    ifold      : int referring to the fold from the k-cross fold validation

    Returns
    -------
    returns  a Keras model instance of the pre-trained siamese net
    """

    projection_head= SIMCLR(args).projection_head
    try:
        file_path = os.path.join(check_dir, "projection/sim_clr.ckpt")
        projection_head.load_weights(file_path)
        return projection_head
    except:
        print("no weight file found")
        return None


if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(args.seed_value)


    acc = []
    precision = []
    recall = []
    auc =[]
    fscore=[]

    csv_file=args.csv_file

    fold_id = os.path.splitext(csv_file)[0].split("_")[3]

    os.makedirs(os.path.join(args.simclr_path, "fold_{}".format(fold_id)), exist_ok=True)

    references=pd.read_csv(csv_file)

    train_bags= references["train"].apply(lambda x:os.path.join(args.feature_path,x+".h5")).values.tolist()

    def func_val(x):
        value = None
        if isinstance(x, str):
                value = os.path.join(args.feature_path, x + ".h5")
        return value

    val_bags = references.apply(lambda row: func_val(row.val), axis=1).dropna().values.tolist()
    test_bags = references.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

    train_bags=np.concatenate((train_bags,val_bags,test_bags))

    train_bags, val_bags = train_test_split(train_bags, test_size = 0.1, random_state = 42)

    test_simclr_model = None

    if args.retrain:
            encoder = load_encoder(os.path.join(args.simclr_path, "fold_{}".format(fold_id)))
            projection_head = load_projection_head(os.path.join(args.simclr_path, "fold_{}".format(fold_id)))
            simclr_net = SIMCLR(args)
            test_simclr_model = simclr_net.train(train_bags,
                                                 val_bags,
                                                 os.path.join(args.simclr_path,
                                                              "fold_{}".format(fold_id)),
                                                 encoder=encoder,
                                                 projection_head=projection_head)
    if test_simclr_model is None:

            simclr_net = SIMCLR(args)
            test_simclr_model = simclr_net.train(train_bags,
                                                 val_bags,
                                                 os.path.join(args.simclr_path,
                                                 "fold_{}".format(fold_id)))




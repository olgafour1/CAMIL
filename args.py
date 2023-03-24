import argparse
import os
import random
import numpy as np
import tensorflow as tf


def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class objecttransmil_ac
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train Graph Att net')

    parser.add_argument('--dataset', dest='dataset',
                        help='select dataset',
                        choices=["camelyon","tcga", 'sarcoma'],
                        default="camelyon", type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory where the weights of the model are stored',
                        default="camelyon_Saved_model", type=str)
    parser.add_argument('--simclr_path', dest='simclr_path',
                        help='directory where the images are stored',
                        default="camelyon_SIMCLR_checkpoints", type=str)
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=0.0002, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=1e-5, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train GRAPH MIL',
                        default=50, type=int)
    parser.add_argument('--seed_value', dest='seed_value',
                        help='use same seed value for reproducability',
                        default=12321, type=int)
    parser.add_argument('--run', dest='run',
                        help='number of experiments to be run',
                        default=1, type=int)
    parser.add_argument('--k', dest='k',
                        help='number of neighbors taken into account',
                        default=3, type=int,
                        choices=range(1,12), metavar="[1-12]")
    parser.add_argument('--feature_path', dest='feature_path',
                        help='directory where the images are stored',
                        default='/home/admin_ofourkioti/Documents/camelyon/resnet_feats/h5_files/', type=str)
    parser.add_argument('--dataset_path', dest='data_path',
                        help='directory where the images are stored',
                        default="",
                        type=str)
    parser.add_argument('--experiment_name', dest='experiment_name',
                        help='the name of the experiment needed for the logs',
                        default="test", type=str)
    parser.add_argument('--simclr_batch_size', dest='simclr_batch_size',
                        help='batch size used bu the siamese network',
                        default=512, type=int)
    parser.add_argument('--simclr_epochs', dest='simclr_epochs',
                        help='epochs for siamese network',
                        default=50, type=int)
    parser.add_argument('--input_shape', dest="input_shape",
                        help='shape of the image',
                        default=(1024,), type=int, nargs=3)
    parser.add_argument('--label_file', dest="label_file",
                        help='csv file with information about the labels',
                       default="label_files/camelyon_data.csv",type=str)
    parser.add_argument('--csv_files', dest="csv_files",
                        help='csv file with information about the labels',
                       default="camelyon_csv_files",type=str)
    parser.add_argument('--raw_save_dir', dest="raw_save_dir",
                        help='directory where the attention weights are saved',
                        default="heatmaps", type=str)
    parser.add_argument('--retrain', dest="retrain",
                        action='store_true', default=False)
    parser.add_argument('--save_exp_code', type=str, default=None,
                        help='experiment code')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
    parser.add_argument('--subtyping', dest="subtyping",
                        action='store_true', default=False)


    args = parser.parse_args()
    return args

#/run/user/1001/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_features/h5_files/'
#/run/user/1001/gvfs/smb-share:server=rds.icr.ac.uk,share=data/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/TCGA_data/TCGA_flat
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/TCGA_data/TCGA_features/h5_files/
#"/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/TCGA_data/TCGA_features/h5_files/"

def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

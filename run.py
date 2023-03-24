import pandas as pd
from args import parse_args
from training.charm import CHARM
import os
from flushed_print import print
import tensorflow as tf
import numpy as np

def set_seed(seed: int = 42) -> None:
  import random
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(12321)

    csv_files=args.csv_files
    acc=[]
    recall=[]
    f_score=[]
    auc=[]
    precision=[]

    for csv_file in os.listdir(csv_files):

            fold_id = os.path.splitext(csv_file)[0].split("_")[1]

            os.makedirs(os.path.join(args.simclr_path, "fold_{}".format(fold_id)), exist_ok=True)

            references=pd.read_csv(os.path.join(csv_files,csv_file))

            train_bags= references["train"].apply(lambda x:os.path.join(args.feature_path,x+".h5")).values.tolist()

            def func_val(x):
                value = None
                if isinstance(x, str):
                        value = os.path.join(args.feature_path, x + ".h5")
                return value

            val_bags =  references.apply(lambda row: func_val(row.val), axis=1).dropna().values.tolist()

            test_bags = references.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

            train_net = CHARM(args)

            train_net.train(train_bags, fold_id, val_bags, args)

            test_net = CHARM(args)

            test_acc, test_auc, test_precision, test_recall, test_f_score = test_net.predict(test_bags,
                                                                                                 fold_id,
                                                                                                 args,
                                                                                                 test_model=test_net.model)
            acc.append(test_acc)
            recall.append(test_recall)
            f_score.append(test_f_score)
            auc.append(test_auc)
            precision.append(test_precision)


    print('mean accuracy = ', np.mean(acc))
    print('std = ', np.std(acc))
    print(' mean precision = ', np.mean(precision))
    print('std = ', np.std(precision))
    print('mean recall = ', np.mean(recall))
    print('std = ', np.std(recall))
    print(' mean fscore = ', np.mean(f_score))
    print('std = ', np.std(f_score))
    print(' mean auc = ', np.mean(auc))
    print('std = ', np.std(auc))






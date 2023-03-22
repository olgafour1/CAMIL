import os
import time
from dataset_utils.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
import argparse
from utils.file_utils import save_hdf5
from flushed_print import print
import h5py
import openslide
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.spatial import distance
from training.SIMCLR import SIMCLR

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
@tf.function
def serve(model, x):
    return model(x, training=False)

def generate_values(projection_head,images, Idx, dist="euclidean"):
    """

    Parameters
    ----------
    images :  np.ndarray of size (numnber of patches,h,w,d) contatining the pathes of an image
    Idx    : indices of the closest neighbors of every image

    Returns
    -------
    a list of np.ndarrays, pairing every patch of an image with its closest neighbors

    """

    rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()
    columns = Idx.ravel()
    values = []
    for row, column in zip(rows, columns):
        m1 = serve(projection_head,np.expand_dims(images[int(row)], axis=0))
        m2 = serve(projection_head,np.expand_dims(images[int(column)], axis=0))
        value = distance.cdist(m1.numpy().reshape(1, -1), m2.numpy().reshape(1, -1), dist)[0][0]
        values.append(value)
    values = np.reshape(values, (Idx.shape[0], Idx.shape[1]))
    return values

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

def feature_extractor(patch_size):
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(patch_size, patch_size, 3))

    layer_name = 'conv4_block6_out'
    intermediate_model = tf.keras.Model(inputs=resnet.input,
                                        outputs=resnet.get_layer(layer_name).output)
    out = GlobalAveragePooling2D()(intermediate_model.output)

    return tf.keras.Model(inputs=resnet.input, outputs=out)

def compute_w_loader(file_path, output_path, wsi,
                     simclr_check_dir,
                     sim_feat_output_path,
                     fold_id,
                     batch_size=8, pretrained=True,
                     custom_downsample=1, target_patch_size=-1,
                    ):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """

    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    ot = (tf.float32, tf.float32)
    loader = tf.data.Dataset.from_generator(dataset, output_types=ot).batch(batch_size)

    resnet = feature_extractor(256)
    encoder=load_encoder(simclr_check_dir)
    projection_head = load_projection_head(simclr_check_dir)

    mode = 'a'
    wsi_coords=[]
    simclr_features=[]
    res_feats=[]
    for count, (batch, coords) in enumerate(loader):
                resnet_features = serve(resnet, batch)
                features = serve(encoder, batch)
                wsi_coords.append(coords.numpy())
                simclr_features.append(features)
                res_feats.append(resnet_features)
                asset_dict = {'features': resnet_features.numpy(), 'coords': coords.numpy()}
                simclr_feat_asset_dict = {'features': features.numpy(), 'coords': coords.numpy()}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                save_hdf5(sim_feat_output_path, simclr_feat_asset_dict, attr_dict=None, mode=mode)

    wsi_coords = np.vstack(wsi_coords)
    simclr_features = np.vstack(simclr_features)
    res_feats = np.vstack(res_feats)
    patch_distances = pairwise_distances(wsi_coords, metric='euclidean', n_jobs=1)
    neighbor_indices = np.argsort(patch_distances, axis=1)[:, :12 + 1]
    similarities = generate_values(projection_head,simclr_features, neighbor_indices)
    resnet_similarities = generate_values(projection_head, res_feats, neighbor_indices)

    asset_dict = {'indices': neighbor_indices, 'similarities_{}'.format(fold_id):similarities}
    resnet_asset_dict = {'indices': neighbor_indices, 'similarities_{}'.format(fold_id): resnet_similarities}
    save_hdf5(output_path, resnet_asset_dict, attr_dict=None, mode=mode)
    save_hdf5(sim_feat_output_path, asset_dict, attr_dict=None, mode=mode)

    return sim_feat_output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default='features')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--simclr_path', type=str, default='lipo_SIMCLR_checkpoints/fold_0/')
parser.add_argument('--simclr_feat_dir', type=str, default='simclr_features')
parser.add_argument('--experiment_name', dest='experiment_name',default="transformer_k", type=str)
parser.add_argument('--simclr_batch_size', dest='simclr_batch_size', default=512, type=int)
parser.add_argument('--simclr_epochs', dest='simclr_epochs',default=50, type=int)
parser.add_argument('--retrain', dest="retrain",action='store_true', default=False)
parser.add_argument('--fold_id', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(args.simclr_feat_dir, exist_ok=True)

    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    os.makedirs(os.path.join(args.simclr_feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'h5_files'))

    print('loading model checkpoint')

    total = len(bags_dataset)

    for bag_candidate_idx in range(total):

        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]

        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        #
        # if not args.no_auto_skip and slide_id + '.h5' in dest_files:
        #     print('skipped {}'.format(slide_id))
        #     continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        sim_feat_output_path = os.path.join(args.simclr_feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            args.simclr_path,
                                            sim_feat_output_path,
                                            fold_id=args.fold_id,
                                            batch_size=args.batch_size,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)


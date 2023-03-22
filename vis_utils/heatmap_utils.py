import numpy as np
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from dataset_utils.wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.spatial import distance
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
import math


@tf.function
def serve(x,trained_model):
    return trained_model(x, training=False)

def get_affinity(Idx,features,k, simclr):
    """
    Create the adjacency matrix of each bag based on the euclidean distances between the patches
    Parameters
    ----------
    Idx:   a list of indices of the closest neighbors of every image
    Returns
    -------
    affinity:  an nxn np.ndarray that contains the neighborhood information for every patch.
    """


    all_rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()
    all_columns = Idx.ravel()

    values = []
    for row, column in zip(all_rows, all_columns):
        m1 = serve(np.expand_dims(features[int(row)], axis=0), simclr)
        m2 = serve(np.expand_dims(features[int(column)], axis=0),simclr)
        value = distance.cdist(m1.numpy().reshape(1, -1), m2.numpy().reshape(1, -1), "euclidean")[0][0]
        values.append(value)

    values = np.reshape(values, (Idx.shape[0], Idx.shape[1]))

    neighbor_indices = Idx[:, :k + 1]

    rows = np.asarray([[enum] * len(item) for enum, item in enumerate(neighbor_indices)]).ravel()
    columns = neighbor_indices.ravel()

    neighbor_matrix = values[:, 1:]

    normalized_matrix = preprocessing.normalize(neighbor_matrix, norm="l2")

    similarities = np.log((normalized_matrix + 1) / (normalized_matrix + np.finfo(np.float32).eps))
    values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)

    values = values[:, :k + 1]

    values = values.ravel().tolist()

    sparse_matrix = tf.sparse.SparseTensor(indices=list(zip(rows, columns)),
                                               values=values,
                                               dense_shape=[neighbor_indices.shape[0], neighbor_indices.shape[0]])

    return sparse_matrix


def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)

    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def feature_extractor(patch_size):
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(patch_size, patch_size, 3))

    layer_name = 'conv4_block6_out'
    intermediate_model = tf.keras.Model(inputs=resnet.input,
                                        outputs=resnet.get_layer(layer_name).output)
    out = GlobalAveragePooling2D()(intermediate_model.output)

    return  tf.keras.Model(inputs=resnet.input, outputs=out)


def compute_from_patches_tf(wsi_object,k,sim_clr,
                            batch_size,model,
                            attn_save_path, ref_scores,path,
                            **wsi_kwargs):
    model.model.load_weights(path)

    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    print('total number of patches to process: ', len(roi_dataset))
    ot = (tf.float32, tf.float32)

    roi_loader = tf.data.Dataset.from_generator(roi_dataset,
                                        output_types=ot)

    num_batches=math.ceil(len(roi_dataset)/batch_size)
    print('total number of batches to process: ',num_batches)

    resnet=feature_extractor(256)

    roi_loader = roi_loader.batch(batch_size)
    mode = "w"


    def test_step(images):
        predictions, attn = model.model(images, training=False)
        return predictions, attn.numpy()

    for idx, batch in enumerate(roi_loader):
        roi=batch[0]
        coords=batch[1]
        patch_distances = pairwise_distances(coords, metric='euclidean', n_jobs=1)
        neighbor_indices = np.argsort(patch_distances, axis=1)[:, :10 + 1]

        roi=preprocess_input(roi)
        features=resnet.predict(roi)

        sparse_matrix=get_affinity(neighbor_indices,features,k, sim_clr)

        if attn_save_path is not None:
            predictions, A = test_step([features,sparse_matrix])

        if ref_scores is not None:
            for score_idx in range(len(A)):
                A[score_idx] = score2percentile(A[score_idx], ref_scores)

        if idx % math.ceil(num_batches * 0.05) == 0:
            print('processed {} / {}'.format(idx, num_batches))


        asset_dict = {'attention_scores': A, 'coords': coords.numpy()}
        save_hdf5(attn_save_path, asset_dict, mode=mode)
        mode = "a"

    return attn_save_path, wsi_object


# def compute_from_patches(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,
#     attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):
#     top_left = wsi_kwargs['top_left']
#     bot_right = wsi_kwargs['bot_right']
#     patch_size = wsi_kwargs['patch_size']
#
#
#     roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
#     roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
#     print('total number of patches to process: ', len(roi_dataset))
#     num_batches = len(roi_loader)
#     print('number of batches: ', len(roi_loader))
#     mode = "w"
#     for idx, (roi, coords) in enumerate(roi_loader):
#
#         roi = roi.to(device)
#         coords = coords.numpy()
#
#         with torch.no_grad():
#             features = feature_extractor(roi)
#
#             if attn_save_path is not None:
#                 A = model(features, attention_only=True)
#
#                 if A.size(0) > 1: #CLAM multi-branch attention
#                     A = A[clam_pred]
#
#                 A = A.view(-1, 1).cpu().numpy()
#
#                 if ref_scores is not None:
#                     for score_idx in range(len(A)):
#                         A[score_idx] = score2percentile(A[score_idx], ref_scores)
#
#                 asset_dict = {'attention_scores': A, 'coords': coords}
#                 save_hdf5(attn_save_path, asset_dict, mode=mode)
#
#         if idx % math.ceil(num_batches * 0.05) == 0:
#             print('procssed {} / {}'.format(idx, num_batches))
#
#         if feat_save_path is not None:
#             asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
#             save_hdf5(feat_save_path, asset_dict, mode=mode)
#
#         mode = "a"
#     return attn_save_path, feat_save_path, wsi_object
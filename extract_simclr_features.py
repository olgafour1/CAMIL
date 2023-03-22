import os
import time
from dataset_utils.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
import argparse
from utils.file_utils import save_hdf5
from flushed_print import print
import h5py
import openslide
import tensorflow as tf
import numpy as np

def compute_w_loader(file_path, wsi,
                     simclr_h5_path,
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

    mode = 'a'
    wsi_coords=[]
    for count, (batch, coords) in enumerate(loader):
                wsi_coords.append(coords.numpy())
                batchImages = []
                for enum, coord in enumerate(coords):
                    img = wsi.read_region(coord.numpy(), 1, (256, 256)).convert('RGB')
                    img = np.array(img, dtype="float32")
                    img = np.array(img, dtype="uint8")
                    img = np.expand_dims(img, axis=0)
                    batchImages.append(img)
                simclr_asset_dict = {'imgs': np.vstack(batchImages), 'coords': coords.numpy()}
                save_hdf5(simclr_h5_path, simclr_asset_dict, attr_dict=None, mode=mode)
    return simclr_h5_path

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--simclr_h5_dir', type=str, default='simclr')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()

if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.simclr_h5_dir, exist_ok=True)

    os.makedirs(os.path.join(args.simclr_h5_dir, 'h5_files'), exist_ok=True)

    print('loading model checkpoint')

    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]

        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))



        sim_clr_output_path = os.path.join(args.simclr_h5_dir, 'h5_files', bag_name)

        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, wsi, sim_clr_output_path,
                                            batch_size=args.batch_size,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['imgs'][:]
        print('features size: ', features.shape)


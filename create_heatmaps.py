from __future__ import print_function
import numpy as np
import argparse
import pdb
import os
import pandas as pd
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches_tf
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from training.SIMCLR import SIMCLR
from training.charm import CHARM_subtype, CHARM
from dataset_utils.camelyon_batch_generator import DataGenerator
from args import parse_args, set_seed
import tensorflow as tf
import shutil
from PIL import Image
from sklearn.metrics import pairwise_distances
from os import path

def infer_single_slide(model,features, reverse_label_dict, k=1):

    model.model.load_weights(checkpoint_path)
    test_gen = DataGenerator(args=contr_mil_args, fold_id=0, batch_size=1, filenames=[features], train=False)

    @tf.function(experimental_relax_shapes=True)
    def test_step(images):
        predictions, attn = model.model(images, training=False)
        return predictions, attn

    x_batch_val, y_batch_val= (list(test_gen)[0])

    Y_prob, A = test_step(x_batch_val)

    Y_hat = np.round(np.clip(Y_prob, 0, 1)).tolist()

    preds_str = np.array([reverse_label_dict[Y_hat[0][0]]])

    return Y_hat[0][0],preds_str[0], Y_prob.numpy()[0][0], A.numpy()


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params
def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict


if __name__ == '__main__':

    contr_mil_args = parse_args()
    fold_id=0
    test_net = CHARM(contr_mil_args)
    checkpoint_path = os.path.join(os.path.join(contr_mil_args.save_dir, str(fold_id), contr_mil_args.experiment_name),
                                   "{}.ckpt".format(contr_mil_args.experiment_name))

    config_path = os.path.join('heatmaps/configs', contr_mil_args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(contr_mil_args, config_dict)


    def load_siamese(check_dir):
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

        projection_head_test = SIMCLR(contr_mil_args).projection_head
        try:
            file_path = os.path.join(check_dir, "sim_clr.ckpt")

            projection_head_test.load_weights(file_path)
            print("weight file found")
            return projection_head_test
        except:
            print("no weight file found")
            return None

    sim_clr=load_siamese("/home/admin_ofourkioti/PycharmProjects/WSI-level-MIL/SIMCLR_checkpoints/"+"fold_{}".format(fold_id))

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))

    decision = input('Continue? Y/N ')
    if decision in ['Y', 'y', 'Yes', 'yes']:
        pass
    elif decision in ['N', 'n', 'No', 'NO']:
        exit()
    else:
        raise NotImplementedError

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1],
                                                                                  patch_args.overlap,
                                                                                  step_size[0], step_size[1]))
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)

    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size,
                         'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                         'use_center_shift': heatmap_args.use_center_shift}

    for i in range(len(process_stack)):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None

        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
                                    filter_params=filter_params)

        xml_dir="/run/user/1001/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/testing/lesion_annotations"
        xml_path=os.path.join(xml_dir,slide_id+".xml")

        if os.path.exists(xml_path):
            print ("Annotation file exists")
            wsi_object.initXML(xml_path)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple(
            (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        vis_params['line_thickness'] = 250
        mask = wsi_object.visWSI(**vis_params, number_contours=False,annot_display=True)
        mask.save(mask_path)

        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')

        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')
        origin_h5_dir=os.path.join(contr_mil_args.feature_path, slide_id + '.h5')

        if not os.path.isfile(h5_path):
            try:
                shutil.copy(origin_h5_dir, h5_path)
                print("File copied successfully.")

            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

        base_name = os.path.splitext(os.path.basename(h5_path))[0]

        Y_hats,Y_hats_str, Y_probs, A = infer_single_slide(test_net,h5_path, reverse_label_dict)


        if not os.path.isfile(block_map_save_path):

            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')),
                                 index=False)
        else:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)

        file = h5py.File(block_map_save_path, 'r')

        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]

        file.close()

        samples = sample_args.samples
        for enum, sample in enumerate(samples):
            if sample['sample']:

                tag = "label_{}_pred_{}".format(label, enum)
                sample_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches',
                                               base_name,
                                               str(tag), sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
                                             score_start=sample.get('score_start', 0),
                                             score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in enumerate(
                        zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level,
                                                       (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir,
                                            '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1],
                                                                                  s_score)))


        samples = sample_args.samples
        for enum, sample in enumerate(samples):
            if sample['sample']:

                root_neigh_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'neigboring_patches',base_name)
                os.makedirs(root_neigh_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
                                             score_start=sample.get('score_start', 0),
                                             score_end=sample.get('score_end', 1))

                for idx, (s_coord, s_score) in enumerate(
                        zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    tag = "patch_{}_{}".format(s_coord[0], s_coord[1])
                    neigh_save_dir=os.path.join(root_neigh_save_dir, tag)
                    os.makedirs(neigh_save_dir,exist_ok=True)
                    patch_distances = pairwise_distances(s_coord.reshape(1, -1),coords, metric='euclidean', n_jobs=1)
                    neighbor_indices = np.argsort(patch_distances, axis=1)[:, :10 + 1]
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level,
                                                       (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch.save(os.path.join(neigh_save_dir,
                                            '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1],
                                                                                  s_score)))
                    for enum,index in enumerate(neighbor_indices[0]):
                        k_coords=coords[index]
                        patch = wsi_object.wsi.read_region(tuple(k_coords), patch_args.patch_level,
                                                           (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                        patch.save(os.path.join(neigh_save_dir,
                                                '{}_{}_x_{}_y_{}_k_{}.png'.format(idx, slide_id, k_coords[0], k_coords[1],enum )))

        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size,
                      'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                      'use_center_shift': heatmap_args.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)

        """when using the camelyon16 dataset, we set the  convert_to_percentiles parameter to False"""
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            pass
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                                  alpha=heatmap_args.alpha,
                                  use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                                  thresh=-1, patch_size=vis_patch_size, convert_to_percentiles=False)

            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(r_slide_save_dir,
                                 '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))


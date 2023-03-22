#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/camelyon_results/create_features.txt
#SBATCH --error=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/camelyon_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/

python extract_features_tf.py --data_h5_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/size_256 --data_slide_dir  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/SAR/ --csv_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/size_256/process_list_autogen_ndpi.csv --simclr_feat_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/tf_feats_256/simclr_feats/  --feat_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/tf_feats_256/resnet_feats/ --batch_size 1024 --slide_ext .ndpi

#python scores.py  --dataset_path  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_slides/slides  --csv_file camelyon_csv_files/fold_0.csv  --new_path  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/SIMCLR_feats_256/ --feature_path   /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/tf_feats_256/h5_files/
#python scores.py --siamese_weights_path   SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_features/clam_features/h5_files/  --csv_file camelyon_csv_files/fold_0.csv --new_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_features/h5_files/

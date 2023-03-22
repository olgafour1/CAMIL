#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/CHARM/camelyon_results/simclr_run_lipo.txt
#SBATCH --error=/home/ofourkioti/Projects/CHARM/camelyon_results/simclr_run_lipo.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/CHARM/

#python run.py  --k 6 --dataset camelyon --experiment_name k_6 --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/tf_feats_256/resnet_feats/h5_files --label_file label_files/camelyon_data.csv --csv_files camelyon_csv_files  --epoch 100 --save_dir cam_Saved_model
python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024

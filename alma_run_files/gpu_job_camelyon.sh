#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/camelyon_results/dense_epoch_100.txt
#SBATCH --error=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/camelyon_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/

for i in {0..4};
do python run.py  --k 3 --experiment_name dense_epoch_100  --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/dsmil-feats/h5_files/ --label_file label_files/camelyon_data.csv --csv_file camelyon_csv_files/splits_${i}.csv  --epoch 100;
done
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/images/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/patches/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/


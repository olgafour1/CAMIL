#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/cam_17_results/cam_17.txt
#SBATCH --error=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/cam_17_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/

#python run.py --dataset camelyon --experiment_name phi --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/tf_feats_256/resnet_feats/h5_files --label_file label_files/camelyon_data.csv --csv_files camelyon_csv_files/splits_0.csv  --epoch 100 --save_dir cam_Saved_model --lambda1 0.00001
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
for i in {0..4};
do python run.py --experiment_name cam_17  --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/feats/h5_files/ --label_file label_files/camelyon_17.csv --csv_file camelyon_17_splits/splits_${i}.csv --lambda1 1 --epoch 200 --save_dir cam17_Saved_model
done


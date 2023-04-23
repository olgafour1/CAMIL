#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/tcga_results/nystromformer_k_2.txt
#SBATCH --error=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/tcga_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/



#python run.py --epochs 100 --experiment_name nystromformer_k_2  --k 2 --dataset tcga --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/TCGA_data/features/h5_files  --label_file label_files/tcga_data.csv --csv_files tcga_csv_files --save_dir TCGA_Saved_model

for i in {0..4};
do python run.py --epochs 100 --experiment_name nystromformer_k_2  --k 2 --dataset tcga --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/TCGA_data/features/h5_files  --label_file label_files/tcga_data.csv --csv_files tcga_csv_files/splits_${i}.csv --save_dir TCGA_Saved_model;
done
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024

#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=50:00:00
#SBATCH --output=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/camelyon_results/ndpi_imgs.out
#SBATCH --error=/home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/camelyon_results/ndpi_imgs.err
#SBATCH --partition=smp

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/Neighbor_constrained_attention_based_MIL/

#python run.py  --k 3 --distance exp  --dataset camelyon --experiment_name  dual_loss_beta_07  --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/camelyon_features/h5_files/  --label_file label_files/camelyon_data.csv --csv_files camelyon_csv_files  --temperature 2 --epoch 100 --mode siamese
#python scores.py  --csv_file camelyon_csv_files/fold_0.csv  --feature_path  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/features/features_256/torch/h5_files  --new_path  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/features/features_256/torch/h5_files

python extract_simclr_features.py --data_slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/SAR/ --data_h5_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/size_256/ --csv_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/size_256/process_list_autogen_ndpi.csv --simclr_h5_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs  --batch_size 1024 --slide_ext .ndpi

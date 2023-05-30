# CAMIL: Context-Aware Multiple Instance Learning for Whole Slide Image Classification

This is an implementation of our model CAMIL (Context-Aware Multiple Instance Learning for Whole Slide Image Classification). CAMIL leverages the dependencies among single tiles within a WSI and imposes contextual constraints as prior knowledge to multiple instance learning model. We tested CAMIL on the subtyping of non-small cell lung cancer (NSLC) and lymph node (LN) metastasis, and the results demonstrated its superiority over other state-of-the-art weakly supervised classification algorithms. 

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

<img src="ieee_figure.png" 
     alt="model figure">

### To install

We recommend creating a new conda environment. In the terminal, run:
```bash 
conda env create --file exp_env.yml

pip install --upgrade pip
``` 
This should take ~5mins or less.

### Hardware requirements
We have tested this software on an Ubuntu 20.04LTS and 18.04LTS with 128Gb RAM and NVIDIA Quadro RTX 6000 GPU.

Using the publicly available CLAM WSI-prepossessing toolbox, we first automatically segmented the tissue region from each slide and exhaustively divided it into 256x256 non-overlapping patches using $\times 20$ magnification. It is important to note that using different parameters in the feature extraction process can lead to different training and test sets, which can subsequently result in different model performance. To ensure the reproducibility of our results, we provide the features that we used in our study.[h5_files](https://drive.google.com/drive/folders/1sJ4EuY08EIw6bpXoMcUhr0rga7YXyHot?usp=sharing)


### SimCLR model
<img src="fig2_ieee.png" 
     alt="simCLR model">


Due to the nature of SimCLR and its underlying mechanisms, using a batch size lower than 512 may result in suboptimal performance and diminished learning outcomes. The algorithm relies on a large batch size to effectively capture and learn from the diverse representations within the data. A smaller batch size could limit the model's ability to generalize well and extract meaningful features, potentially leading to reduced accuracy or convergence issues. Therefore, it is advisable to adhere to a batch size of 512 or higher for optimal results when utilizing SimCLR.


The datasets used for
SimCLR consist of patches extracted from the WSIs. To extract the patches run the folloqing command:
If you do not have the necessary hardware resources to accommodate a batch size of 512 or higher, it may be best to skip the step of training SimCLR altogether.

```bash 
python extract_simclr_patches.py --simclr_h5_dir simclr_imgs/ --data_h5_dir  patches/  --data_slide_dir  --csv_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/process_list_autogen.csv --slide_ext .tif
```
To train the simCLR model:

```bash 
python run_simclr.py --simclr_path  SIMCLR_checkpoints --feature_path patches/   --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
```

## Usage
To run the CAMIL model:

```bash 
python run.py  --k 3 --experiment_name exp_name  --feature_path h5_files --label_file label_files/camelyon_data.csv --csv_file camelyon_csv_files/splits_0.csv
```





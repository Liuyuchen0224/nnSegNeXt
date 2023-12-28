# nnSegNeXt: A 3D Convolutional Network for Brain Tissue Segmentation based on Quality Evaluation

---
## Installation
#### 1. System requirements
We run nnSegnext on a system running Ubuntu 18.04, with Python 3.9, PyTorch 1.10.1, and CUDA 10.7. For a full list of software packages and version numbers, see the Conda environment file `environment.yml`. 

This software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation. Thus, systems lacking a suitable GPU would likely take an extremely long time to train or evaluate models. The software was tested with the NVIDIA RTX 2080 TI GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory. 

#### 2. Installation guide
We recommend installation of the required packages using the conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：
```
git clone https://github.com/Liuyuchen0224/nnSegNeXt.git
cd nnSegnext
conda env create -f environment.yml
source activate nnSegnext
pip install -e .
```

#### 3. Functions of scripts and folders
- **For evaluation:**
  - ``nnSegnext/nnsegnext/inference.py``
  
- **Data split:**
  - ``nnSegnext/nnsegnext/dataset_json/``
  
- **For inference:**
  - ``nnSegnext/nnsegnext/inference/predict_simple.py``
  
- **Network architecture:**
  - ``nnSegnext/nnsegnext/network_architecture/nnSegnext.py``
  
- **For training:**
  - ``nnSegnext/nnsegnext/run/run_training.py``
  
- **Trainer for dataset:**
  - ``nnSegnext/nnsegnext/training/network_training/nnSegnextTrainerV2_nnsegnext.py``

---

## Training
#### 1. Dataset download
Datasets can be acquired by request.

#### 2. Setting up the datasets
After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./Pretrained_weight/
./nnSegnext/
./DATASET/
  ├── nnSegnext_raw/
      ├── nnSegnext_raw_data/
          ├── Task01_HCP/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_IXI/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_SALD/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── nnSegnext_cropped_data/
  ├── nnSegnext_trained_models/
  ├── nnSegnext_preprocessed/
```
You can refer to ``nnSegnext/nnsegnext/dataset_json/`` for data split.

After that, you can preprocess the above data using following commands:
```
nnSegnext_convert_decathlon_task -i ../DATASET/nnSegnext_raw/nnSegnext_raw_data/Task01_HCP
nnSegnext_convert_decathlon_task -i ../DATASET/nnSegnext_raw/nnSegnext_raw_data/Task02_IXI
nnSegnext_convert_decathlon_task -i ../DATASET/nnSegnext_raw/nnSegnext_raw_data/Task03_SALD

nnSegnext_plan_and_preprocess -t 1
nnSegnext_plan_and_preprocess -t 2
nnSegnext_plan_and_preprocess -t 3
```

#### 3. Training and Testing
- Commands for training and testing:

```
bash train_inference.sh -c 0 -n nnsegnext -t 1 
#-c stands for the index of your cuda device
#-n denotes the suffix of the trainer located at nnSegnext/nnsegnext/training/network_training/
#-t denotes the task index
```
If you want use your own data, please create a new trainer file in the path ```nnsegnext/training/network_training``` and make sure the class name in the trainer file is the same as the trainer file. Some hyperparameters could be adjust in the trainer file, but the batch size and crop size should be adjust in the file```nnsegnext/run/default_configuration.py```.
 


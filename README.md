# Automatic Detection of Microsleep Events using Modified Sleep Staging Techniques
- Sleep detection beyond the 30 second rule

Author: Sindri Jónsson

s202056@student.dtu.dk

---
## Description
This GitHub repository contains the code for my master thesis project conducted at the Techincal University of Denmark (DTU). The project focused on the development of automatic classification models for microsleep detection by adapting established automatic sleep staging models. This consisted of three different methods of post-processing and thresholding (pat-USleep), fine-tuning via transfer learning (mU-Sleep), and self-supervise learning (mU-SSL).

## Structure of the repository
The model development and evaluation was developed using Python (notebooks) and Matlab. 

At the root of the folder the following files and folders are of interest:


| **File Name**                               | **Description**                                                                     | **Type** |
|---------------------------------------------|-------------------------------------------------------------------------------------|----------|
| make_usleep_predictions.ipynb                                        | Create sleep predictions from U-Sleep at desired prediction rates.| Notebook   |
| ssl.ipynb                                        | Notebook for self-supervised learning model development (mU-SSL)                                     | Notebook   |
|SSL_18022023| Folder containing model weights and training history for mU-SSL | Folder | 
|transfer_learning.ipynb| Notebook for developing U-sleep using transfer learning (mU-Sleep) | Notebook |
|SSL_data|Raw data used in this thesis with zero-padding for windowing data in SSL task|Folder|
|TL_17022023| Folder containing model weights and training history of mU-Sleep | Folder |
|malafeev_ref.ipynb| Notebook to replicate a 16 second CNN as benchmark model \cite{} | Notebook |
|malafeev42_new | Contains model weights and training history of the CNN-16s | Folder |
|edf_data/| Raw data used for this thesis converted to EDF files |Folder|
|Matlab/| Folder where benchmark models were replicated and models were evaluated | Folder |
|Matlab/main.m| Main script for running benchmark models and evaluating all the models| Script |
|Matlab/main_comparison/|Folder containing the benchmark models \cite{} and model comparisons| Folder |
|Matlab/data/|Raw data used for this thesis (76 MWT recordings) \cite{}|Folder|
|Matlab/utils/|Folder containing helper functions used by main script and during analysis|Folder|
|Matlab/analysis/|Folder containing model analysis scripts|Folder|


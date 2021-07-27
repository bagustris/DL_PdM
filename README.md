# Deep Learning for Predictive Maintenance

This module contains the code used in our survey paper: [Deep Learning and Its Applications to Machine Health Monitoring](https://www.researchgate.net/publication/311839720_Deep_Learning_and_Its_Applications_to_Machine_Health_Monitoring_A_Survey). It has been published in [Mechanical Systems and Signal Processing](https://linkinghub.elsevier.com/retrieve/pii/S0888327018303108). This version is [my fork](https://github.com/bagustrius) from the [original authors](https://github.com/ClockworkBunny/MHMS_DEEPLEARNING.) to keep the code run in Python 3.6 and Tensorflow 1.15.5.

## Table of Contents

<!-- TOC START min:1 max:3 link:true update:true -->
- [Deep Learning and Its Applications to Machine Health Monitoring: A Survey](#DL-MHMS)
  - [Table of Contents](#table-of-contents)
  - [Data](#data)
  - [Code](#code)
    - [Feature Extraction](#feature-extraction)
    - [Deep Learning Models](#deep-learning-models)
    - [Machine Learning Models](#machine-learning-models)
    - [Main Test](#main-test)
<!-- TOC END -->



## Data
This folder contains two pickle files, which are extracted features and labels for tool wear sensing experiments. Each pickle file contain x_train, y_train, x_test, y_test. The task is defined as a regression problem.

- data_normal: each data sample is a vector. The features are extracted from the whole time sequences. 
- data_seq: each data sample is a tensor. The features are extracted from windows of the time sequences. 

Especially, data_seq can be used by LSTM and CNN models. data_normal can be utilized by conventional ML models.

## Code
This folder contains codes for feature extraction, traditional machine learning models, deep learning models and test modules. 

### Feature Extraction
RMS, VAR, MAX, Peak, Skew, Kurt, Wavelet, Spectral Kurt, Spectral Skewness, Spectral Powder features are extracted from the input time series. 

### Deep Learning Models
Based on Keras, autoencoder and its variants, implementations of LSTM, Bi-directional LSTM and CNN models are provided

### Traditioanl Machine Learning Models
SVR with two kernels (linear and rbf), Random Forest, and Neural Network are provided.

### Main Test
To replicate the results reported in the paper (python 3.6)
```
# if you prefer to use virtual environment
python3.6 -m venv venv
source venv/bin/activate 
pip install -r code/requirements.txt 
# without venv you can run the following codes
python3.6 main_test.py
python3.6 parselog.py
```
Change `python3.6` to your preferable python version. I just check it works with python3.6.


The results will be stored in output.log. In addition, a python notebook file is provided to parse the raw log file for mean and std accuracy computation. 
Due to randomness, we run all of these models five times. 


# BSSAD

### BSSAD: Towards A Novel Bayesian State Space Approach for Anomaly Detection in Multivariate Time Series

## Create Virtual Environment in Anaconda and install Spyder

#### Dependencies (python 3.6) 

```shell
pip install -r requirements.txt
```

#### Run Consolidated Experiment

```shell
python Consolidated_experiment.py
```
Specify File path in data_loader.py
Specify dataset and filterType inside Consolidated_experiment.py

| Dataset             | SWAT   |        | WADI   |        | PUMP   |        | HTTP   |        | ASD    |        |
| ------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|                     | F1     | MCC    | F1     | MCC    | F1     | MCC    | F1     | MCC    | F1     | MCC    |
| Interfusion         | 0.8645 | 0.8572 | 0.8696 | 0.8598 | 0.7204 | 0.7213 |        |        |        |        |
| Anomaly Transformer | 0.7918 | 0.7732 | 0.9329 | 0.9299 | 0.8605 | 0.8528 | 0.4579 | 0.5405 | 0.9183 | 0.9089 |
| BeatGAN             | 0.7549 | 0.7577 | 0.2409 | 0.1989 | 0.4186 | 0.3992 | 0.5888 | 0.6407 | 0.9183 | 0.9089 |
| FGANomaly           | 0.9144 | 0.7206 | 0.5409 | 0.1237 | 0.9587 | 0.7378 | 0.0189 | 0.0486 | 0.866  | 0.1267 |
| NSIBF               | 0.9189 | 0.9107 | 0.901  | 0.8928 | 0.912  | 0.9208 | 0.942  | 0.942  | 0.8889 | 0.8785 |
| NSIBF-PRED          | 0.815  | 0.8084 | 0.7692 | 0.755  | 0.845  | 0.8301 | 0.9765 | 0.9764 | 96,837 | 0.655  |
| BSSAD-EnKF-10       | 0.8842 | 0.8679 | 0.7225 | 0.7007 | 0.9645 | 0.9606 | 0.8195 | 0.8197 | 0.74   | 0.7052 |
| BSSAD-EnKF-20       | 0.8997 | 0.8861 | 0.8061 | 0.7904 | 0.9838 | 0.982  | 0.9714 | 0.9713 | 0.8636 | 0.8445 |
| BSSAD-EnKF-50       | 0.9007 | 0.8895 | 0.7929 | 0.7888 | 0.963  | 0.9594 | 0.6136 | 0.6588 | 0.8395 | 0.8231 |
| BSSAD-PF-500        | 0.9174 | 0.9088 | 0.914  | 0.9084 | 0.8854 | 0.8806 | 0.9842 | 0.9842 | 0.9438 | 0.9369 |
| BSSAD-PF-1000       | 0.9189 | 0.9107 | 0.9417 | 0.9386 | 0.8849 | 0.8799 | 0.9816 | 0.9816 | 0.9231 | 0.9125 |
| BSSAD-PF-2000       | 0.9195 | 0.9113 | 0.9238 | 0.9203 | 0.8849 | 0.8799 | 0.9842 | 0.9844 | 0.9231 | 0.9125 |

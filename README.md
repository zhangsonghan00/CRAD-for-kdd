# CRAD
This repository provides supplemental artifact (source code) for our paper "CRAD: A Contamination-Resilient and Efficient Framework for Time Series Anomaly Detection". 
Please follow the below steps to replicate the results in our paper.

## Datasets

### SWaT dataset
You can get the SWaT dataset by filling out the form at:
https://docs.google.com/forms/d/1GOLYXa7TX0KlayqugUOOPMvbcwSQiGNMOjHuNqKcieA/viewform?edit_requested=true

### PSM dataset
Dataset downloadable at:
https://github.com/eBay/RANSynCoders/tree/main/data

### SMD dataset
Dataset downloadable at: 
https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset

### MSL and SMAP dataset
You can get the MSL and SMAP datasets using:
```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip
rm data.zip
cd data
wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

## Result Reproduction
To run CRAD on a dataset, run the following command:
```bash
python main.py --dataset <dataset> --device <device>
```

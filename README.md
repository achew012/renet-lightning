# RENET on Complex Events : MLP version

This is our Pytorch implementation for RENET on complex evenets: MLP version:

## Requirements

- OS: Ubuntu 18.04 or higher version
- python == 3.7.3 or above
- supported(tested) CUDA versions: 10.2 or 11.3
- Pytorch == 1.8.0 or above
- dgl == 0.8.1

## Code Structure

1. The entry script for training and evaluation is: `main.py`
2. The config file is: `config.yaml`
3. The script for data preprocess and dataloader: `get_history_graph.py` and `dataloader.py`

## How to run the code

Revise dataset path in `config.yaml`

Train RENET on complex events dataset with GPU 0 and info test (for log and tensorboard):

> python main.py --gpu 0 --info test

You can specify the gpu id and addtional infomation in commind line, while you can tune the hyper-parameters by revising the configy file `config.yaml`.

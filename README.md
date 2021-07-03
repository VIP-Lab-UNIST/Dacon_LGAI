# LG AI Challenge: Image enhencement 
This repository hosts the code for LG AI challenge on the [DACON](https://dacon.io/competitions/official/235746/overview/description).

## Folder Structure
  ```
  DACON_LGAI/
  │
  ├── lib/
  │   ├──datasets/ - about dataloading 
  │         ├── dataset.py
  │         ├── transforms.py
  │         └── info/ - anything about data loading goes here
  │             ├── test_img.txt
  │             ├── train_gt.txt
  │             ├── train_img.txt
  │             ├── val_gt.txt
  │             └── val_img.txt
  │   └── utils/ - utility functions
  │         └── util.py
  │
  ├── models/ - network and loss function
  │   ├── network.py
  │   └── loss.py
  │
  ├── .gitignore
  ├── main.py
  ├── test.sh
  └── train.sh
  ```
## Experiments


1. Train
Set the environment in the `train.sh`.
```bash
./train.sh
```
The training results will be stored under `logs/` which is automatically made.

2. Test
```bash
./test.sh
```

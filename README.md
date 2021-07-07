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
- Set the environment in the `train.sh` and `lib/datasets/dataset.py`.
- The training results will be stored under `logs/` which is automatically made.
```bash
./train.sh
```


2. Test
```bash
./test.sh
```

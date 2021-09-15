# LG AI Challenge: Light halation effect remove
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
  ├── run/ 
  │   ├── train.py
  │   └── test.py
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

## Results

1. Result 1 (before, after)

    <figure style="text-align: center;">
    <img src="outputs/Input1.gif" width=30% height=30% style="margin-right: 0; margin-left: 50; "/>
    <img src="outputs/Output1.gif" width=30% height=30% style="margin-right: 50px; margin-left: 0; "/>
    </figure>

2. Result 1 (before, after)

    <figure style="text-align: center;">
    <img src="outputs/Input2.gif" width=30% height=30% style="margin-right: 0px; margin-left: 50; "/>
    <img src="outputs/Output2.gif" width=30% height=30% style="margin-right: 50px; margin-left: 0; "/>
    </figure>

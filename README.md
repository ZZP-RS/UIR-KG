## Environment Requirement
```
The code has been tested running under Python 3.7.10. The required packages are as follows:
* torch == 1.6.0
* numpy == 1.21.4
* pandas == 1.3.5
* scipy == 1.5.2
* tqdm == 4.62.3
* scikit-learn == 1.0.1
```

## Run the Codes
1. run selector.py to generate Long-tail Neighbors
```
python Neighbor_selector.py
```
2. start UIR-KG
```
python main_UIR-KG.py
```


## datasets
We provided two datasets to validate UIR-KG: last-fm and Amazon-book, they are obtained from KGAT. The following table shows the information of two datasets:

|                | Last-FM |  Amazon-book  |
| :------------: | :-----: | :-----: |
|    n_users     |  23566  |  70,679   |
|    n_items     |  48123  |  24,915   |
| n_interactions | 3034796 | 847,733  |
|   n_entities   | 58266  | 88,572  |
|  n_relations   |    9    |   39    |
|   n_triples    | 464567  | 2,557,746 |


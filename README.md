# UIR-KG
This is the code for the paper:
>Zhipeng Zhang, Yuhang Zhang, Anqi Wang, Pinglei Zhou, Yao Zhang, Yonggong Ren. User-oriented interest representation on knowledge graph for long-tail recommendation[C]. Proceedings of the 19th International Conference on Advanced Data Mining and Applications(ADMA), pp.340-355, Shenyang, China, August 21-23, 2023. https://doi.org/10.1007/978-3-031-46674-8_24


## Introduction
UIR-KG is a new neural network recommendation model that utilizes rich semantic information on the knowledge graph to learn users' long tail interest representations. UIR-KG maximizes the recommendation of long tail projects while meeting the mainstream interests of users as much as possible.

If you want to use codes and datasets in your research, please contact the paper authors and cite the following paper as the reference:
```
@inproceedings{UIR-KG,
  author    = {Zhipeng Zhang and
               Yuhang Zhang and
               Anqi Wang and
               Pinglei Zhou and
               Yao Zhang and
               Yonggong Ren
  title     = {User-Oriented Interest Representation on Knowledge Graph for Long-Tail Recommendation},
  booktitle = {{ADMA2023}},
  pages     = {340-355},
  year      = {2023}
}
```


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
We provided two datasets to validate UIR-KG: last-fm and ml-1m, the former obtained from KGAT, and the latter is a version released by Movielens-1m. The following table shows the information of two datasets:

|                | Last-FM |  ml-1m  |
| :------------: | :-----: | :-----: |
|    n_users     |  23566  |  6040   |
|    n_items     |  48123  |  3655   |
| n_interactions | 3034796 | 997579  |
|   n_entities   | 58266  | 398505  |
|  n_relations   |    9    |   57    |
|   n_triples    | 464567  | 3396595 |


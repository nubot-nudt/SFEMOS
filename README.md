## Joint Scene Flow Estimation and Moving Object Segmentation on Rotational LiDAR Data

This repo contains the label generation code for our T-ITS paper: Joint Scene Flow Estimation and Moving Object Segmentation on Rotational LiDAR Data.


## 1. Installation

### Prerequisites

- CUDA 11.3
- Python 3.8
- PyTorch 1.10
- easydict, scipy, sklearn, tqdm, yaml, open3d

## 2. Data Preprocess

To run our experiments on Semantic Kitti dataset, you need to preprocess the original dataset into our scene flow format. All related folders or files are put under `utils/`,

**Running the preprocessing code using:**

```
python utils/generate_gts.py --root_dir $ROOT_DIR$ --save_dir $SAVE_DIR$
```

## Publication
If you use our implementation in your academic work, please cite the corresponding [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10623536):

	@article{chen2021icra,
		author = {Chen, Xieyuanli and Cui, Jiafeng and Liu, Yufei and Zhang, Xianjing and Sun, Jiadai and Ai, Rui and Gu, Weihao and Xu, Jintao and Lu, Huimin},
		title = {Joint Scene Flow Estimation and Moving Object Segmentation on Rotational LiDAR Data},
		booktitle = tits,
		year = {2024},
	        volume={25},
	        number={11},
	        pages={17733-17743},
	}


# RL_3DOD

Official repository of "Towards Robust 3D Object Detection with LiDAR and 4D Radar Fusion in Various Weather Conditions", CVPR2024. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chae_Towards_Robust_3D_Object_Detection_with_LiDAR_and_4D_Radar_CVPR_2024_paper.pdf)

## Requirements
The code has been tested with
- python 3.8
- CUDA 11.1
- pytorch 1.10.1 
- spconv-cu111 2.1.25
- open3d 0.15.2
- opencv-python 4.8.1.78
- matplotlib 3.5.3
- numba 0.53.0
- nms 0.1.6

## Usage

### Train
```
# Stage 1
python models/img_cls/cls_train.py

# Stage 2
python main_train_0.py
```

### Test
```
python main_cond_0.py
```

## Citation

If you find our work helpful, please consider citing our paper:
```
@InProceedings{Chae_2024_CVPR,
    author    = {Chae, Yujeong and Kim, Hyeonseong and Yoon, Kuk-Jin},
    title     = {Towards Robust 3D Object Detection with LiDAR and 4D Radar Fusion in Various Weather Conditions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {15162-15172}
}
```

## Acknowledgements

This work is developed based on the [K-Radar dataset and codebase](https://github.com/kaist-avelab/K-Radar).

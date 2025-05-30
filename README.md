<h2 align="center">Multi-LoRA Fine-Tuned Segment Anything Model for Extraction of Urban Man-Made Objects </h2>

<h5 align="center"> <a href="https://scholar.google.com/citations?user=MDA37NMAAAAJ&hl=zh-CN">Xiaoyan LU</a> and
<a href="https://scholar.google.com/citations?user=SbbCxE8AAAAJ&hl=zh-CN">Qihao WENG</a></h5>


[[`Paper`](https://ieeexplore.ieee.org/abstract/document/10637992)] 


## Multi-LoRA Fine-Tuned SAM Framework

<div align="center">
  <img src="./img/SAM_LoRA.png?raw=true">
</div>

## The Training Set

1. [<b>DeepGlobe Road Training Dataset </b>](https://competitions.codalab.org/competitions/18467#participate-get_data): 4696 samples
2. [<b>SpaceNet Building AOI2 and AOI4 Dataset </b>](https://spacenet.ai/spacenet-buildings-dataset-v2/): 8429 samples

## The Validation Set

1. [<b>DeepGlobe Road Test Dataset </b>](https://competitions.codalab.org/competitions/18467#participate-get_data): 1530 samples
2. [<b>SpaceNet Building AOI3 and AOI5 Dataset </b>](https://spacenet.ai/spacenet-buildings-dataset-v2/): 1148 (Paris) and 1101 (Khartoum) samples
3. [<b>The WHU building (Christchurch) dataset</b>](http://gpcv.whu.edu.cn/data/building_dataset.html): 2416 samples

## Weights Download
1. The trained weights of SAM_Adapter, SAM_LoRA (r=96), and SAM_MLoRA (r=32,n=3) are released at [<b>Baidu Drive</b>](https://pan.baidu.com/s/1_uscV5L8mDt4kihwt1iSqg), Code: MODE
2. Or through [<b>Google Drive</b>](https://drive.google.com/drive/folders/1RJ7nZXZSQ6o-u8YnEq-nmQQ3xpeJ35RO?usp=sharing)

## Road Extraction

SAM_Adapter
```
python train_sam_adapter.py --name='b_adapter_sam'
```

SAM_LoRA (r=96) 
```
python train_sam_adapter.py --name='b_adapter_sam_lora96_96'
```

SAM_MLoRA (r=32,n=3)
```
python train_sam_adapter.py --name='b_adapter_sam_multi_lora'
```

## Building Extraction

SAM_Adapter
```
python train_sam_adapter_build.py --name='b_adapter_sam_sp24'
```

SAM_LoRA (r=96) 
```
python train_sam_adapter_build.py --name='b_adapter_sam_lora96_96_sp24'
```

SAM_MLoRA (r=32,n=3)
```
python train_sam_adapter_build.py --name='b_adapter_sam_multi_lora32_sp24'
```


## Citation
If this code or dataset contributes to your research, please kindly consider citing our paper :)
```
@article{Lu2024MLoRA,
    title = {Multi-LoRA Fine-Tuned Segment Anything Model for Urban Man-Made Object Extraction},
    author = {Xiaoyan LU and Qihao Weng},
    journal = {IEEE Transactions on Geoscience and Remote Sensing},
    volume = {62},
    pages = {1-19},
    year = {2024},
    doi = {https://doi.org/10.1109/TGRS.2024.3435745}
}
```

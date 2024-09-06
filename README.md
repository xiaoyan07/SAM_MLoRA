<h2 align="center">Multi-LoRA Fine-Tuned Segment Anything Model for Extraction of Urban Man-Made Objects </h2>

<h5 align="center"> <a href="https://scholar.google.com/citations?user=MDA37NMAAAAJ&hl=zh-CN">Xiaoyan LU</a> and
<a>Qihao WENG</a></h5>


[[`Paper`](https://ieeexplore.ieee.org/abstract/document/10637992)] 


## Multi-LoRA Fine-Tuned SAM Framework

<div align="center">
  <img src="./img/SAM_LoRA.png?raw=true">
</div>

## The training dataset

1. [<b>DeepGlobe Road Training Dataset </b>](https://competitions.codalab.org/competitions/18467#participate-get_data): 4696 samples
2. [<b>SpaceNet Building AOI2 and AOI4 Dataset </b>](https://spacenet.ai/spacenet-buildings-dataset-v2/): 8429 samples

## The validation set spans across five continents.

1. [<b>DeepGlobe Road Test Dataset </b>](https://competitions.codalab.org/competitions/18467#participate-get_data): 1530 samples
2. [<b>SpaceNet Building AOI3 and AOI5 Dataset </b>](https://spacenet.ai/spacenet-buildings-dataset-v2/): 1148 (Paris) and 1101 (Khartoum) samples
3. [<b>The WHU building (Christchurch) dataset</b>](http://gpcv.whu.edu.cn/data/building_dataset.html): 2416 samples
4. [<b>The other validation dataset</b>]( ): Baidu Drive(Code:)

<div align="center">
  <img src="./img/val_data.png?raw=true">
</div>

If you have difficulty processing this data, feel free to reach out to me at xiaoyan07.lu@polyu.edu.hk for help.


Road Extraction
Model: SAM_Adapter
```
python train_sam_adapter.py --name='b_adapter_sam'
```

Model: SAM_LoRA (r=96) 
```
python train_sam_adapter.py --name='b_adapter_sam_lora96_96'
```

SAM_MLoRA (r=32,n=3)
```
python train_sam_adapter.py --name='b_adapter_sam_multi_lora'
```


Building Extraction
Model: SAM_Adapter
```
python train_sam_adapter_build.py --name='b_adapter_sam_sp24'
```

Model: SAM_LoRA (r=96) 
```
python train_sam_adapter_build.py --name='b_adapter_sam_lora96_96_sp24'
```

SAM_MLoRA (r=32,n=3)
```
python train_sam_adapter_build.py --name='b_adapter_sam_multi_lora32_sp24'
```

## 
The pre-trained SAM_MLoRA is released at [<b>Baidu Drive</b>](), Code:

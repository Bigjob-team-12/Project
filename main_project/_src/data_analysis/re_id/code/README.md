<h1 align="center"> Pet_reID_baseline_pytorch </h1>

### Usage
* [v] Calculating image similarity(cosine distance) 
* [v] Image search at gallery directory or target list
* [v] sorting image similarity
- Query 이미지와 검색 대상 이미지들의 특징을 추출하여 수치화.
- 이미지 특징을 기반으로 image similarity계산. 
- 계산된 수치를 정렬 및 Top N 출력

### Example  
<p align="center"><img src="example.png" width="640"\></p>


### Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.


## Prerequisites
- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- [Optional] apex (for float16) 
- [Optional] [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)


## Getting started
### Installation
@@ -152,17 +49,7 @@ Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .

### Train
Train a model by
@@ -202,23 +89,6 @@ python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batch
`--data_dir` the path of the testing data.

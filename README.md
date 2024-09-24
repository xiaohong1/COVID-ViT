
# COVID-ViT

COVID-VIT: Classification of Covid-19 from CT chest images based on vision transformer models

This code is to response to te MIA-COV19 competition on classification of covid from non-covid chest volumetric CT datasets.

Pre-trained models for ViT and DenseNet can be download from https://drive.google.com/drive/folders/1usJv3vhuKGrVRXWeqWb3PJQ76DB-P6KL?usp=drive_link. 

Both 2D and 3D versions of training and test code are provided. It appears classificaiton based on 2D slices performs better. The final score is subject based, i.e. for a dataset, if more than 25% or more slices are classfied as COVID, then this subject has COVID. Otherwise, the patient in concern will be classified as normal. This threshold (e.g 25%) can be determined from validation stage.

The ViT is heavily based on vit-pytorch at https://github.com/lucidrains/vit-pytorch and is in the form of both notebook and python.

The DenseNet-CT is built upon https://github.com/UCSD-AI4H/COVID-CT. 

More details are at the paper ar Arxiv (https://arxiv.org/) with the following information:
"Xiaohong Gao, Yu Qian, Alice Gao, COVID-VIT: Classification of Covid-19 from CT chest images based on vision transformer models"


# Torch implementation of ELoPE
This repository contains the implementation of the paper [ELoPE: Fine-grained Visual Classification with Efficient Localization, Pooling and Embedding](http://openaccess.thecvf.com/content_WACV_2020/html/Hanselmann_ELoPE_Fine-Grained_Visual_Classification_with_Efficient_Localization_Pooling_and_Embedding_WACV_2020_paper.html)

When using this repository, please cite:
```
@InProceedings{Hanselmann_2020_WACV,
    author = {Hanselmann, Harald and Ney, Hermann},
    title = {ELoPE: Fine-Grained Visual Classification with Efficient Localization, Pooling and Embedding},
    booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
    month = {March},
    year = {2020}
} 
```


### Requirements
##### Software
[Torch](http://torch.ch/docs/getting-started.html)\
[torchx](https://github.com/nicholas-leonard/torchx)

##### Backbone network
Elope expects a backbone network defined as torch-file as input. We use ResNet-50 and ResNet-101 as defined in [fb.resnet.torch](https://github.com/facebookarchive/fb.resnet.torch).

ResNet models pre-trained on ImageNet can be found here: 
[ResNet pre-trained](https://github.com/facebookarchive/fb.resnet.torch/tree/master/pretrained)

**Note**: For some fine-grained datasets there is an overlap between the test-set and the ImageNet training set (e.g. CUB200-2011 and Stanford Cars). In these instances the ResNet models should be pre-trained from scratch while removing the overlapping images (this can be done conveniently using [fb.resnet.torch](https://github.com/facebookarchive/fb.resnet.torch)).


### Setting up the data
To prepare a dataset the following text-files need to be created:
```
train_images.txt
test_images.txt
val_images.txt (optional)
```
Each line in such a file represents one image. It first contains the class represented as integer (starting at 1) and then the image path (can be a relative path).

For the validation set there are three options:
1. No validation set:
```
th setupData.lua --data_path *path_to_dataset*/ --name name --val_mode 0
```
2. Separate validation set from training set (validation set size as percentage of training set):
```
th setupData.lua --data_path *path_to_dataset*/ --name name --val_mode 1 --val_size 0.1
```
3. Pre-defined validation set in val_images.txt:
```
th setupData.lua --data_path *path_to_dataset*/ --name name --val_mode 2
```

### Training a classification model
Train a classification model (no localization module yet):
```
th elope.lua --backbone *path_to_backbone*/backbone.t7 --image_path *path_to_images*/ --save_path *path_to_checkpoints*/ --name model --dataset_path *path_to_dataset*/ --dataset_name name
```

The resulting model is saved in the path *path_to_checkpoints* with the prefix given by *--name* and the important selected parameters. For each checkpoint four files are saved:
```
- network: Contains the model except for the final classification layer
- centers: Contains the learned class centers
- meta: Contains meta information about the training procedure
- fc: Contains the final classificationi layer
```

### Training of the localization module
Assuming the checkpoint of the baseline classification network is called *model_network.t7*, the localization module is trained with:
```
th loc.lua --backbone *path_to_checkpoints*/model_network.t7 --image_path *path_to_images*/ --save_path *path_to_checkpoints*/ --name loc_module --dataset_path *path_to_dataset*/ --dataset_name name
```

The resulting localization module is saved in the path *path_to_checkpoints* with the argument of *--name* as prefix. Only *network* and *meta* are saved.

### Training a classification model with localization
With the localization module trained (assuming the checkpoint is called *loc_module_network.t7*):
```
th elope.lua --backbone *path_to_backbone*/backbone.t7 --image_path *path_to_images*/ --save_path *path_to_checkpoints*/ --name model --dataset_path *path_to_dataset*/ --dataset_name name --localization_module *path_to_checkpoints*/loc_module_network.t7
```

### Finetuning with global weighted K-Max pooling
Use *--KMaxWeight* flag to add weights the K-Max pooling (can also be used when training from scratch).
```
th elope.lua --finetune *path_to_checkpoints*/model_network.t7 --centers *path_to_checkpoints*/model_centers.t7 --localization_module *path_to_checkpoints*/localization_module.t7 --learning_rate 0.001 --learning_rate_decay_step 20 --epochs 40 --KMaxWeight
```

### Notes
We are using the pre-processing, the basic checkpoint writing and reading functions and the shareGradInput function from [fb.resnet.torch](https://github.com/facebookarchive/fb.resnet.torch).


# Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation

Official repository for [```Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation```](https://arxiv.org/abs/2311.05858), accepted to ```WACV 2024```. This repository includes other continual/gradual test-time adaptation methods for classification. We refer to our approach as ```LAW(Layer-wise Auto-Weighting)``` for a simplicity. 

---

### Environmental setting
Environmental setting follows [robustbench](https://github.com/RobustBench/robustbench) and [TTA baselines](https://github.com/mariodoebler/test-time-adaptation).

```bash
conda update conda
conda env create -f environment.yml
conda activate layerwise
```

---

### Datasets

TTA for classification mainly uses corruption datasets such as ```ImageNet-C```. ```CIFAR-10C``` and ```CIFAR-100C``` can be easily obtained from the code itself. In the case of a dataset, when you initially load the dataset from ```robustBench```, it is automatically downloaded. However, if there are any issues with the download you may need to **manually construct the dataset configurations** from the link below.

- ```CIFAR-10C``` : The data is automatically downloaded. Otherwise follow the [Link](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1).
- ```CIFAR-100C``` : The data is automatically downloaded. Otherwise follow the [Link](https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1).
- ```ImageNet-C``` : Download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-C](https://zenodo.org/records/2235448#.Yj2RO_co_mF).

### Datasets directory

```bash
LayerwiseTTA
├── classification
│   ├── data
│   │   ├── CIFAR-10-C
│   │   ├── CIFAR-100-C
│   │   ├── ImageNet-C
...
```

And you need to specify the root folder for all datasets in ```conf.py```. If you configure your dataset directory as above, simply put ```_C.DATA_DIR = "./data"``` would be sufficient.

---

### Pretrained models
All pretrained models are provided in [robustbench](https://github.com/RobustBench/robustbench) or ```torchvision``` or ```timm``` so there is no need to download them manually. In the case of ResNet-50 used for additional experiments in supplementary, I used pretrained models in [TTT++](https://github.com/vita-epfl/ttt-plus-plus/tree/main/cifar).
I have referenced and modified the provided github code and added them to the ```get_model.py```. Additionally, I have configured the script to automatically download the ```.ckpt``` files from the Google Drive links, ensuring seamless downloading if the corresponding ```.ckpt``` file is not already available locally.

Furthermore in online test-time adaptation, since performance evaluation is conducted concurrently with optimization, we do not provide separately trained pretrained checkpoints(```.ckpt```) in classification task.

---

### Experimental Settings
- ```reset_each_shift``` : Initialize the model state to the source-pretrained parameters after each adaptation to a new domain.
- ```continual``` : Optimize the model for continuously shifting domains without prior knowledge of when a domain shift occurs. Therefore, there is no resetting process.
- ```gradual``` : Optimize the model for gradually increasing or decreasing domain shifts without prior knowledge of when a domain shift occurs. Therefore, there is no resetting process.

---

### Run Test-Time Adaptation
We provide config files for all experiments and methods from the [baselines](https://github.com/mariodoebler/test-time-adaptation). Simply run with the corresponding [config files](./classification/cfgs).

**Continual test-time adaptation**

```bash
python test_time.py --cfg cfgs/cifar10_c/law.yaml SETTING continual
```

**Gradual test-time adaptation**

```bash
python test_time.py --cfg cfgs/cifar10_c/law.yaml SETTING gradual
```

---

### About downloading issue in ```robustbench```
I found a downloading issue in ```robustbench/zenodo_download.py``` when uploading the final version of official code. Issue occurs because of the modification in meta-data from url requests. Therefore I replaced ```download_file``` in [original version](https://github.com/RobustBench/robustbench/blob/master/robustbench/zenodo_download.py) with ```wget```.

# Citation

<details>
<summary>Open</summary>
  
```bibtex
@inproceedings{park2024layer,
  title={Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation},
  author={Park, Junyoung and Kim, Jin and Kwon, Hyeongjun and Yoon, Ilhoon and Sohn, Kwanghoon},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1414--1423},
  year={2024}
}
```

</details>


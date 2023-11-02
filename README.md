# Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation

<p align="center">
  <img src="https://github.com/junia3/LayerwiseTTA/assets/79881119/783a7a75-41ed-414d-b51e-ebccf8e52616">
</p>  

> Code will be uploaded soon.

---

## Environmental setting
Environmental setting follows [robustbench](https://github.com/RobustBench/robustbench) and [TTA baselines](https://github.com/mariodoebler/test-time-adaptation).

```bash
conda update conda
conda env create -f environment.yml
conda activate tta
```

---

## Datasets

TTA for classification mainly uses corruption datasets such as ```ImageNet-C```. ```CIFAR-10C``` and ```CIFAR-100C``` can be easily obtained from the code itself, however ```ImageNet-C``` download link in baseline might be corrupted so it can be an issue.

- ```CIFAR-10C``` : [Link](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1)
- ```CIFAR-100C``` : [Link](https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1)
- ```ImageNet-C``` : [Link](https://zenodo.org/records/2235448)

## Datasets directory

```bash
LayerwiseTTA
├── classification
│   ├── data
│   │   ├── CIFAR-10-C
│   │   ├── CIFAR-100-C
│   │   ├── ImageNet-C
...
```


---

## Pretrained models
All pretrained models are provided in [robustbench](https://github.com/RobustBench/robustbench) or ```Torchvision``` or ```Timm``` so there is no need to download them manually. In the case of ResNet-50 used for additional experiments in supplementary, I used pretrained models in [TTT++](https://github.com/vita-epfl/ttt-plus-plus/tree/main/cifar). Furthermore in online test-time adaptation, since performance evaluation is conducted concurrently with optimization, we do not provide separately trained pretrained checkpoints(```.ckpt```) in classification task.

---

## Run test-time adaptation

### Continual test-time adaptation

- Example : source/CIFAR-10C
```bash
python test_time.py --cfg cfgs/cifar10_c/source.yaml SETTING continual RNG_SEED 0
```

- Example : [TENT](https://arxiv.org/abs/2006.10726)/CIFAR-10C
```bash
python test_time.py --cfg cfgs/cifar10_c/tent.yaml SETTING continual RNG_SEED 0
```

### Gradual test-time adaptation
- Example : source/CIFAR-10C
```bash
python test_time.py --cfg cfgs/cifar10_c/source.yaml SETTING gradual RNG_SEED 0
```

- Example : [TENT](https://arxiv.org/abs/2006.10726)/CIFAR-10C
```bash
python test_time.py --cfg cfgs/cifar10_c/tent.yaml SETTING gradual RNG_SEED 0
```

---

# Citation

```bash
---
```

---

# Contact
If you have any issue with code, feel free to contact ```jun_yonsei@yonsei.ac.kr```.

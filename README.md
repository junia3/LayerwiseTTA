# Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation

<p align="center">
  <img src="https://github.com/junia3/LayerwiseTTA/assets/79881119/4b72124a-7e99-4799-8fdf-088195fe382c" width="800">
</p>

> Official Code Implementation for WACV-24 accepted paper.

---

## Environmental setting
Overall environmental setting follows [robustbench](https://github.com/RobustBench/robustbench) and [TTA baselines](https://github.com/mariodoebler/test-time-adaptation).

```bash
conda update conda
conda env create -f environment.yml
conda activate tta
```

---

## Datasets

TTA for classification mainly uses corruption datasets such as ImageNet-C. CIFAR-10C and CIFAR-100C can be easily obtained from the code itself, however ImageNet-C download link in baseline might be corrupted so it can be an issue.

- CIFAR-10C : [Link](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1)
- CIFAR-100C : [Link](https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1)
- ImageNet-C : [Link](https://zenodo.org/records/2235448)

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
All pretrained models are provided in [robustbench](https://github.com/RobustBench/robustbench). In the case of ResNet-50 used for additional experiments in supplementary, I used pretrained models in [TTT++](https://github.com/vita-epfl/ttt-plus-plus/tree/main/cifar).

---

## Run test-time adaptation

### Continual test-time adaptation

TBD

### Gradual test-time adaptation

TBD

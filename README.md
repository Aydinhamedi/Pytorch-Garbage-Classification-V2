# Garbage Classification V2 with PyTorch

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white"/>  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Pytorch project for garbage classification using the **EfficientNet-B6** model to achieve a **95.78%** accuracy on the test set.

> [!IMPORTANT]
> This project is a new version of the original project, which can be found [here](https://github.com/Aydinhamedi/Pytorch-Garbage-Classification) **but with a significantly improved training process + code and a different dataset**.

## 😉 Bonus
This project is not hard coded for this specific dataset, so it can be used for any image classification task and it has all the necessary tools to train a model from scratch and more. **(I will make a pytorch classification template soon)**

## 📦 Release
> ### Newest release 📃
> #### [Go to newest release](https://github.com/Aydinhamedi/Pytorch-Garbage-Classification-V2/releases/latest)

## 📂 Dataset

The dataset used for this project is the [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) from Kaggle. It contains images of garbage, divided into six categories.
### Data Structure
~~~
├───Database
│   └───Data # Put all the folders with images here
#       Example ⬎
│       ├───battery
│       ├───biological
│       ├───brown-glass
│       ...
│       └───white-glass
~~~

## 🧪 Model

I used the **EfficientNet-B6** model for this project. **EfficientNet-B6** is a convolutional neural network that is pretrained on the ImageNet dataset. It is known for its efficiency and high performance on a variety of image classification tasks. [Original paper](https://arxiv.org/abs/1905.11946)

## 🔰 Installation

To run the code in this repository, you will need to install the required libraries. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

> [!WARNING]
> The requirements are auto generated by `pipreqs` and may not contain all the necessary dependencies. like hidden ones like Tensorboard.

## 🚀 Usage

The main code for this project is in a Jupyter notebook named `Main.ipynb`. To run the notebook, use the following command:

```bash
jupyter notebook Main.ipynb
```

## 📃 Results

| Metric                           |     Value |
|----------------------------------|-----------|
| Loss                             | 0.0330466 |
| F1 Score (macro)                 | 0.95472   |
| Precision (macro)                | 0.952111  |
| Recall (macro)                   | 0.957959  |
| AUROC                            | 0.993324  |
| Accuracy                         | 0.957839  |
| Cohen's Kappa                    | 0.948292  |
| Matthews Correlation Coefficient | 0.948374  |

![alt text](<doc/Best results/cfm.png>)
![alt text](<doc/Best results/norm_cfm.png>)


## 📚 License
<pre>
 Copyright (c) 2024 Aydin Hamedi
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
</pre>
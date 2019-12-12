# A-Visual-Attention-Grounding-Neural-Model
## 1. Overview
This code repository contains the implementation of our research project: _[A Visual Attention Grounding Neural Model for Multimodal Machine Translation](https://arxiv.org/abs/1808.08266)._

We introduce a novel multimodal machine translation model that utilizes parallel visual and texture information. Our model jointly optimizes learning of a shared visual-language embedding and translating languages. It does this with the aid of a visual attention grounding mechanism which links the visual semantics in the image with the corresponding textual se-
mantics.

![An Overview of the Visual Attention NMT](https://github.com/zmykevin/A-Visual-Attention-Grounding-Neural-Model/blob/master/AGV-NMT.jpg)

## 2. Prerequisite
The code is successfully tested on Ubuntu 16.04 with NVIDIA GPUs and the following things are expected as prerequisite:
1. Python 3.6
2. [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
3. CUDNN 7.1.4
4. [Conda](https://conda.io/miniconda.html)

While we havn't tested this code with other OS system, we expect it can be runned on any Linux Based OS with a minor adjustment. 

One more thing, to properly use the METEOR score to evaluate the model's performance, you will need to download a set of METEOR paraphrase files and store it under the repository of machine_translation_vision/meteor/data. These paraphrase files are available to be download from [here](https://github.com/cmu-mtlab/meteor/tree/master/data).
## 3. How to run the code?
Once you have meet with all the preresuisites, you can start to run our code. The first step is to reconstruct the software environment for the code. We provide a Conda virtual environment file to help users to reconstruct the same software environment as the one we used to run the code. Using the following command to create the same software environment:
```
conda env create -f machine_translation_vision.yml
```
Then a virtual environment named "machine_translation_vision" will be created. To lauch this environment, simply run:
```
source activate machine_translation_vision
```
Once you have launched the virtual environment, you can start to run our code. To train a VAG-NMT, you will need to run the file "nmt_multimodal_beam_DE.py" or "nmt_multimodal_beam_FR.py", depending on the languages you plan to work with. If you want to build a English to German translation model, then you can run:
```
  python nmt_multimodal_beam_DE.py --data_path path/to/data --trained_model_path /path/to/save/model --sr en --tg de
```
You need to define at least four things in order to run this code: the directory for the dataset, the directory to save the trained model, the source language, and the target language. The languages that our model can work with include: English=> "en", German->"de" and French->“fr”.

To test a trained model on a test dataset, you can run `test_multimodal.py` and `test_monomodal.py` respectively to evaluate the trained multimodal NMT and trained text-only NMT. You need to modify the parameters in the block of "User Defined Area" according to your own situation. The way to define each parameter is the same as that defined in the training process.

We have the Preprocessed Multi30K Dataset available in this [link](https://drive.google.com/drive/folders/1G645SexvhMsLPJhPAPBjc4FnNF7v3N6w?usp=sharing), which can be downloaded to train the model.

## 4. IKEA Dataset
The collected product description multimodal machine translation benchmark cralwed from IKEA website is stored under the github repo [here](https://github.com/sampalomad/IKEA-Dataset)


__Last Updated: 9/17/2018__
### To be Added 
1. Introduction on how to prepare your own dataset to train the model
2. Create the script to generate full automatic evaluation results.

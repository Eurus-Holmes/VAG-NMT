# VAG-NMT

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)  

> Pytorch implementation for the paper "[A Visual Attention Grounding Neural Model for Multimodal Machine Translation](https://arxiv.org/abs/1808.08266)".


## Datasets

  - [Multi30K](https://github.com/multi30k/dataset): The Preprocessed Multi30K Dataset available in this [link](https://drive.google.com/drive/folders/1G645SexvhMsLPJhPAPBjc4FnNF7v3N6w?usp=sharing), which can be downloaded to train the model.

  - [IKEA](https://github.com/sampalomad/IKEA-Dataset): The collected product description multimodal machine translation benchmark cralwed from IKEA website is stored under the github repo [here](https://github.com/sampalomad/IKEA-Dataset)


## Prerequisite

```bash
pip install -r requirements.txt
```

One more thing, to properly use the METEOR score to evaluate the model's performance, you will need to download a set of METEOR paraphrase files and store it under the repository of `machine_translation_vision/meteor/data`. 

These paraphrase files are available to be download from [here](https://github.com/cmu-mtlab/meteor/tree/master/data).

## Run

To train a VAG-NMT, you will need to run the file "nmt_multimodal_beam_DE.py" or "nmt_multimodal_beam_FR.py", depending on the languages you plan to work with. If you want to build a English to German translation model, then you can run:

```bash
  python nmt_multimodal_beam_DE.py --data_path ./path/to/data --trained_model_path ./path/to/save/model --sr en --tg de
```

You need to define at least four things in order to run this code: the directory for the dataset, the directory to save the trained model, the source language, and the target language. The languages that our model can work with include: English=> "en", German->"de" and French->“fr”.

To test a trained model on a test dataset, you can run `test_multimodal.py` and `test_monomodal.py` respectively to evaluate the trained multimodal NMT and trained text-only NMT. You need to modify the parameters in the block of "User Defined Area" according to your own situation. The way to define each parameter is the same as that defined in the training process.



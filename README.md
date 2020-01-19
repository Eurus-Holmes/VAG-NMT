# VAG-NMT

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)  

> Pytorch implementation for the paper "[A Visual Attention Grounding Neural Model for Multimodal Machine Translation](https://arxiv.org/abs/1808.08266)". 

> Original author's implementation is [here](https://github.com/zmykevin/A-Visual-Attention-Grounding-Neural-Model).


## Datasets

  - [Multi30K](https://github.com/multi30k/dataset): The Preprocessed Multi30K Dataset available in this [link](https://drive.google.com/drive/folders/1G645SexvhMsLPJhPAPBjc4FnNF7v3N6w?usp=sharing), which can be downloaded to train the model.

  - [IKEA](https://github.com/sampalomad/IKEA-Dataset): The collected product description multimodal machine translation benchmark cralwed from IKEA website is stored under the github repo [here](https://github.com/sampalomad/IKEA-Dataset)


## Prerequisite

```bash
pip install -r requirements.txt
```

One more thing, to properly use the METEOR score to evaluate the model's performance, you will need to download a set of METEOR paraphrase files and store it under the repository of `machine_translation_vision/meteor/data`. 

These paraphrase files are available to be download from [here](https://github.com/cmu-mtlab/meteor/tree/master/data).

## Run the Code

### Train

To train a VAG-NMT, you will need to run the file "nmt_multimodal_beam_DE.py" or "nmt_multimodal_beam_FR.py", depending on the languages you plan to work with. If you want to build a English to German translation model, then you can run:

```bash
  python nmt_multimodal_beam_DE.py --data_path ./path/to/data --trained_model_path ./path/to/save/model --sr en --tg de
```

You need to define at least four things in order to run this code: the directory for the dataset, the directory to save the trained model, the source language, and the target language. The languages that our model can work with include: English=> "en", German->"de" and French->“fr”.

### Test

To test a trained model on a test dataset, you can run `test_multimodal.py` and `test_monomodal.py` respectively to evaluate the trained multimodal NMT and trained text-only NMT. You need to modify the parameters in the block of "User Defined Area" according to your own situation. The way to define each parameter is the same as that defined in the training process.

```bash
  python test_multimodal.py
```


### Results

```bash
nohup python nmt_multimodal_beam_DE.py --data_path ./path/to/data --trained_model_path ./path/to/save/model --sr en --tg de &

nohup python nmt_multimodal_beam_FR.py --data_path ./path/to/data --trained_model_path ./path/to/save/model --sr en --tg fr &
```

Output: [nohup.out](https://github.com/Eurus-Holmes/VAG-NMT/blob/master/nohup.out)


#### En-DE
```
Text-only 


Best Loss so far is: 2.2789595127105713
Best BLEU so far is: 0.37752992649401007
Best METEOR so far is: 0.5666695663662518
Training is done.
Evalute the Test Result
Test BLEU score from the best BLEU model: 0.28795071418055507
Test METEOR score from the best BLEU model: 0.5041343275322484

Test BLEU score from the best LOSS model: 0.2908121238955742
Test METEOR score from the best LOSS model: 0.5102397970452623

Test BLEU score from the best METEOR model: 0.28746455565942547
Test METEOR score from the best METEOR model: 0.5116179300972875

Multimodal

Best Loss so far is: 2.324632949062756
Best BLEU so far is: 0.4072734023593675
Best METEOR so far is: 0.5805528322085509
Training is done.
Evalute the Test Result
Test BLEU score from the best BLEU model: 0.3156321668098113
Test METEOR score from the best BLEU model: 0.5215313737631495

Test BLEU score from the best LOSS model: 0.3125071718114073
Test METEOR score from the best LOSS model: 0.5200889548226774

Test BLEU score from the best METEOR model: 0.3156321668098113
Test METEOR score from the best METEOR model: 0.5215313737631495

```


## Citation

```tex
@article{zhou2018visual,
  title={A visual attention grounding neural model for multimodal machine translation},
  author={Zhou, Mingyang and Cheng, Runxiang and Lee, Yong Jae and Yu, Zhou},
  journal={arXiv preprint arXiv:1808.08266},
  year={2018}
}
```


# Pytorch implementation of QANet
## Model description
Model for reading comprehension style question answering based on [QANet paper](https://arxiv.org/abs/1804.09541)
## Requirements
* Python 3.5 or higher
* PyTorch 1.0.1 or higher
* Numpy 1.16.2 or higher
* Spacy 2.1.2 or higher
* tensorboardX
## Usage
### Preproccessing
First of all, it's neccessary to create folder with dataset and glove embeddings in it, e.g. `./dataset/train-v1.1.json`,
`./dataset/dev-v1.1.json` and `./dataset/glove.840B.300d.txt`.  Preprocessor builds vocabulary and converts text into
tensors. 

For preproccess use:

```python preproccess.py```
### Training
For train model use:

```python train.py ```

After training model is saving into `./models_dumps/` directory.

You can tune model with lots arguments available in model. Default configuration is used, but you can change it in 
`./config/train_config.yml`.

### Testing
For testing use:

```python train.py --mode=test```

**See help of each module for more information and available arguments.**

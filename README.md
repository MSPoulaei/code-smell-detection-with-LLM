# Code Smell Detection With LLM

## Task

This project is a multi-label classification problem in which the code smells of input code are detected using a Large Language Model

## Dataset

## Classes

## Code

There are 3 main source code files for this project:

1. PrepareDataset: this code fetches raw dataset and extract the code and labels and save as a csv file
1. TrainTest: there are two methods to train and test a Large Language Model:
    - finetune the model without quantization
    - finetune the model with quantization (QLora method)
1. Inference: In this method input code and a prompt is fed into the model and the code extracts the code smells from the output of the model

Therefore, we have two methods discussed in this project:

- Multi-Label Classification
- Prompt Engineering

## Thesis

The thesis of project is written in latex.
you can build the pdf file using these commands:

```sh
xelatex main
bibtex main
xelatex main
```

After that the main.pdf file is built.

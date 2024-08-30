# Code Smell Detection With LLM

## Contributors

### Authors

- Mohammad Sadegh Poulaei
- Sayin Ala

### Supervisor

- Dr.Saeid Parsa

## Abstract

This research proposes a model for detecting code smells using large language models. Code smells refer to concepts and features in programming code that may indicate deeper issues in software design and implementation. These issues can lead to reduced code quality and increased complexity in maintenance and development. The proposed method leverages large language models trained on labeled datasets, capable of detecting 28 different types of code smells. The model is developed using advanced deep learning techniques and architectures, such as Transformers. Evaluation results show that the proposed model significantly improves the accuracy of code smell detection and can serve as an effective tool for software developers. The research also addresses the challenges in training and optimizing large language models and provides solutions to enhance the model's performance.

## 📚 Overview

This project addresses a multi-label classification problem where code smells in input code are detected using a Large Language Model (LLM).

## Dataset

The dataset used for this project consists of code snippets labeled with various code smells. The dataset should be formatted to facilitate both training and evaluation of the model:

- Source Code: The raw dataset contains source code snippets extracted from various projects.
- Labels: Each code snippet is annotated with one or more code smell labels corresponding to the identified issues.

## Classes

| Code Smell Class                            | Description                                                                 |
|---------------------------------------------|-----------------------------------------------------------------------------|
| Missing Hierarchy                           | Lack of a well-defined inheritance structure where one is needed.           |
| Long Parameter List                         | Methods or functions with too many parameters, making them hard to use.     |
| Unnecessary Abstraction                     | Abstractions that add no clear benefit or are not justified in the context. |
| Imperative Abstraction                      | Abstractions that enforce imperative logic, reducing flexibility.           |
| Empty catch clause                          | Catch blocks that fail to handle exceptions, leading to silent failures.    |
| Deficient Encapsulation                     | Poorly protected data, exposing internal details unnecessarily.             |
| Long Identifier                             | Excessively lengthy names for variables, methods, or classes.               |
| Multifaceted Abstraction                    | Classes or interfaces with too many responsibilities.                      |
| Wide Hierarchy                              | Inheritance trees that are too broad, making navigation difficult.          |
| Complex Conditional                         | Conditional expressions that are hard to read or understand.                |
| Rebellious Hierarchy                        | Inheritance structures that violate logical or expected relationships.      |
| Magic Number                                | Use of literal numbers without explanation, reducing code clarity.          |
| Missing default                             | Switch statements lacking a default case, risking unhandled scenarios.      |
| Long Method                                 | Methods that are too long, making them hard to understand or maintain.      |
| Broken Modularization                       | Modules that do not function as intended or lack cohesion.                  |
| Broken Hierarchy                            | Disorganized or illogical inheritance structures.                           |
| Unutilized Abstraction                      | Abstractions present but not effectively utilized in the code.              |
| Long Statement                              | Single statements that span multiple lines, reducing readability.           |
| Cyclic-Dependent Modularization             | Modules that depend on each other circularly, complicating dependencies.    |
| Multipath Hierarchy                         | Inheritance paths that are too complex, creating confusion.                 |
| Deep Hierarchy                              | Inheritance trees that are too deep, making it hard to follow the logic.    |
| Hub-like Modularization                     | Modules that centralize too many responsibilities, becoming bottlenecks.    |
| Insufficient Modularization                 | Lack of modularity, resulting in large, monolithic blocks of code.          |
| Cyclic Hierarchy                            | Inheritance loops causing infinite recursion or logical errors.             |
| Unexploited Encapsulation                   | Encapsulation used but not fully leveraged to protect data or behavior.     |
| Abstract Function Call From Constructor     | Calling abstract methods within a constructor, leading to potential issues. |
| Complex Method                              | Methods with high complexity, often due to intricate logic or many branches.|

## Code

There are 3 main source code files for this project:

1. PrepareDataset: this code fetches raw dataset and extract the code and labels and save as a csv file
1. TrainTest: there are two methods to train and test a Large Language Model:
    - finetune the model without quantization
    - finetune the model with quantization (QLora method)
1. Inference: In this method input code and a prompt is fed into the model and the code extracts the code smells from the output of the model

## Methods

### Multi-Label Classification

- Objective: Train the LLM to identify multiple code smells present in a single code snippet.
- Approach: Use labeled dataset to train the model so it can predict multiple labels for each input.

### Prompt Engineering

- Objective: Improve the model's performance by crafting effective prompts that guide the LLM to deliver accurate outputs
- Approach: Experiment with different prompt formats and structures to determine which are most effective at eliciting correct responses from the model.

## 📊 Evaluation Results

The results of the code smell detection evaluation on various models are shown in Table 1. These models were trained for one epoch on 10,000 data points with a maximum input length of 3,000 tokens, using AdamW as the optimizer and binary cross-entropy as the loss function.

| Model             | Precision | Recall  | Accuracy | F1      |
|-------------------|-----------|---------|----------|---------|
| LLaMA 3.1-8B      | 36.46%   | 64.79% | 75.78%  | 30.28% |
| gemma 2-9B        | 35.04%   | 65.15% | 73.21%  | 29.46% |
| LLaMA 3-8B        | 35.54%   | 61.92% | 77.02%  | 28.57% |
| LLaMA 2-7B        | 34.06%   | 64.39% | 72.18%  | 28.57% |
| mistral 7B        | 34.87%   | 60.41% | 76.71%  | 27.82% |
| phi 3.5 mini 3.8B | 35.07%   | 60.01% | 75.87%  | 28.38% |
| smoLM 2B          | 35.10%   | 59.70% | 73.25%  | 29.53% |
| GPT2-large        | 31.83%   | 61.03% | 72.68%  | 25.38% |

## Thesis

The thesis for this project is written in LaTeX. To build the PDF file, use the following commands:

```sh
xelatex main
bibtex main
xelatex main
xelatex main
```

This sequence of commands compiles the LaTeX document and generates the bibliography, resulting in a complete `main.pdf` file.

## Future Work

- Dataset Expansion: Increase the variety and size of the dataset to improve model accuracy and generalization.
- Model Optimization: Explore additional fine-tuning techniques and model architectures to enhance performance.
- Real-time Detection: Develop a plugin for code editors to provide real-time code smell detection and suggestions for improvement.

## Conclusion

This project leverages large language models to automate the detection of code smells in software projects. By applying advanced techniques in multi-label classification and prompt engineering, the project aims to provide a robust tool for software quality assurance and improvement.

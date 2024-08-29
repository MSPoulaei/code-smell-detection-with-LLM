# Code Smell Detection With LLM

## Contributors

### Authors

- Mohammad Sadegh Poulaei
- Sayin Ala

### Supervisor

- Dr.Saeid Parsa

## Task

<!-- This project is a multi-label classification problem in which the code smells of input code are detected using a Large Language Model -->
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

# fusion-opt

<h1> Introduction </h1>
Bayesian Optimization (BO) has proven effective in optimizing costly black-box functions, yet users’ lack of confidence hinders its adoption in real-world applications. Recent advancements have proposed human-AI collaborative approaches to enhance the trustworthiness of BO, integrating human expertise into the optimization rocess. 
However, current methods primarily focus on single-task scenarios, which start the optimization process from scratch when tackling a new function, neglecting the opportunity to leverage knowledge from related functions. We propose to tackle this issue by leveraging Meta-Learning and introducing Meta Bayesian Optimization with Human Feedback (MBO-HF), an explainable human-in-the-loop framework for Meta Bayesian Optimization (Meta-BO). MBO-HF utilizes a Transformer Neural Process (TNP) to construct an adaptable surrogate model. This surrogate model, coupled with MBO-HF’s novel acquisition function (AF), suggests candidate points to human users. Additionally, MBO-HF integratesH an explainable framework, leveraging Shapley values to provide explanations about the candidates, helping human users make informed decisions regarding the next best point to evaluate. We demonstrate that MBO-HF surpasses the state-of-the-art on human-AI collaborative BO in experiments on five different hyperparameter
optimization benchmarks and a real-world task of battery design. MBO-HF can also be used for optimizations used for Inertial Confinement Fusion.

## Documentation and Installation
Refer to the [docs](https://hewlettpackard.github.io/fusion-opt/) for documentation of the Fusion-opt.

<h1>Quick Start Guide</h1>
<h2>Prerequisites</h2>
Python3.9

Conda

## Installation
First, download the repository. If using HTML, execute:
```bash
$ git clone https://github.com/HewlettPackard/fusion-opt.git
```

If using SSH, execute:
```bash
$ git clone git@github.com:HewlettPackard/fusion-opt.git
```


Create a conda environment and install dependencies:
```bash
$ conda create -n mbohf python=3.9
$ conda activate mbohf
$ pip install -r requirements.txt
```


## Usage

<h3>Training  </h3> Run the appropriate script for the target function 16d_xgboost.py, 6d_ranger.py, 6d_rpart.py, 8d_svm.py, 9d_ranger.py
<h3>Test</h3>  Run synthetic_meta_ptest.py








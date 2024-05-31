# fusion-opt

<h1> Introduction </h1>
Bayesian Optimization (BO) has proven effective in optimizing costly black-box1
functions, yet users’ lack of confidence hinders its adoption in real-world applica-2
tions. Recent advancements have proposed human-AI collaborative approaches to3
enhance the trustworthiness of BO, integrating human expertise into the optimiza-4
tion process. However, current methods primarily focus on single-task scenarios,5
which start the optimization process from scratch when tackling a new function,6
neglecting the opportunity to leverage knowledge from related functions. In this7
paper, we propose to tackle this issue by leveraging Meta-Learning and introducing8
Meta Bayesian Optimization with Human Feedback (MBO-HF), an explainable9
human-in-the-loop framework for Meta Bayesian Optimization (Meta-BO). MBO-10
HF utilizes a Transformer Neural Process (TNP) to construct an adaptable surrogate11
model. This surrogate model, coupled with MBO-HF’s novel acquisition function12
(AF), suggests candidate points to human users. Additionally, MBO-HF integrates13
an explainable framework, leveraging Shapley values to provide explanations about14
the candidates, helping human users make informed decisions regarding the next15
best point to evaluate. We demonstrate that MBO-HF surpasses the state-of-the-art16
on human-AI collaborative BO in experiments on five different hyperparameter17
optimization benchmarks and a real-world task of battery design.

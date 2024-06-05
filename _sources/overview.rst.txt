========================================
Overview
========================================


- Using state of the art Neural Acquistion process(NAP) as a surrogate for  black box functions
- Using preference model trained on experts' feedback to enable scoring preference of the points to query next 
- Suggesting multiple candidate points to human(expert), from which one to be selected as next point to be queried 
- A newly designed Acquistion Function combines the preference model and the NAP to suggest one candidate point
- The other candidate points are chosen by desired well-known statistical or Monte-carlo based Acquistion functions such as EI, MES, ...
- Modular design, i.e users can utilize pre-defined modules for preference models, or build their own
- Explainability, the framework explains each of the candidate points so that human in the loop can make better decsion for selecting the next point


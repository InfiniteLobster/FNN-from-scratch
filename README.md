# 02456 Deep Learning DTU project 
group 70:
- Szymon Cholewiński (s253711)
- Thorvaldur Ludviksson (s242975)
- Ismael Tekaya (S251701)
- Andrés Hlynsson (s242978)

## Project topic: 26. Implementing a Neural Network from Scratch with NumPy: Training, Optimization, and Experiment Tracking with Weights & Biases (WandB)
Project goals:
- Implementing: (showcased in: TrainingExamples.ipynb and some parts tested in: Testing.ipynb)
  - Forward pass -> matrix multiplications + activation functions : **implemented** (FNN.py, ActivFunctions.py, SuppFunctions.py)
  - Loss computation -> MSE or cross-entropy with L2 regularization : **implemented** (LossFunctions.py)
  - Backward pass -> manual derivative calculation and weight updates : **implemented** (FNN.py, SuppFunctions.py)
  - Training loop -> mini-batch gradient descent : **implemented** (OptimizerFunctions.py, SuppFunctions.py)
  - Evaluation -> compute accuracy, loss curves, and confusion matrices : **implemented** (TestingFunctions.py)
- WandBi:
  - Learning curves (train_loss, val_loss, accuracy, val_acc): **logging implemented** (SweepFunctions.py)
  - Parameter histograms and gradient norms: **logging implemented** (SweepFunctions.py)
  - Hyperparameter sweeps (random or Bayesian) across architectures and optimizers: **done** (via SweepExample.ipynb, logged to WandBi team)
  - Summary reports comparing activation functions and initializations: **done** (WandBi team site)
## WandBi team link: https://wandb.ai/DL_project_Group_70/reports
## Project description
In this project flexible feed-forward neural network was implemented. From design perspective there is no limit on number of neurons or layers that can be used. 
It is so flexible in design, that each neuron in layer can have different activation function. It can be used for both regression and classification (binary and multi-class) tasks.
Following activation functions were implemented: identity, sigmoid, tanh, ReLu, leaky ReLu and softmax. As for loss function both Mean Square Error and Cross-Entropy are implemented.
Furthermore special Softmax+Cross-Entropy option is availible, for cases when whole output layer is softmax and used loss function is softmax. It improves speed of calculations and results of training by calculating derivative with regards to logits.
Normal option for softmax and Cross-Entropy (standard backward, i.e. calculating Jacobian etc.) is also possible, so it is option, not forced version.
## Implementation idea
xxx
## Project files description:
- Testing.ipynb:
- TrainingExamples.ipynb:
- SweepExample.ipynb:
- FNN.py:
- Layer.py:
- Neuron.py:
- InitFunctions.py:
- ActivFunctions.py:
- LossFunctions.py:
- ErrorClasses.py:
- OptimizersFunctions.py:
- TestingFunctions.py:
- SuppFunctions.py:
- SweepFunctions.py:
## References
[]

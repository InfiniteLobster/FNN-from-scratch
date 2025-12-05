# 02456 Deep Learning DTU project 
group 70:
- Szymon Cholewiński (s253711)
- Thorvaldur Ludviksson (s242975)
- Ismael Tekaya (S251701)
- Andrés Hlynsson (s242978)

## Project topic: 26. Implementing a Neural Network from Scratch with NumPy: Training, Optimization, and Experiment Tracking with Weights & Biases (WandB)
Project goals:
- Implementing:
  - Forward pass: matrix multiplications + activation functions : implemented (FNN.py, ActivFunctions.py, SuppFunctions.py)
  - Loss computation: MSE or cross-entropy with L2 regularization : implemented (LossFunctions.py)
  - Backward pass: manual derivative calculation and weight updates : implemented (FNN.py, ActivFunctions.py, LossFunctions.py, SuppFunctions.py)
  - Training loop: mini-batch gradient descent : implemented (OptimizerFunctions.py, SuppFunctions.py)
  - Evaluation: compute accuracy, loss curves, and confusion matrices : implemented
- WandBi:
  - Learning curves (train_loss, val_loss, accuracy, val_acc):
  - Parameter histograms and gradient norms
  - Hyperparameter sweeps (random or Bayesian) across architectures and optimizers
  - Summary reports comparing activation functions and initializations
##WandBi team link: https://wandb.ai/DL_project_Group_70/projects

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
### General
In this project flexible feed-forward neural network was implemented by using NumPy not any ready Machine Learning/Deep Learning library like PyTorch or Tenser Flow. From design perspective there is no limit on number of neurons or layers that can be used. 
It is so flexible in design, that each neuron in layer can have different activation function. It can be used for both regression and classification (binary and multi-class) tasks.
For training both tabular and images (as long as it is collapsed for pixel vector) data can be used. This functionalities were tested on 3 datasets: Breast Cancer Wisconsin (Diagnostic) [1], MNIST [2] and CIFAR-10 [3].
On first dataset both regression and binary classification was tested (models were trained without issues and reached expected results -> almost no mistakes as problem was easy for ANN). 
On second and third dataset multi-class classification was tested. On MNIST dataset accuracy around 98% was reached, which was expected as implemented architecture should have been robust enough to handle the problem.
As for CIFAR-10 highest accuracy reached was around 54%. This result was also expected as CIFAR-10 dataset is too complicated for standard FFNNs implemented here. 
In this case to reach better results convolutional neural networks are needed, but they were not in the scope of this project. So, in this cause failure was expected and succes might have meant some problems in implementation (like wrong method to calculate accuracy).

### More details about implementation
For weigths initialization following methods were implemented: random, Xavier(Glorot) [4] and He(Kaiming) [5] with all having both Uniform and Normal distribution versions. As for further implementation, following activation functions were implemented: identity, sigmoid, tanh, ReLu, leaky ReLu and softmax. As for loss function both Mean Square Error and Cross-Entropy are implemented. Both l1 and l2 regularization were also added as options.
Furthermore special Softmax+Cross-Entropy option is availible, for cases when whole output layer is softmax and used loss function is Cross-Entropy. It improves speed of calculations and results of training by directly calculating derivative with regards to the logits.
Normal option for softmax and Cross-Entropy (standard backward, i.e. calculating Jacobian etc.) is also possible, so it is option, not forced version. For optimizers, following were implemented: Stochastic Gradient Descent (SGD) [6], SGD with momentum [7], Root Mean Square Propagation (RMSprop) [8], Nestorov accelerated gradient (NAG) [9] and Adaptive Moment Estimation (Adam) [10].
Functions to compute accuracy and make confusion matrices are built-in. Creation of any accuracy or loss curves is not directly built-in into code as it would make training process longer, which is not desirable on CPU based approach (NumPy). 
In our interpretation such outputs are possible and are showcased in WandBi sweeps as they were required. Nonethenless, base version (presented in TrainingExamples) have no such testing methods present (only accuracy calculation and confusion matrix plotting after training) for better efficiency of training.

### WandBi
According to the project goals FFNN implementation was used to performed WandBi sweeps (SweepFunctions.py and SweepExample.ipynb). Two datasets, MNIST [2] and CIFAR-10 [3], were used to perform sweeps. The best (at 05.12.2025 17:20) reached validation accuracy in sweeps for MNIST was 98.22% (dandy-sweep-112) and 54.22% (pretty-sweep-111) for CIFAR-10.
Following hyperparameters were used during sweeps:
- number of epochs
- number of hidden layers
- number of hidden units (architectures for sweeps were constructed such that each layer has same number of neurons - specified by this parameter)
- optimizer
- learning rate
- gradient clipping (value, was used always)
- momentum (value, relevant when SGD with momentum or RMSprop were used as optimizers as it is control parameter for them)
- beta (value, used only when RMSpropwas used as optimizer as it is control parameter for it)
- beta1 (value, used only when Adam used as optimizer as it is control parameter for it)
- beta2 (value, used only when Adam used as optimizer as it is control parameter for it)
- batch size
- regularization method (l1, l2 or none)
- l cooefficient (value, used only when l1 or l2 regularization was used as it is control parameter for them)
- method of weights initialization
- activation function for hidden layers ()
- activation function for output layers ()
- loss function

     
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
[1] Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. From: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
[2] LeCun, Yann; Cortez, Corinna; Burges, Christopher C.J. "The MNIST Handwritten Digit Database". Used by Keras.  
[3] Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, Technical Report, Computer Science Department, University of Toronto, 2009. Used by Keras.  
[4] Glorot X., Bengio Y., Understanding the difficulty of training deep feedforward neural networks, (2010) Journal of Machine Learning Research, 9, pp. 249 - 256,  
[5]He K., Zhang X., Ren S., Sun J., Delving deep into rectifiers: Surpassing human-level performance on imagenet classification, (2015) Proceedings of the IEEE International Conference on Computer Vision, 2015 International Conference on Computer Vision, ICCV 2015, art. no. 7410480, pp. 1026 - 1034, DOI: 10.1109/ICCV.2015.123  
[6] Bottou L., Stochastic gradient descent tricks, (2012) Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 7700 LECTURE NO, pp. 421 - 436, DOI: 10.1007/978-3-642-35289-8_25  
[7] Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. Nature, 323, 533-536.  
[8] Geoffrey Hinton, "Coursera Neural Networks for Machine Learning lecture 6", 2018. Link: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf  
[9] Sutskever I., Martens J., Dahl G., Hinton G., On the importance of initialization and momentum in deep learning, (2013) 30th International Conference on Machine Learning, ICML 2013, (PART 3), pp. 2176 - 2184  
[10] Kingma D.P., Ba J.L., Adam: A method for stochastic optimization, (2015) 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings  


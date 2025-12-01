import numpy as np
import pandas as pd
import wandb

from FNN import FNN
from ActivFunctions import (
    relu,
    sigmoid,
    tanh,
    softmax_vec,
)
from LossFunctions import SoftmaxCrossEntropy, SoftmaxCrossEntropyDerivative

from gradient_descent import (
    train_minibatch_sgd,
    train_minibatch_sgd_momentum,
    train_minibatch_rmsprop,
    train_minibatch_nag,
    train_minibatch_adam,
)

# DEFAULT CONFIG (can be overridden by a sweep)
DEFAULT_CONFIG = dict(
    train_path="Data/fashion-mnist_train.csv",
    test_path="Data/fashion-mnist_test.csv",

    # architecture
    num_hidden_layers=3,
    hidden_width=256,          # used to build n_hidden_units = [hidden_width]*num_hidden_layers
    activation_hidden="relu",  # "relu", "tanh", "sigmoid"
    activation_output="softmax_vec",
    weights_init="HeNor",

    # training
    epochs=15,
    learning_rate=5e-4,
    batch_size=128,
    optimizer="adam",          # "sgd", "sgd_momentum", "nag", "rmsprop", "adam"

    # optimizer-specific
    momentum=0.9,              # sgd_momentum, nag
    rmsprop_beta=0.9,          # rmsprop
    beta1=0.9,                 # adam
    beta2=0.999,               # adam
    epsilon=1e-8,              # adam, rmsprop

    # regularization
    l1_coeff=0.0,
    l2_coeff=0.0,

    # data split
    val_fraction=0.1,
)


# Activation mapping (strings -> functions)
ACTIVATION_FUNCS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
}

OUTPUT_ACTIVATION_FUNCS = {
    "softmax_vec": softmax_vec,
}


# DATA HELPERS
def load_fashion_mnist(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df["label"].to_numpy()
    X_train = train_df.drop(columns=["label"]).to_numpy().astype(np.float32)

    y_test = test_df["label"].to_numpy()
    X_test = test_df.drop(columns=["label"]).to_numpy().astype(np.float32)

    # normalize
    X_train /= 255.0
    X_test /= 255.0

    # (features, samples)
    X_train = X_train.T
    X_test = X_test.T

    return (X_train, y_train), (X_test, y_test)


def one_hot(y, num_classes=10):
    out = np.zeros((num_classes, y.shape[0]))
    out[y, np.arange(y.shape[0])] = 1
    return out


# LOSS + L2
def compute_loss_with_l2(network, X, Y, l2_coeff):
    """
    Total loss = data loss (SoftmaxCrossEntropy) + L2 penalty (if l2_coeff > 0).
    Bias weights (last column) are not regularized.
    """
    _, a_values = network.forward(X)
    logits = a_values[-1]

    data_loss = SoftmaxCrossEntropy(Y, logits)

    if l2_coeff != 0.0:
        l2 = 0.0
        for W in network.weights_list:
            W_no_bias = W[:, :-1]
            l2 += np.sum(W_no_bias ** 2)
        data_loss += 0.5 * l2_coeff * l2

    return data_loss


# OPTIMIZER DISPATCH
def train_one_epoch(net, X, Y, cfg):
    """
    Train the network for exactly ONE epoch using the optimizer specified in cfg.
    """
    opt = cfg.optimizer.lower()

    if opt == "sgd":
        return train_minibatch_sgd(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=SoftmaxCrossEntropyDerivative,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
        )

    if opt == "sgd_momentum":
        return train_minibatch_sgd_momentum(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=SoftmaxCrossEntropyDerivative,
            momentum=cfg.momentum,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
        )

    if opt == "nag":
        return train_minibatch_nag(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=SoftmaxCrossEntropyDerivative,
            momentum=cfg.momentum,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
        )

    if opt == "rmsprop":
        return train_minibatch_rmsprop(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=SoftmaxCrossEntropyDerivative,
            beta=cfg.rmsprop_beta,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
        )

    if opt == "adam":
        return train_minibatch_adam(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=SoftmaxCrossEntropyDerivative,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            epsilon=cfg.epsilon,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
        )

    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def accuracy(network, X, Y):
    _, a_values = network.forward(X)
    preds = np.argmax(a_values[-1], axis=0)
    labels = np.argmax(Y, axis=0)
    return np.mean(preds == labels)


# MAIN TRAIN FUNCTION (USED BY SWEEPS)
def main():
    """
    This function is executed once per run.
    When used with sweeps, W&B overrides values in DEFAULT_CONFIG.
    """
    # Initialize wandb and load config
    run = wandb.init(config=DEFAULT_CONFIG, project="ffnn-from-scratch")
    cfg = wandb.config  # this holds defaults + sweep overrides

    # 1. Build derived hyperparameters
    # Build n_hidden_units from num_hidden_layers + hidden_width
    n_hidden_units = [cfg.hidden_width] * cfg.num_hidden_layers

    # Map string activations to functions
    act_hidden_fn = ACTIVATION_FUNCS[cfg.activation_hidden]
    act_output_fn = OUTPUT_ACTIVATION_FUNCS[cfg.activation_output]

    # 2. Load data + split train/val
    (X_train_full, y_train_full), (X_test, y_test) = load_fashion_mnist(
        cfg.train_path, cfg.test_path
    )
    Y_train_full = one_hot(y_train_full, num_classes=10)
    Y_test = one_hot(y_test, num_classes=10)

    N = X_train_full.shape[1]
    indices = np.random.permutation(N)
    val_size = int(N * cfg.val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = X_train_full[:, train_idx]
    Y_train = Y_train_full[:, train_idx]

    X_val = X_train_full[:, val_idx]
    Y_val = Y_train_full[:, val_idx]

    # 3. Build FNN
    input_dim = X_train.shape[0]
    num_classes = 10

    layer_sizes = [input_dim] + list(n_hidden_units) + [num_classes]
    activations = [act_hidden_fn] * cfg.num_hidden_layers + [act_output_fn]

    net = FNN(
        weights_info=layer_sizes,
        activ_functions_info=activations,
        method_ini=cfg.weights_init,
    )

    # 4. Training loop (log train/val metrics each epoch)
    for epoch in range(cfg.epochs):
        net = train_one_epoch(net, X_train, Y_train, cfg)

        train_loss = compute_loss_with_l2(net, X_train, Y_train, cfg.l2_coeff)
        train_acc = accuracy(net, X_train, Y_train)

        val_loss = compute_loss_with_l2(net, X_val, Y_val, cfg.l2_coeff)
        val_acc = accuracy(net, X_val, Y_val)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    # 5. Final test metrics
    test_loss = compute_loss_with_l2(net, X_test, Y_test, cfg.l2_coeff)
    test_acc = accuracy(net, X_test, Y_test)

    wandb.log(
        {
            "final_test_loss": test_loss,
            "final_test_acc": test_acc,
        }
    )

    print(
        f"[RUN DONE] opt={cfg.optimizer}, act={cfg.activation_hidden}, "
        f"layers={cfg.num_hidden_layers}, width={cfg.hidden_width} -> "
        f"test_acc={test_acc:.4f}"
    )

    run.finish()


if __name__ == "__main__":
    main()

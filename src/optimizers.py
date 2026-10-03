"""
CNG403 Assignment 1 — Scratch Implementation: Optimisers
=========================================================
Implement SGD and Batch Gradient Descent using only tensor operations.

Terminology used in this assignment:
  - SGD            : update weights after every SINGLE sample  (batch_size = 1)
  - Batch GD       : update weights after the FULL dataset     (batch_size = N)
  - Mini-batch GD  : update weights after a MINI-BATCH         (1 < batch_size < N)

In practice, the update rule is identical for all three — the difference is
purely in how much data is used to compute the gradient before each step.
The training loop (in train.py) controls the batch size; the optimiser
just applies the update rule.

Run the sanity check at the bottom:
  python optimizers.py
"""

import torch
from network import FFNN


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Optimizer:
    """Abstract base for all optimisers."""

    def __init__(self, model: FFNN, lr: float) -> None:
        self.model = model
        self.lr    = lr

    def step(self) -> None:
        """Apply one parameter update using the gradients stored in each Linear layer."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset all gradients to zero before the next forward/backward pass."""
        for layer in self.model.linear_layers:
            layer.dW.zero_()
            layer.db.zero_()


# ---------------------------------------------------------------------------
# SGD (and by extension Mini-batch / Batch GD)
# ---------------------------------------------------------------------------

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Update rules:
        Without momentum:
            W ← W - lr * dW
            b ← b - lr * db

        With momentum (Polyak momentum):
            v_W ← momentum * v_W + dW
            v_b ← momentum * v_b + db
            W   ← W - lr * v_W
            b   ← b - lr * v_b

    Attributes:
        momentum  (float): momentum coefficient (0 = vanilla SGD).
        velocity_W (list[torch.Tensor]): velocity buffers for weights, one per Linear layer.
        velocity_b (list[torch.Tensor]): velocity buffers for biases,  one per Linear layer.
    """

    def __init__(self, model: FFNN, lr: float, momentum: float = 0.0) -> None:
        super().__init__(model, lr)
        self.momentum   = momentum
        # Initialise velocity buffers to zero (same shape as W and b)
        self.velocity_W = [torch.zeros_like(l.W) for l in model.linear_layers]
        self.velocity_b = [torch.zeros_like(l.b) for l in model.linear_layers]

    def step(self) -> None:
        """
        Apply one SGD update step to all Linear layers.

        Use self.model.linear_layers to iterate over layers.
        Use self.velocity_W and self.velocity_b for momentum buffers.
        """
        # TODO: implement the update rule described in the docstring above.
        #
        # Remember: update parameters IN-PLACE (layer.W -= ..., not layer.W = ...)
        # to avoid breaking references held elsewhere.
        i  = 0
        for layer in self.model.linear_layers:
            if self.momentum > 0:
                self.velocity_W[i] = self.momentum * self.velocity_W[i] + layer.dW
                self.velocity_b[i] = self.momentum * self.velocity_b[i] +  layer.db
                layer.W = layer.W - self.lr * self.velocity_W[i]
                layer.b = layer.b - self.lr * self.velocity_b[i]
            else:
                layer.W = layer.W - self.lr * layer.dW
                layer.b = layer.b - self.lr * layer.db
            i = i+1


# ---------------------------------------------------------------------------
# Batch Gradient Descent (explicit class for clarity)
# ---------------------------------------------------------------------------

class BatchGD(SGD):
    """
    Batch (Full) Gradient Descent — identical update rule to SGD.

    The distinction is semantic: BatchGD is meant to be called once per epoch
    with gradients accumulated over the entire dataset.  The training loop
    in train.py enforces this by setting batch_size = len(train_set).

    No additional implementation needed — inherits step() from SGD.
    """
    pass


# ---------------------------------------------------------------------------
# Sanity check  (run with: python optimizers.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn_ref
    import torch.optim as optim_ref
    from layers import CrossEntropyLoss

    torch.manual_seed(7)
    print("=" * 60)
    print("Sanity check for optimizers.py")
    print("=" * 60)

    B, D_in = 8, 32
    hidden  = [16]
    C       = 4

    # ---- Vanilla SGD (no momentum) ----------------------------------------
    print("\n[SGD — no momentum]")

    net = FFNN(D_in, hidden, C, activation="relu")
    ref_net = nn_ref.Sequential(
        nn_ref.Linear(D_in, hidden[0]), nn_ref.ReLU(),
        nn_ref.Linear(hidden[0], C),
    )
    # Sync weights
    ref_linears = [m for m in ref_net.modules() if isinstance(m, nn_ref.Linear)]
    for sl, rl in zip(net.linear_layers, ref_linears):
        rl.weight.data = sl.W.clone()
        rl.bias.data   = sl.b.clone()

    x      = torch.randn(B, D_in)
    labels = torch.randint(0, C, (B,))
    lr     = 0.05

    # Reference step
    ref_opt = optim_ref.SGD(ref_net.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    ref_opt.zero_grad()
    nn_ref.CrossEntropyLoss()(ref_net(x), labels).backward()
    ref_opt.step()

    # Scratch step
    opt = SGD(net, lr=lr, momentum=0.0)
    opt.zero_grad()
    loss_fn = CrossEntropyLoss()
    loss_fn(net(x), labels)
    net.backward(loss_fn.backward())
    opt.step()

    for sl, rl in zip(net.linear_layers, ref_linears):
        dW = (sl.W - rl.weight).abs().max()
        db = (sl.b - rl.bias).abs().max()
        print(f"  W_diff={dW:.2e}  b_diff={db:.2e}  (expect < 1e-5)")

    # ---- SGD with momentum ------------------------------------------------
    print("\n[SGD — momentum=0.9]")

    net2 = FFNN(D_in, hidden, C, activation="relu")
    ref_net2 = nn_ref.Sequential(
        nn_ref.Linear(D_in, hidden[0]), nn_ref.ReLU(),
        nn_ref.Linear(hidden[0], C),
    )
    ref_linears2 = [m for m in ref_net2.modules() if isinstance(m, nn_ref.Linear)]
    for sl, rl in zip(net2.linear_layers, ref_linears2):
        rl.weight.data = sl.W.clone()
        rl.bias.data   = sl.b.clone()

    momentum = 0.9
    ref_opt2 = optim_ref.SGD(ref_net2.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)
    opt2     = SGD(net2, lr=lr, momentum=momentum)

    # Run 3 steps to exercise the velocity buffers
    for step_i in range(3):
        x_i      = torch.randn(B, D_in)
        labels_i = torch.randint(0, C, (B,))

        ref_opt2.zero_grad()
        nn_ref.CrossEntropyLoss()(ref_net2(x_i), labels_i).backward()
        ref_opt2.step()

        opt2.zero_grad()
        lf = CrossEntropyLoss()
        lf(net2(x_i), labels_i)
        net2.backward(lf.backward())
        opt2.step()

    for sl, rl in zip(net2.linear_layers, ref_linears2):
        dW = (sl.W - rl.weight).abs().max()
        db = (sl.b - rl.bias).abs().max()
        print(f"  W_diff={dW:.2e}  b_diff={db:.2e}  (expect < 1e-5)")

    print("\nAll checks passed (if all diffs < 1e-5).\n")

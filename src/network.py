"""
CNG403 Assignment 1 — Scratch Implementation: FFNN
===================================================
Implement the FFNN class using the layers you built in layers.py.

Rules:
  - No torch.nn, no torch.autograd, no torch.optim.
  - Only use your own Layer subclasses from layers.py.

Run the sanity check at the bottom:
  python network.py
"""

import torch
from layers import CrossEntropyLoss, Linear, ReLU, Sigmoid, Tanh


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    "relu":    ReLU,
    "sigmoid": Sigmoid,
    "tanh":    Tanh,
}


# ---------------------------------------------------------------------------
# FFNN
# ---------------------------------------------------------------------------

class FFNN:
    """
    Feed-Forward Neural Network built from scratch.

    Architecture:
        Input → [Linear → Activation] × n_hidden → Linear (output logits)

    Attributes:
        layers (list): ordered list of Layer objects (Linear and activation layers).
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int], num_classes: int, activation: str) -> None:
        """
        Build the list of layers.

        Args:
            input_dim    (int):       dimensionality of the input feature vector.
            hidden_sizes (list[int]): number of units in each hidden layer.
            num_classes  (int):       number of output classes.
            activation   (str):       one of "relu", "sigmoid", "tanh".

        Example — hidden_sizes=[256, 128]:
            Linear(input_dim → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 10)
        """
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")

        # TODO: populate self.layers with Linear and activation Layer objects
        #
        # Hint: iterate over hidden_sizes, adding a Linear then an activation
        # for each hidden layer. Then add the final Linear output layer.
        self.layers = []
        number = input_dim
        for i in hidden_sizes:
            self.layers += [Linear(number, i)]

            self.layers += [ACTIVATIONS[activation]()]

            number = i
        self.layers += [Linear(number, num_classes)]


    # ------------------------------------------------------------------
    # Trainable layers convenience property
    # ------------------------------------------------------------------

    @property
    def linear_layers(self) -> list[Linear]:
        """Return only the Linear layers (the ones with learnable parameters)."""
        return [l for l in self.layers if isinstance(l, Linear)]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass x through all layers in order.

        Args:
            x:       (batch_size, input_dim)
        Returns:
            logits:  (batch_size, num_classes)  — raw scores before softmax
        """
        # TODO: sequentially apply each layer in self.layers
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, grad_loss: torch.Tensor) -> None:
        """
        Backpropagate the loss gradient through all layers in reverse order.

        After this call, every Linear layer in self.layers will have its
        .dW and .db attributes updated.

        Args:
            grad_loss: (batch_size, num_classes) — gradient from CrossEntropyLoss.backward()
        """
        # TODO: propagate grad_loss backwards through self.layers in reverse
        for layer in reversed(self.layers):
            grad_loss = layer.backward(grad_loss)


    # ------------------------------------------------------------------
    # L2 regularisation gradient contribution
    # ------------------------------------------------------------------

    def l2_grad(self, l2_lambda: float) -> None:
        """
        Add the L2 regularisation term to each Linear layer's weight gradient.

        The L2 loss is:  lambda * sum(W^2)
        Its gradient is: 2 * lambda * W

        This should be called AFTER backward() and BEFORE the optimiser step.

        Args:
            l2_lambda (float): regularisation strength.
        """
        # TODO: for each Linear layer, add 2 * l2_lambda * layer.W to layer.dW
        for e in self.linear_layers:
            e.dW = e.dW + 2 * l2_lambda * e.W

# ---------------------------------------------------------------------------
# Sanity check  (run with: python network.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn_ref

    torch.manual_seed(42)
    print("=" * 60)
    print("Sanity check for network.py")
    print("=" * 60)

    B, D_in = 16, 144
    hidden_sizes = [64, 32]
    num_classes  = 10

    # Build scratch network
    net = FFNN(D_in, hidden_sizes, num_classes, activation="relu")

    # Build equivalent reference network with the SAME weights
    ref_layers = []
    prev = D_in
    lin_idx = 0
    for h in hidden_sizes:
        ref_layers += [nn_ref.Linear(prev, h), nn_ref.ReLU()]
        prev = h
    ref_layers.append(nn_ref.Linear(prev, num_classes))
    ref_net = nn_ref.Sequential(*ref_layers)

    # Copy weights from scratch net into reference net
    ref_linears = [m for m in ref_net.modules() if isinstance(m, nn_ref.Linear)]
    for scratch_l, ref_l in zip(net.linear_layers, ref_linears):
        ref_l.weight.data = scratch_l.W.clone()
        ref_l.bias.data   = scratch_l.b.clone()

    x = torch.randn(B, D_in)

    # Forward
    out_scratch = net(x)
    out_ref     = ref_net(x)
    print(f"\nForward  max_diff = {(out_scratch - out_ref).abs().max():.2e}  (expect < 1e-5)")

    # Backward
    labels   = torch.randint(0, num_classes, (B,))
    loss_fn  = CrossEntropyLoss()
    loss     = loss_fn(out_scratch, labels)
    grad     = loss_fn.backward()
    net.backward(grad)

    x_ref   = x.clone().requires_grad_(True)
    out_ref2 = ref_net(x_ref)
    nn_ref.CrossEntropyLoss()(out_ref2, labels).backward()

    for i, (sl, rl) in enumerate(zip(net.linear_layers, ref_linears)):
        dW_diff = (sl.dW - rl.weight.grad).abs().max()
        db_diff = (sl.db - rl.bias.grad).abs().max()
        print(f"  Layer {i}  dW_diff={dW_diff:.2e}  db_diff={db_diff:.2e}  (expect < 1e-5)")

    print("\nAll checks passed (if all diffs < 1e-5).\n")

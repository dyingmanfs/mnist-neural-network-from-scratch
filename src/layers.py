"""
CNG403 Assignment 1 — Scratch Implementation: Layers & Activations
===================================================================
Implement every class marked with TODO using only torch.Tensor operations.

Rules:
  - No torch.nn, no torch.autograd, no torch.optim.
  - You may use torch.tensor, torch.zeros, torch.matmul, @, +, *, etc.
  - Every forward() must store the values needed for backward() in self.cache.
  - Tensor shapes are documented on every method — match them exactly.

Run the sanity checks at the bottom of this file to test your implementation:
  python layers.py
"""

import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Layer:
    """Abstract base for all layers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class Linear(Layer):
    """
    Fully-connected (affine) layer:  out = x @ W.T + b

    Attributes:
        W  (torch.Tensor): weight matrix of shape (out_features, in_features).
        b  (torch.Tensor): bias vector  of shape (out_features,).
        dW (torch.Tensor): gradient w.r.t. W, same shape as W. Set in backward().
        db (torch.Tensor): gradient w.r.t. b, same shape as b. Set in backward().
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        # Kaiming uniform initialisation (matches PyTorch default for nn.Linear)
        std = (2.0 / in_features) ** 0.5
        self.W  = torch.randn(out_features, in_features) * std

        self.b  = torch.zeros(out_features)
        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)
        self.cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch_size, in_features)
        Returns:
            out: (batch_size, out_features)
        """
        # TODO: compute the affine transformation and store x in self.cache
        self.cache["x"] = x
        z = x @ self.W.T + self.b
        return z


    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients given the upstream gradient.

        Args:
            grad_out: (batch_size, out_features)  — gradient of loss w.r.t. output
        Returns:
            grad_x:   (batch_size, in_features)   — gradient of loss w.r.t. input

        Side effects:
            Sets self.dW and self.db (averaged over the batch).
        """
        # TODO: compute grad_x, self.dW, self.db using the chain rule
        #
        # Hints:
        #   grad_x  = grad_out @ W
        #   dW      = grad_out.T @ x   (then average over batch: / batch_size)
        #   db      = grad_out.mean(dim=0)
        x = self.cache["x"]
        batch_size = x.shape[0]
        grad_x = grad_out @ self.W
        self.dW  = (grad_out.T @ x )
        self.db = grad_out.sum(dim=0)
        return grad_x




# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class ReLU(Layer):
    """
    Rectified Linear Unit:  out = max(0, x)
    """
    def __init__(self):
        self.cache = {}
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch_size, features)
        Returns:
            out: (batch_size, features)
        """
        # TODO: apply ReLU and store the mask needed for backward in self.cache
        out = torch.maximum(x, torch.zeros_like(x))
        self.cache["x"] = x > 0
        return out

    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grad_out: (batch_size, features)
        Returns:
            grad_x:   (batch_size, features)
        """
        # TODO: gradient is 1 where x > 0, else 0
        x = self.cache["x"]
        grad_x = torch.zeros_like(grad_out)

        grad_x = grad_out * (self.cache["x"])
        return grad_x


class Sigmoid(Layer):
    def __init__(self):
        self.cache = {}
    """
    Sigmoid function:  out = 1 / (1 + exp(-x))
    Derivative:        d_sigmoid = out * (1 - out)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch_size, features)
        Returns:
            out: (batch_size, features)
        """
        # TODO: compute sigmoid and cache the output (needed for backward)
        z = 1/(1 + torch.exp(-x))
        self.cache["z"] =  z
        return z


    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grad_out: (batch_size, features)
        Returns:
            grad_x:   (batch_size, features)
        """
        # TODO: use the cached sigmoid output to compute the gradient
        z=self.cache["z"]
        grad_x = grad_out * z * (1 - z)
        return grad_x





class Tanh(Layer):
    """
    Hyperbolic tangent:  out = tanh(x)
    Derivative:          d_tanh = 1 - out^2
    """
    def __init__(self):
        self.cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch_size, features)
        Returns:
            out: (batch_size, features)
        """
        # TODO: compute tanh and cache the output
        z =  torch.tanh(x)
        self.cache["z"] = z
        return z


    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grad_out: (batch_size, features)
        Returns:
            grad_x:   (batch_size, features)
        """
        # TODO: use the cached tanh output to compute the gradient
        z = self.cache["z"]
        grad_x = grad_out * (1 - z * z)
        return grad_x


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CrossEntropyLoss:
    """

    Numerically stable softmax cross-entropy loss.

    Combines softmax + negative log-likelihood in one step, which is more
    numerically stable than computing them separately.

    Usage:
        loss_fn = CrossEntropyLoss()
        loss    = loss_fn(logits, labels)   # forward
        grad    = loss_fn.backward()         # upstream gradient = 1
    """
    def __init__(self):
        self.cache = {}

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.forward(logits, labels)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)  — raw scores (before softmax)
            labels: (batch_size,)              — integer class indices in [0, C)
        Returns:
            loss:   scalar tensor              — mean cross-entropy over the batch

        Steps (implement in order):
            1. Subtract max per row for numerical stability.
            2. Compute softmax probabilities.
            3. Gather the probability of the true class for each sample.
            4. Compute negative log and average over the batch.
            5. Cache whatever you need for backward.
        """
        # TODO
        batch_size = logits.shape[0]
        numarical = logits - logits.max(dim=1, keepdim=True).values

        softmax = torch.exp(numarical)
        softmaxprob = softmax / torch.sum(softmax, dim=1, keepdim=True)
        tureclassprobs = softmaxprob[torch.arange(batch_size), labels]
        loss = torch.log(tureclassprobs).mean()
        loss = loss*-1
        self.cache = {"probs": softmaxprob,"labels": labels , "batch_size": batch_size }
        return loss


    def backward(self) -> torch.Tensor:
        """
        Returns the gradient of the loss w.r.t. the logits.

        Returns:
            grad_logits: (batch_size, num_classes)

        Hint:
            The combined softmax + cross-entropy gradient is elegantly:
                grad_logits[i, j] = (softmax[i, j] - 1{j == label[i]}) / batch_size
        """
        # TODO
        probs = self.cache["probs"]
        labels = self.cache["labels"]
        batch_size = self.cache["batch_size"]

        grad_logits = probs.clone()

        for i in range(batch_size):
            j = labels[i]
            grad_logits[i][j] -= 1

        grad_logits = grad_logits / batch_size

        return grad_logits


# ---------------------------------------------------------------------------
# Sanity checks  (run with: python layers.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn_ref

    torch.manual_seed(0)
    print("=" * 60)
    print("Sanity checks for layers.py")
    print("=" * 60)

    B, D_in, D_out = 8, 16, 4

    # ---- Linear -----------------------------------------------------------
    print("\n[Linear]")
    x   = torch.randn(B, D_in)
    ref = nn_ref.Linear(D_in, D_out, bias=True)

    layer = Linear(D_in, D_out)
    layer.W = ref.weight.clone().detach()
    layer.b = ref.bias.clone().detach()

    out_scratch = layer.forward(x)
    out_ref     = ref(x)
    print(f"  forward  max_diff = {(out_scratch - out_ref).abs().max():.2e}  (expect < 1e-5)")

    grad_out = torch.randn(B, D_out)
    grad_x_scratch = layer.backward(grad_out)

    x_ref = x.clone().requires_grad_(True)
    out_ref = ref(x_ref)
    out_ref.backward(grad_out)
    print(f"  grad_x   max_diff = {(grad_x_scratch - x_ref.grad).abs().max():.2e}  (expect < 1e-5)")
    print(f"  grad_W   max_diff = {(layer.dW - ref.weight.grad).abs().max():.2e}  (expect < 1e-5)")
    print(f"  grad_b   max_diff = {(layer.db - ref.bias.grad).abs().max():.2e}  (expect < 1e-5)")

    # ---- ReLU -------------------------------------------------------------
    print("\n[ReLU]")
    x       = torch.randn(B, D_in)
    x_ref   = x.clone().requires_grad_(True)
    act     = ReLU()
    out     = act.forward(x)
    out_ref = torch.relu(x_ref)
    print(f"  forward  max_diff = {(out - out_ref).abs().max():.2e}  (expect < 1e-5)")
    grad_out = torch.randn(B, D_in)
    out_ref.backward(grad_out)
    grad_x = act.backward(grad_out)
    print(f"  backward max_diff = {(grad_x - x_ref.grad).abs().max():.2e}  (expect < 1e-5)")

    # ---- Sigmoid ----------------------------------------------------------
    print("\n[Sigmoid]")
    x       = torch.randn(B, D_in)
    x_ref   = x.clone().requires_grad_(True)
    act     = Sigmoid()
    out     = act.forward(x)
    out_ref = torch.sigmoid(x_ref)
    print(f"  forward  max_diff = {(out - out_ref).abs().max():.2e}  (expect < 1e-5)")
    grad_out = torch.randn(B, D_in)
    out_ref.backward(grad_out)
    grad_x = act.backward(grad_out)
    print(f"  backward max_diff = {(grad_x - x_ref.grad).abs().max():.2e}  (expect < 1e-5)")

    # ---- Tanh -------------------------------------------------------------
    print("\n[Tanh]")
    x       = torch.randn(B, D_in)
    x_ref   = x.clone().requires_grad_(True)
    act     = Tanh()
    out     = act.forward(x)
    out_ref = torch.tanh(x_ref)
    print(f"  forward  max_diff = {(out - out_ref).abs().max():.2e}  (expect < 1e-5)")
    grad_out = torch.randn(B, D_in)
    out_ref.backward(grad_out)
    grad_x = act.backward(grad_out)
    print(f"  backward max_diff = {(grad_x - x_ref.grad).abs().max():.2e}  (expect < 1e-5)")

    # ---- CrossEntropyLoss -------------------------------------------------
    print("\n[CrossEntropyLoss]")
    logits   = torch.randn(B, 10)
    labels   = torch.randint(0, 10, (B,))
    loss_fn  = CrossEntropyLoss()
    loss     = loss_fn(logits, labels)
    ref_loss = nn_ref.CrossEntropyLoss()(logits, labels)
    print(f"  forward  max_diff = {(loss - ref_loss).abs().max():.2e}  (expect < 1e-5)")
    grad     = loss_fn.backward()
    logits_r = logits.clone().requires_grad_(True)
    nn_ref.CrossEntropyLoss()(logits_r, labels).backward()
    print(f"  backward max_diff = {(grad - logits_r.grad).abs().max():.2e}  (expect < 1e-5)")

    print("\nAll checks passed (if all diffs < 1e-5).\n")

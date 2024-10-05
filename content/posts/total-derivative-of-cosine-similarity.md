+++
title = 'The gradient of cosine similarity'
date = 2024-10-04T12:36:14-07:00
draft = true
tag = ['cosine-similarity', 'gradient', 'random-notes']
+++

## Introduction

In this post, we compute the gradient of the cosine similarity function.
Then, we numerically verify the result's correctness using PyTorch's `gradcheck()`
function.

<!--more-->

$$
    \newcommand{\cS}{\mathcal{S}}
    \newcommand{\bR}{\mathbb{R}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Cosine similarity

The *cosine similarity function* is defined by
$$
\begin{align*}
    \cS : \bR^n_* \times \bR^n_* &\to \bR \\
    (x, y) &\mapsto \cS(x, y) = \frac{\langle x, y \rangle}{\| x \| \| y \|},
\end{align*}
$$
where \(\bR^n_*\) denotes \(\bR^n\) with the origin \(0_n\) removed, \(\langle \cdot, \cdot \rangle\) is the
Euclidean inner product on \(\bR^n\), and \(\| \cdot \|\) is the Euclidean norm
on \(\bR^n\).  The function \(\cS\) is smooth by composition.

## Total derivative

The total derivative and partial derivatives of \(\cS\) at \((x, y)\) are related by
$$
    d\cS(x,y) \cdot (\tilde{x},\tilde{y}) =
    d_x \cS(x,y) \cdot \tilde{x} + d_y \cS(x,y) \cdot \tilde{y}.
$$
By the quotient rule, the partial derivative of \(\cS\) with respect to \(x\) at \((x, y)\) is
$$
\begin{align*}
    d_x \cS(x,y) \cdot \tilde{x}
    &= \frac{\| x \| \| y \| \langle \tilde{x}, y \rangle - \langle x, y \rangle
    \frac{\langle \tilde{x}, x \rangle}{\| x \|} \| y \|}{\| x \|^2 \| y \|^2} \\
    &= \frac{\langle \tilde{x}, y \rangle}{\| x \| \| y \|} -
    \frac{\langle x, y \rangle \langle \tilde{x}, x \rangle}{\| x \|^3 \| y \|} \\
    &= \frac{\langle \tilde{x}, y \rangle}{\| x \| \| y \|} -
    \cS(x, y) \frac{\langle \tilde{x}, x \rangle}{\| x \|^2} \\
    &= \left\langle
        \tilde{x},
        \frac{1}{\| x \| \| y \|} y - \frac{\cS(x, y)}{\| x \|^2} x
    \right\rangle.
\end{align*}
$$
Here, we have used the easily-verified fact that
$$
    d(x \mapsto \| x \|)(x) \cdot \tilde{x} = \frac{\langle \tilde{x}, x \rangle}{\| x \|}
$$
for \(x \neq 0_n\). Similarly, the partial derivative of \(\cS\) with respect to \(y\) at \((x, y)\) is
$$
\begin{align*}
    d_y \cS(x,y) \cdot \tilde{y}
    &= \left\langle
        \tilde{y},
        \frac{1}{\| x \| \| y \|} x - \frac{\cS(x, y)}{\| y \|^2} y
    \right\rangle.
\end{align*}
$$
Consequently
$$
\colorbox{magicmint}
{
$
    d\cS(x,y) \cdot (\tilde{x},\tilde{y}) =
    \left\langle
        \tilde{x},
        \frac{1}{\| x \| \| y \|} y - \frac{\cS(x, y)}{\| x \|^2} x
    \right\rangle
    +
    \left\langle
        \tilde{y},
        \frac{1}{\| x \| \| y \|} x - \frac{\cS(x, y)}{\| y \|^2} y
    \right\rangle.
$
}
$$
Alternatively, after rearranging some terms
$$
\colorbox{lesserbox}
{
$
    d\cS(x,y) \cdot (\tilde{x},\tilde{y}) =
    \frac{\langle \tilde{x}, y \rangle + \langle x, \tilde{y} \rangle}{\| x \| \| y \|} -
    \cS(x, y) \left(
        \frac{\langle \tilde{x}, x \rangle}{\| x \|^2} -
        \frac{\langle y, \tilde{y} \rangle}{\| y \|^2}
    \right).
$
}
$$

## Gradient

In this section, we compute the gradient of \(\cS\) at \((x, y)\), denoted by \(\nabla \cS(x,y)\).

Recall that \(\nabla \cS(x,y)\) is the unique element of \(\bR^n \times \bR^n\) that satisfies
$$
    d\cS(x, y) \cdot (\tilde{x}, \tilde{y}) = \langle (\tilde{x}, \tilde{y}), \nabla \cS(x, y) \rangle_{\bR^n \times \bR^n}
$$

The main result of the previous section can be rewritten as
$$
    d\cS(x,y) \cdot (\tilde{x},\tilde{y}) =
    \left\langle
        (\tilde{x},\tilde{y}),
        \left(\frac{1}{\| x \| \| y \|} y - \frac{\cS(x, y)}{\| x \|^2} x,
        \frac{1}{\| x \| \| y \|} x - \frac{\cS(x, y)}{\| y \|^2} y
        \right)
    \right\rangle_{\bR^n \times \bR^n},
$$
where \(\langle \cdot, \cdot \rangle_{\bR^n \times \bR^n}\) is the natural inner
product on \(\bR^n \times \bR^n\).  That is,
$$
    \langle (v_1, w_1), (v_2, w_2) \rangle_{\bR^n \times \bR^n} =
    \langle v_1, v_2 \rangle + \langle w_1, w_2 \rangle.
$$
It follows immediately that
$$
    \nabla \cS(x,y) = \left(
        \frac{1}{\| x \| \| y \|} y - \frac{\cS(x, y)}{\| x \|^2} x,
        \frac{1}{\| x \| \| y \|} x - \frac{\cS(x, y)}{\| y \|^2} y
    \right).
$$

## Verification

For verification, we can implement \(\cS\) as a `torch.autograd.Function`, as follows:

```python
from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function, gradcheck
from torch.autograd.function import FunctionCtx


class CosineSimilarity(Function):
    """Cosine similarity function."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, x: Float[Tensor, "n 1"], y: Float[Tensor, "n 1"]
    ) -> Float[Tensor, "1 1"]:
        """Compute cosine similarity function output.

        Args:
            ctx (FunctionCtx): Context used to stash data for `backward()`.
            x (Tensor): Input tensor 1.
            y (Tensor): Input tensor 2.
        """
        x_norm = x.norm(keepdim=True)
        y_norm = y.norm(keepdim=True)
        s = torch.sum(x * y, dim=0, keepdim=True) / (x_norm * y_norm)
        ctx.save_for_backward(x, y, x_norm, y_norm, s)
        return s

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: FunctionCtx, grad_output: Float[Tensor, "1 1"]
    ) -> Tuple[Float[Tensor, "n 1"]]:
        """Compute gradient of cosine similarity function.

        Args:
            ctx (FunctionCtx): Context used to retrieve stashed data from `forward()`.
            grad_output (Tensor): Output gradient tensor.
        """
        x, y, x_norm, y_norm, s = ctx.saved_tensors  # type: ignore[attr-defined]
        norm_product = x_norm * y_norm

        # Compute gradient with respect to x
        grad_x = y / norm_product - (s / (x_norm * x_norm)) * x
        grad_x = grad_output * grad_x

        # Compute gradient with respect to y
        grad_y = x / norm_product - (s / (y_norm * y_norm)) * y
        grad_y = grad_output * grad_y

        # Return gradient
        return grad_x, grad_y
```

Then we can compare numerical to analytical gradients, using `gradcheck()`:

```python
def main():
    """Check correctness of `CosineSimilarity.backward()` using `gradcheck()`."""
    n = 16
    x = torch.ones(n, 1, dtype=torch.double, requires_grad=True)
    y = torch.ones(n, 1, dtype=torch.double, requires_grad=True)

    if gradcheck(CosineSimilarity.apply, [x, y], eps=1e-6, atol=1e-4):
        print("success!")
    else:
        print("failure...")


if __name__ == "__main__":
    main()
```

The results are...

```console
$ python cosine_similarity.py
success!
```

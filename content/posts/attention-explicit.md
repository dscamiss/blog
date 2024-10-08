+++
title = 'An explicit formula for the gradient of scaled dot-product attention'
date = 2024-09-19T11:03:04-07:00
draft = false
tag = ['attention', 'gradient', 'backpropagation', 'random-notes']
+++

## Introduction

In this post, we derive an explicit formula for the gradient of the
scaled dot-product attention map.  Then, we numerically verify the formula's
correctness using PyTorch's `gradcheck()` function.

<!--more-->

$$
    \newcommand{\Attn}{\mathrm{Att}}
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\sL}{\mathscr{L}}
    \newcommand{\tr}{\mathrm{tr}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Dot-product attention

We begin by defining
$$
    \Theta = \bR^{n \times d} \times \bR^{n \times d} \times \bR^{n \times d}
$$
with generic element \(\theta = (Q, K, V)\).

We regard \(\Theta\) as an inner product space with the inner product
$$
    \langle (Q_1, K_1, V_1), (Q_2, K_2, V_2) \rangle_\Theta =
    \langle Q_1, Q_2 \rangle_F +
    \langle K_1, K_2 \rangle_F +
    \langle V_1, V_2 \rangle_F.
$$
Here, \(\langle A, B \rangle_F = \tr(A^t B)\) is the Frobenius inner product on \(\bR^{n \times d}\).

The *dot-product attention map* is defined by
$$
\begin{align*}
    \Attn : \Theta &\to \bR^{n \times d} \\
    \theta &\mapsto \Attn(\theta) = \sigma(QK^t) V,
\end{align*}
$$
where \(\sigma : \bR^n \to \bR^n\) is the [softmax map]({{< ref "softmax-derivatives" >}})
applied row-wise.  This means that
$$
\begin{align*}
    \Attn(\theta) &= \sum_{i=1}^{n} e_i \sigma((e_i^t Q K^t)^t)^t V \\
    &= \sum_{i=1}^{n} e_i \sigma(K Q^t e_i)^t V,
\end{align*}
$$
where \(e_i\) is the \(i\)-th Euclidean basis vector in \(\bR^n\).  The
"extra" transposes are needed to ensure that the inputs to \(\sigma\)
are column vectors, and the outputs of \(\sigma\) are row vectors.
For notational convenience, we have chosen not to include the standard scaling
factor \(1/\sqrt{d}\) applied to \(QK^t\); the "scaled" case is discussed
further below.  Also, we have elected to have the dimensions
of \(Q, K\) and \(V\) match, since this is typical in applications.

## Gradient

Recall that the notion of "gradient" is only well-defined for smooth
functions (i.e., smooth real-valued maps).  In particular, the "gradient" of \(\Attn\) is
not well-defined, so we will work instead with the composition of \(\Attn\) and
an (arbitrary) smooth function \(\ell : \bR^{n \times d} \to \bR\).

To this end, define the smooth function
$$
\begin{align*}
    \sL = \ell \circ \Attn.
\end{align*}
$$
We will make use of the following quantities:
* \(\nabla \sL(\theta)\): The (total) gradient of \(\sL\) at \(\theta\).
* \(d \sL(\theta)\): The total derivative of \(\sL\) at \(\theta\).
* \(\nabla_\square \sL(\theta)\): The partial gradient of \(\sL\) at \(\theta\) with respect to \(\square \in \{Q, K, V\}\).
* \(d_\square \sL(\theta)\): The partial derivative of \(\sL\) at \(\theta\) with respect to \(\square \in \{Q, K, V\}\).

Recall that \(\nabla \sL(\theta)\) is the unique element of \(\Theta\) that satisfies
$$
\begin{align*}
    d\sL(\theta) \cdot (\tilde{Q}, \tilde{K}, \tilde{V}) =
    \langle (\tilde{Q}, \tilde{K}, \tilde{V}), \nabla \sL(\theta) \rangle_\Theta.
\end{align*}
$$
Similarly, \(\nabla_\square \sL(\theta)\) is the unique element of \(\bR^{n \times d}\) that satisfies
$$
\begin{align*}
    d_\square \sL(\theta) \cdot \tilde{\square} =
    \langle \tilde{\square}, \nabla_\square \sL(\theta) \rangle_F.
\end{align*}
$$
The (total) gradient and partial gradients are related by
$$
    \nabla \sL(\theta) = (\nabla_Q \sL(\theta), \nabla_K \sL(\theta), \nabla_V \sL(\theta)).
$$
To compute the (total) gradient, we will now compute the partial gradients.

## Partial gradient with respect to \(Q\)

To compute \(\nabla_Q \sL(\theta)\), first observe that
$$
\begin{align*}
    d_Q \sL(\theta) \cdot \tilde{Q}
    &= \left\langle d_Q \sL(\theta) \cdot \tilde{Q}, 1 \right\rangle_\bR \\
    &= \left\langle d \ell(\Attn(\theta)) \cdot d_Q \Attn(\theta) \cdot \tilde{Q}, 1 \right\rangle_\bR \\
    &= \left\langle \tilde{Q}, d_Q \Attn(\theta)^* \circ d \ell(\Attn(\theta))^* \cdot 1 \right\rangle_F \\
    &= \left\langle \tilde{Q}, d_Q \Attn(\theta)^* \cdot \nabla \ell(\Attn(\theta)) \right\rangle_F \\
    &= \left\langle \tilde{Q}, d_Q \Attn(\theta)^* \cdot \Lambda(\theta) \right\rangle_F,
\end{align*}
$$
where \(d_Q \Attn(\theta)\) is the partial derivative of \(\Attn\) at \(\theta\)
with respect \(Q\), the superscript \(*\) denote adjoint, and
$$
    \Lambda(\theta) = \nabla \ell(\Attn(\theta)) \in \bR^{n \times d}.
$$
By the chain rule,
$$
\begin{align*}
    d_Q \Attn(\theta) \cdot \tilde{Q}
    &= \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot K\tilde{Q}^t e_i]^t V.
\end{align*}
$$
Here we have introduced the shorthand \(p_i = K Q^t e_i\). Also, \(d \sigma(x)\) denotes the total derivative of
\(\sigma\) at \(x\), which we identify with the Jacobian matrix of \(\sigma\) at \(x\).

To evaluate the adjoint \(d_Q \Attn(\theta)^*\), compute
$$
\begin{align*}
    \left\langle d_Q \Attn(\theta) \cdot \tilde{Q}, P \right\rangle_F
    &= \left\langle \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot K\tilde{Q}^t e_i]^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i [d \sigma(p_i) \cdot K\tilde{Q}^t e_i]^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i e_i^t \tilde{Q} K^t d \sigma(p_i)^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \tr\left( V^t d \sigma(p_i) K \tilde{Q}^t e_i e_i^t P \right) \\
    &= \sum_{i=1}^{n} \tr\left( \tilde{Q}^t e_i e_i^t P V^t d \sigma(p_i) K \right) &\color{gray}\mbox{(Cyclic invariance)} \\
    &= \sum_{i=1}^{n} \left\langle \tilde{Q}, e_i e_i^t P V^t d \sigma(p_i) K \right\rangle_F \\
    &= \left\langle \tilde{Q}, \sum_{i=1}^{n} e_i e_i^t P V^t d \sigma(p_i) K \right\rangle_F.
\end{align*}
$$
We have shown that
$$
    d_Q \Attn(\theta)^* \cdot P = \sum_{i=1}^{n} e_i e_i^t P V^t d \sigma(p_i) K
$$
and consequently
$$
\colorbox{lesserbox}
{
$
    \nabla_Q \sL(\theta) = \sum_{i=1}^{n} e_i e_i^t \Lambda(\theta) V^t d \sigma(p_i) K.
$
}
$$

## Partial gradient with respect to \(K\)

To compute \(\nabla_K \sL(\theta)\), first observe that
$$
\begin{align*}
    d_K \sL(\theta) \cdot \tilde{K}
    &= \left\langle \tilde{K}, d_K \Attn(\theta)^* \cdot \Lambda(\theta) \right\rangle_F,
\end{align*}
$$
where \(d_K \Attn(\theta)\) is the partial derivative of \(\Attn\) at \(\theta\) with respect to \(K\).

By the chain rule,
$$
\begin{align*}
    d_K \Attn(\theta) \cdot \tilde{K}
    &= \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot \tilde{K} Q^t e_i]^t V.
\end{align*}
$$

To evaluate the adjoint \(d_K \Attn(\theta)^*\), compute
$$
\begin{align*}
    \left\langle d_K \Attn(\theta) \cdot \tilde{K}, P \right\rangle_F
    &= \left\langle \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot \tilde{K} Q^t e_i]^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i [d \sigma(p_i) \cdot \tilde{K} Q^t e_i]^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i e_i^t Q \tilde{K}^t d \sigma(p_i)^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \tr\left( V^t d \sigma(p_i) \tilde{K} Q^t e_i e_i^t P \right) \\
    &= \sum_{i=1}^{n} \tr\left( Q^t e_i e_i^t P V^t d \sigma(p_i) \tilde{K} \right) &\color{gray}\mbox{(Cyclic invariance)} \\
    &= \sum_{i=1}^{n} \tr\left( \tilde{K}^t d\sigma(p_i)^t V P^t e_i e_i^t Q \right) &\color{gray}\mbox{(Transpose invariance)} \\
    &= \sum_{i=1}^{n} \left\langle \tilde{K}, d\sigma(p_i)^t V P^t e_i e_i^t Q \right\rangle_F \\
    &= \left\langle \tilde{K}, \sum_{i=1}^{n} d\sigma(p_i)^t V P^t e_i e_i^t Q \right\rangle_F.
\end{align*}
$$
We have shown that
$$
    d_K \Attn(\theta)^* \cdot P = \sum_{i=1}^{n} d \sigma(p_i)^t V P^t e_i e_i^t Q
$$
and consequently
$$
\colorbox{lesserbox}
{
$
    \nabla_K \sL(\theta) = \sum_{i=1}^{n} d\sigma(p_i)^t V \Lambda(\theta)^t e_i e_i^t Q.
$
}
$$

## Partial gradient with respect to \(V\)

To compute \(\nabla_V \sL(\theta)\), first observe that
$$
\begin{align*}
    d_V \sL(\theta) \cdot \tilde{V}
    &= \left\langle \tilde{V}, d_V \Attn(\theta)^* \cdot \Lambda(\theta) \right\rangle_F,
\end{align*}
$$
where \(d_V \Attn(\theta)\) is the partial derivative of \(\Attn\) at \(\theta\) with respect to \(V\).

Differentiating, we obtain
$$
    d_V \Attn(\theta) \cdot \tilde{V} = \sum_{i=1}^{n} e_i \sigma(p_i)^t \tilde{V}.
$$
To evaluate the adjoint \(d_V \Attn(\theta)^*\), compute
$$
\begin{align*}
    \left\langle d_V \Attn(\theta) \cdot \tilde{V}, P \right\rangle_F
    &= \left\langle \sum_{i=1}^{n} e_i \sigma(p_i)^t \tilde{V}, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i \sigma(p_i)^t \tilde{V}, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle \tilde{V}, \sigma(p_i) e_i^t P \right\rangle_F \\
    &= \left\langle \tilde{V}, \sum_{i=1}^{n} \sigma(p_i) e_i^t P \right\rangle_F.
\end{align*}
$$
We have shown that
$$
    d_V \Attn(\theta)^* \cdot P = \sum_{i=1}^{n} \sigma(p_i) e_i^t P
$$
and consequently
$$
\colorbox{lesserbox}
{
$
    \nabla_V \sL(\theta) = \sum_{i=1}^{n} \sigma(p_i) e_i^t \Lambda(\theta).
$
}
$$

## Main result

Putting everything together, we have
$$
\colorbox{magicmint}
{
$
    \nabla \sL(\theta) =
    \left(
    \Omega(\theta) K,
    \Omega(\theta)^t Q,
    \sum_{i=1}^{n} \sigma(p_i) e_i^t \Lambda(\theta)
    \right),
$
}
$$
where
$$
\colorbox{lesserbox}
{
$
    \Omega(\theta) = \sum_{i=1}^{n} e_i e_i^t \Lambda(\theta) V^t d \sigma(p_i).
$
}
$$
To be more explicit, note that we can write \(\Omega(\theta)\) in "matrix form" as
$$
    \Omega(\theta) =
    \begin{bmatrix}
    e_1^t \Lambda(\theta) V^t d \sigma(p_1) \\
    \vdots \\
    e_n^t \Lambda(\theta) V^t d \sigma(p_n)
    \end{bmatrix}.
$$
Expanding each \(d\sigma(p_i)\) as in [this post]({{<ref "posts/softmax-derivatives">}}),
we obtain
$$
    \Omega(\theta) =
    \begin{bmatrix}
    e_1^t \Lambda(\theta) V^t (\Delta \sigma(p_1) - \sigma(p_1) \sigma(p_1)^t) \\
    \vdots \\
    e_n^t \Lambda(\theta) V^t (\Delta \sigma(p_n) - \sigma(p_n) \sigma(p_n)^t)
    \end{bmatrix}.
$$

## Example

As a simple example, suppose that \(\ell(\theta) = e_j^t \theta \epsilon_k\),
where \(\epsilon_k\) is the \(k\)-th Euclidean basis vector in \(\bR^d\).
In other words, \(\ell(\theta)\) is the \((j,k)\)-th component of \(\theta\).  Then
$$
    \Lambda(\theta) = e_j \epsilon_k^t.
$$
It follows that
$$
\begin{align*}
    \Omega(\theta)
    &= e_j \epsilon_k^t V^t d \sigma(p_j)
\end{align*}
$$
and
$$
\begin{align*}
    \nabla \sL(\theta) =
    \left(
    e_j \epsilon_k^t V^t d \sigma(p_j) K,
    d \sigma(p_j)^t V \epsilon_k e_j^t Q,
    \sigma(p_j) \epsilon_k^t
    \right).
\end{align*}
$$

## Scaled dot-product attention

For completeness, the *scaled dot-product attention map* is defined by
$$
\begin{align*}
    \Attn : \Theta &\to \bR^{n \times d} \\
    \theta &\mapsto \Attn(\theta) = \sigma\left(\frac{QK^t}{\sqrt{d}}\right) V.
\end{align*}
$$
In this case, we have
$$
\colorbox{magicmint}
{
$
    \nabla \sL(\theta) =
    \left(
    \frac{\Omega(\theta) K}{\sqrt{d}},
    \frac{\Omega(\theta)^t Q}{\sqrt{d}},
    \sum_{i=1}^{n} \sigma(p_i) e_i^t \Lambda(\theta)
    \right).
$
}
$$

## Verification

For verification, we can implement \(\Attn\) as a `torch.autograd.Function`, as follows:

```python
import math

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class Attention(Function):
    """Scaled dot-product attention map."""

    @staticmethod
    def forward(ctx: FunctionCtx, theta: Float[Tensor, "n 3d"]) -> Float[Tensor, "n d"]:
        """Compute scaled dot-product attention map output.

        Args:
            ctx (FunctionCtx): Context used to stash data for `backward()`.
            theta (Tensor): Input tensor of shape `(n, 3d)`; `theta = [Q, K, V]`.

        Raises:
            ValueError: If the shape of `theta` is invalid (it is not a 2-dimensional
                tensor or its last dimension is not divisible by 3).
        """
        if theta.dim() != 2 or theta.shape[-1] % 3 != 0:
            raise ValueError(f"theta has invalid shape {theta.shape}")
        d = theta.shape[-1] // 3
        q, k, v = theta.split(d, dim=-1)
        s = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), dim=-1)
        ctx.save_for_backward(q, k, v, s)
        return s @ v

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: FunctionCtx, grad_output: Float[Tensor, "n d"]
    ) -> Float[Tensor, "n 3d"]:
        """Compute gradient of scaled dot-product attention map.

        Args:
            ctx (FunctionCtx): Context used to retrieve stashed data from `forward()`.
            grad_output (Tensor): Gradient tensor of shape `(n, 3d)`.
        """
        q, k, v, s = ctx.saved_tensors  # type: ignore[attr-defined]
        n, d = q.shape
        v_transpose = v.transpose(-1, -2)

        # Compute "omega" matrix, row-by-row
        omega = torch.zeros(n, n, dtype=torch.double)
        for i in range(n):
            s_row = s[i, :].unsqueeze(0)
            s_col = s_row.transpose(-1, -2)
            dsigma = torch.diag(s_col.squeeze()) - (s_col @ s_row)
            omega[i, :] = grad_output[i, :].unsqueeze(0) @ v_transpose @ dsigma

        # Incorporate scaling factor
        omega = (1.0 / math.sqrt(d)) * omega

        # Compute "Q" component
        q_comp = omega @ k

        # Compute "K" component
        k_comp = omega.transpose(-1, -2) @ q

        # Compute "V" component
        v_comp = torch.zeros(n, d, dtype=torch.double)
        for i in range(n):
            s_col = s[i, :].unsqueeze(0).transpose(-1, -2)
            v_comp += s_col @ grad_output[i, :].unsqueeze(0)

        # Gradient concatenates all components
        return torch.cat((q_comp, k_comp, v_comp), dim=-1)
```

Then we can compare numerical to analytical gradients, using `gradcheck()`:

```python
import torch
from torch.autograd import gradcheck

from attention import Attention


def test_attention_gradient() -> None:
    """Check correctness of `Attention.backward()` using `gradcheck()`."""
    n, d = 8, 16

    q = torch.randn(n, d, dtype=torch.double, requires_grad=True)
    k = torch.randn(n, d, dtype=torch.double, requires_grad=True)
    v = torch.randn(n, d, dtype=torch.double, requires_grad=True)

    theta = torch.cat((q, k, v), dim=-1)

    if gradcheck(Attention.apply, theta, eps=1e-6, atol=1e-4):
        print("success!")
    else:
        print("failure...")
```

The results are...

```console
$ python -m pytest .
success!
```

This code is hosted on GitHub [here](https://github.com/dscamiss/attention-gradient/).
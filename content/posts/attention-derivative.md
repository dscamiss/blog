+++
title = 'Modified attention maps with simpler total derivatives'
date = 2024-08-09T13:27:04-07:00
draft = false
tag = ['attention', 'backpropagation', 'random-notes']
+++

$$
    \newcommand{\Attn}{\mathrm{Att}}
    \newcommand{\bR}{\mathbb{R}}
$$

In this post, we show that a modified attention map, which
has softmax replaced by a different normalizing map,
has a simpler total derivative.  The goal is to improve
the computational efficiency of backpropagation through
attention-type maps.

## Standard attention

The *standard attention map* is defined by
$$
\begin{align*}
    \Attn : \bR^{n \times d} \times \bR^{n \times d} \times \bR^{n \times d} &\to \bR^{n \times d} \\
    (Q, K, V) &\mapsto \Attn(Q, K, V) = \sigma(QK^t) V,
\end{align*}
$$
where \(\sigma : \bR^n \to \bR^n\) is the [softmax map]({{< ref "softmax-derivatives" >}})
applied row-wise.  This means that
$$
\begin{align*}
    \Attn(Q, K, V) &= \sum_{i=1}^{n} e_i \sigma((e_i^t Q K^t)^t)^t V \\
    &= \sum_{i=1}^{n} e_i \sigma(K Q^t e_i)^t V,
\end{align*}
$$
where \(e_i\) is the \(i\)th Euclidean basis vector in \(\bR^n\).  This
expression could be written in matrix form, of course,
but it seems cleaner to put everything on a single line.

Note: Our definition of the attention map does not scale the entries of \(QK^t\)
by \(1/\sqrt{d}\).  This is just for convenience, to avoid writing the square root
everywhere.

The total derivative of \(\Attn\) is
$$
\begin{align*}
    d \Attn(\Theta) \cdot \tilde{\Theta} =
    d_Q \Attn(\Theta) \cdot \tilde{Q} +
    d_K \Attn(\Theta) \cdot \tilde{K} +
    d_V \Attn(\Theta) \cdot \tilde{V},
\end{align*}
$$
where \(\Theta = (Q,K,V)\) and \(\tilde{\Theta} = (\tilde{Q},\tilde{K},\tilde{V})\).

By the Leibniz rule, the partial derivatives of \(\Attn\) are
$$
\begin{align*}
    d_Q \Attn(\Theta) \cdot \tilde{Q}
    &= \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot \tilde{q}_i]^t V \\
    d_K \Attn(\Theta) \cdot \tilde{K}
    &= \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot \tilde{k}_i]^t V \\
    d_V \Attn(\Theta) \cdot \tilde{V}
    &= \sum_{i=1}^{n} e_i \sigma(p_i)^t \tilde{V} = \Attn(Q, K, \tilde{V}),
\end{align*}
$$
where \(p_i = K Q^t e_i\), \(\tilde{q}_i = K \tilde{Q}^t e_i\), and
\(\tilde{k}_i = \tilde{K} Q^t e_i\).  Combining terms, we have
$$
\begin{align*}
    d \Attn(\Theta) \cdot \tilde{\Theta}
    &= \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot \tilde{z}_i]^t V
    + \Attn(Q, K, \tilde{V}),
\end{align*}
$$
where \(\tilde{z}_i = \tilde{q}_i + \tilde{k}_i\).
Using the [well-known formula]({{< ref "softmax-derivatives" >}}) for
\(d \sigma\), we have
$$
\begin{align*}
    &{} \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot \tilde{z}_i]^t V \\
    &\qquad = \sum_{i=1}^{n} e_i (\sigma(p_i) \odot \tilde{z}_i)^t V -
    \sum_{i=1}^{n} \langle \sigma(p_i), \tilde{z}_i \rangle e_i \sigma(p_i)^t V \\
    &\qquad = \sum_{i=1}^{n} e_i (\sigma(p_i) \odot \tilde{z}_i)^t V -
    (\tilde{\iota}(Q,K) \otimes 1_n^t) \odot \Attn(Q,K,V),
\end{align*}
$$
where \(\odot\) is the element-wise product and
\(\tilde{\iota}(Q,K) \otimes 1_n^t\) is the Kronecker product of
$$
\tilde{\iota}(Q,K) =
\begin{bmatrix}
    \langle \sigma(p_1), \tilde{z}_1 \rangle \\
    \vdots \\
    \langle \sigma(p_n), \tilde{z}_n \rangle
\end{bmatrix}
\qquad \mbox{and} \qquad
1_n^t = (1, \dots, 1).
$$
In total, we have
$$
\begin{align*}
    d\Attn(\Theta) \cdot \tilde{\Theta}
    &= \sum_{i=1}^{n} e_i ({\color{cornflowerblue}\sigma(p_i)} \odot \tilde{z}_i)^t V
    - (\underbrace{\tilde{\iota}(Q,K)}_{\color{cornflowerblue}\sigma(p_i)}
     \otimes 1_n^t) \odot {\color{cornflowerblue}\Attn(\Theta)}
    + \underbrace{\Attn(Q,K,\tilde{V})}_{\color{cornflowerblue}\sum_{i=1}^{n} e_i \sigma(p_i)^t}.
\end{align*}
$$
The quantities in blue can be re-used from the forward pass.

In the next section, we consider how the above result changes if we replace
\(\sigma\).

## Other normalizing maps

In this section, \(\beta: \bR^n \to \bR^n\) is a smooth map with
the following homogeneity property: There exists a smooth function
\(f : \bR^n \times \bR^n \to \bR\) such that
$$
\begin{align*}
    d \beta(x) \cdot h = f (x, h) \beta(x),
\end{align*}
$$

Replacing \(\sigma\) with \(\beta\), the *\(\beta\)-attention map* is defined by
$$
\begin{align*}
    \Attn_\beta : \bR^{n \times d} \times \bR^{n \times d} \times \bR^{n \times d} &\to \bR^{n \times d} \\
    (Q, K, V) &\mapsto \Attn_\beta(Q, K, V) = \beta(QK^t) V,
\end{align*}
$$
where \(\beta\) is applied row-wise.

Repeating the analysis of the previous section, we have
$$
\begin{align*}
    \Attn_\beta (\Theta) &= \sum_{i=1}^{n} e_i \beta(p_i)^t V
\end{align*}
$$
and the total derivative of \(\Attn_\beta\) at \(\Theta\) is
$$
\begin{align*}
    d \Attn_\beta (\Theta) \cdot \tilde{\Theta}
    &= \sum_{i=1}^{n} e_i [d \beta(p_i) \cdot \tilde{z}_i]^t V
    + \Attn_\beta (Q, K, \tilde{V}) \\
    &= \sum_{i=1}^{n} f(p_i, \tilde{z}_i) e_i \beta(p_i)^t V
    + \Attn_\beta (Q, K, \tilde{V}) \\
    &= (\tilde{f}(Q,K) \otimes 1_n^t) \odot \Attn_\beta (\Theta)
    + \Attn_\beta (Q, K, \tilde{V}),
\end{align*}
$$
where \(p_i\), \(\tilde{z}_i\) are defined as in the previous section and
$$
\tilde{f}(Q,K) =
\begin{bmatrix}
    f(p_1, \tilde{z}_1) \\
    \vdots \\
    f(p_n, \tilde{z}_n)
\end{bmatrix}.
$$
In total, we have
$$
\begin{align*}
    d \Attn_\beta (\Theta) \cdot \tilde{\Theta}
    &= (\underbrace{\tilde{f}(Q,K)}_{\color{cornflowerblue}p_i} \otimes 1_n^t) \odot
    {\color{cornflowerblue}\Attn_\beta (\Theta)}
    + \underbrace{\Attn_\beta(Q,K,\tilde{V})}_{\color{cornflowerblue}\sum_{i=1}^{n} e_i \beta(p_i)^t}.
\end{align*}
$$
The quantities in blue can be re-used from the forward pass.

The punchline is that the homogeneity property could make the computation
of \(d\Attn_\beta(\Theta) \cdot \tilde{\Theta}\) more efficient
by "removing" the first term of \(d\Attn(\Theta)\cdot \tilde{\Theta}\).

For a particular example, consider the simple normalizing map
$$
\begin{align*}
    \beta(x) = \frac{x}{1 + \| x \|}.
\end{align*}
$$
The total derivatives of \(\beta\) are
$$
\begin{align*}
    d\beta(x) \cdot h
    &= \begin{cases}
    \displaystyle
    \frac{h}{1 + \|x\|}
    - \frac{\langle x, h \rangle x}{\|x\| (1 + \|x\|)^2}, & x \neq 0_n \\
    h, & x = 0_n.
    \end{cases}
\end{align*}
$$
By definition, \(f\) must satisfy
$$
\begin{align*}
    f(x, h) \frac{x}{1 + \| x\|}
    &= \begin{cases}
    \displaystyle
    \frac{h}{1 + \|x\|} - \frac{\langle x, h \rangle x}{\| x\| (1 + \|x\|)^2}, & x \neq 0_n \\
    h, & x = 0_n.
    \end{cases}
\end{align*}
$$
This is clearly not possible for \(x = 0_n\), so from this point forward
let's assume that we are working away from \(x = 0_n\) (this can be made rigorous,
but we'll skip that).

Rearranging terms, we see that
$$
\begin{align*}
    f(x, h)
    &= \frac{\langle x, h \rangle}{\|x\|^2}
    - \frac{\langle x, h \rangle}{\| x\| (1 + \|x\|)} \\
    &= \frac{\langle x, h \rangle}{\|x\|^2 (1 + \|x\|)}.
  \end{align*}
$$

## Plausibility of replacing \(\sigma\)

Putting aside potential efficiency gains, can we learn effectively with \(\Attn_\beta\)?

To quickly test this for
$$
\begin{align*}
    \beta(x) = \frac{x}{1 + \| x \|},
\end{align*}
$$
we build on the [`nanoGPT`](https://github.com/karpathy/nanoGPT "nanoGPT hosted on GitHub")
project.  In a nutshell, we need to implement \(\beta\), disable flash attention, and
adjust the causal masking logic to accommodate \(\beta\).

The `nn.Module` that implements \(\beta\) is very straightforward:
```
"""Implementation of beta map."""

import torch
import torch.nn as nn
from torch import Tensor

class Beta(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Compute beta() along last dimension of x."""
        return x / (1.0 + torch.norm(x, dim=-1, keepdim=True))
```
Training character-level Tiny Shakespeare for 10000 iterations with `nanoGPT` defaults:
```
step 10000: train loss 1.0040, val loss 1.5550
```
The final validation loss is comparable to the results obtained using standard attention:
```
step 10000: train loss 0.7140, val loss 1.6204
```
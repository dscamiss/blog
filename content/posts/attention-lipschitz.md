+++
title = 'Scaled dot-product self-attention is not Lipschitz'
date = 2025-01-08T08:59:04-07:00
draft = true
tag = ['attention', 'lipschitz', 'random-notes']
+++

## Introduction

In this post, we prove that the scaled dot-product self-attention
map is not Lipschitz for unbounded inputs.  This fact is
already known in the literature, for example as [1, Theorem 3.1].
Our proof is more straightforward, since it avoids computing individual
components of the self-attention map's Jacobian matrix.

<!--more-->

$$
    \newcommand{\Att}{\mathrm{Att}}
    \newcommand{\bR}{\mathbb{R}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Scaled dot-product self-attention

Throughout, \(W_Q, W_K \in \bR^{d \times d_k}\) and \(W_V \in \bR^{d \times d_v}\)
are matrices such that \(W_Q W_K^t \neq 0\).

The *scaled dot-product self-attention map* is defined by
$$
\begin{align*}
    \Att : \bR^{n \times d} &\to \bR^{n \times d_v} \\
    X &\mapsto \Att(X) = \vec{\sigma}
    \left(
        \frac{X W_Q (X W_K)^t}{\sqrt{d_k}}
    \right) X W_V,
\end{align*}
$$
where \(\vec{\sigma} : \bR^{n \times n} \to \bR^{n \times n}\) is the standard
[softmax map]({{< ref "softmax-derivatives" >}}) applied row-wise.

To keep the notation concise, we write
$$
    A = \frac{W_Q W_K^t}{\sqrt{d}} \neq 0 \qquad \mbox{and} \qquad
    P \equiv P(X) = X A X^t.
$$
Using this shorthand, we have
$$
    \Att(X) = \vec{\sigma}(P) X W_V
$$
or more explicitly
$$
    \Att(X) = \begin{bmatrix}
        \sigma(P^t e_1)^t \\
        \vdots \\
        \sigma(P^t e_n)^t
    \end{bmatrix} X W_V,
$$
where \(\sigma : \bR^n \to \bR^n\) is the standard softmax map and \(e_i\)
is the \(i\)th Euclidean basis vector.  Note
that the transpose operations are necessary to ensure that (1) the inputs
to \(\sigma\) are column vectors and (2) the outputs of \(\sigma\)
are stacked up as row vectors.

Using the chain rule, the total derivative of \(\Att\) is
$$
    d \Att(X) \cdot \tilde{X}
    = M X W_V
    + \vec{\sigma}(P) \tilde{X} W_V,
$$
where
$$
    M = \begin{bmatrix}
        \left( d \sigma(P^t e_1)^t \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_1 \right)^t \\
        \vdots \\
        \left( d \sigma(P^t e_n)^t \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_n \right)^t \\
    \end{bmatrix}.
$$

## Lipschitz considerations

Set \(r_i\) to be the transpose of the \(i\)th row of \(M\), so that
$$
    r_i = d \sigma(P^t e_i)^t \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_i.
$$
By assumption \(A \neq 0_{n \times n}\), so we can choose \(\tilde{X}\) and an index \(i\) such that
$$
    v = A^t \tilde{X}^t e_i \neq 0_{n \times 1}.
$$
If \(X\) is such that \(X^t e_i = 0_{d \times 1}\), then
$$
    P^t e_i = X A^t X^t e_i = 0_{n \times 1}.
$$
From this, it follows that
$$
\begin{align*}
    s(P^t e_i) &= \frac{1}{n} 1_{n \times 1} \\
    ds(P^t e_i) &= \frac{1}{n} I_n - \frac{1}{n^2} 1_{n \times 1} 1_{n \times 1}^t,
\end{align*}
$$
where \(I_n\) is the \(n\)-dimensional identity matrix.
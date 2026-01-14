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
are matrices such that \(W_Q W_K^t \neq 0\).  For reasons that will be made
clear below, we also assume that \(n > 1\).

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

By the chain rule, the total derivative of \(\Att\) is
$$
    d \Att(X) \cdot \tilde{X}
    = M(X, \tilde{X}) + \vec{\sigma}(P) \tilde{X} W_V,
$$
where
$$
    M(X, \tilde{X}) = \begin{bmatrix}
        \left( d \sigma(P^t e_1) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_1 \right)^t X W_V \\
        \vdots \\
        \left( d \sigma(P^t e_n) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_n \right)^t X W_V \\
    \end{bmatrix}.
$$

## Lipschitz considerations

The \(i\)th row of \(M(X, \tilde{X})\) is
$$
    M(X, \tilde{X})_i =
    \left( d \sigma(P^t e_i) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_i \right)^t X W_V.
$$
Suppose that we have chosen
* \(\tilde{X}\) such that \(v = A^t \tilde{X}^t e_i \neq \mathbf{0}\), and
* \(X\) such that \(X^t e_i = \mathbf{0}\).

Then \(P^t e_i = X A^t X^t e_i = \mathbf{0}\) and consequently
$$
\begin{align*}
    M(X, \tilde{X})_i
    &= \left(
        d \sigma(\mathbf{0}) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_i
    \right)^t X W_V \\
    &= \left(
        d \sigma(\mathbf{0}) \cdot X A^t \tilde{X}^t e_i
    \right)^t X W_V \\
    &= \left(
        d \sigma(\mathbf{0}) \cdot X v
    \right)^t X W_V \\
    &= \left(
    \left(
        \frac{1}{n} I_n - \frac{1}{n^2} \mathbf{1} \mathbf{1}^t
    \right) Xv
    \right)^t X W_V,
\end{align*}
$$
where \(I_n\) is the \(n\)-dimensional identity matrix.  Here we have
used the fact that
$$
    d\sigma(x) \cdot \tilde{x} = (\mathrm{diag}(x)
    - \sigma(x) \sigma(x)^t) \tilde{x}.
$$

In particular, suppose that
$$
    X = C e_j \frac{v^t}{\| v \|^2},
$$
where \(C > 0\) and \(i \neq j\) (note that such an index \(j\)
exists since \(n > 1\)).

Clearly \(X\) satisfies \(X^t e_i = \mathbf{0}\) and
$$
\begin{align*}
    M(X, \tilde{X})_i
    &= \left(
        \left(
        \frac{1}{n} I_n - \frac{1}{n^2} \mathbf{1} \mathbf{1}^t
        \right)
        C e_j \frac{v^t v}{\| v \|^2}
        \right)^t C e_j \frac{v^t}{\| v \|^2} W_V \\
    &= C^2 e_j^t
        \left(
        \frac{1}{n} I_n - \frac{1}{n^2} \mathbf{1} \mathbf{1}^t
        \right)
        e_j \frac{v^t}{\| v \|^2} W_V \\
    &= \frac{C^2}{\| v \|^2}
        \left(
        \frac{1}{n} - \frac{1}{n^2}
        \right)
        v^t W_V.
\end{align*}
$$

## Main result

*Definition*: We say that \(v\) is a *coupling vector* if it satisfies:
* \(v \neq \mathbf{0}\),
* \(v = A^t u\) with \(\| u \|_\infty = 1\), and
* \(v^t W_V \neq \mathbf{0}\).

<!--
"If"     : A W_V != 0 ==> There exists a coupling vector
"Only if": There exists a coupling vector ==> A W_V != 0
           <==> A W_V = 0 ==> There does not exist a coupling vector
-->

*Lemma*: There exists a coupling vector \(v\) if and only if \(A W_V \neq \mathbf{0}\).

*Proof*: For the "if" direction, suppose that \(A W_V \neq \mathbf{0}\).
Choose \(i\) such that \(c = A W_V e_i \neq \mathbf{0}\) and set
\(v = A^t u\), where \(u = c / \| c \|_\infty\).
To see that \(v \neq \mathbf{0}\), assume to the contrary that \(v = \mathbf{0}\).
Then \(c^t c = c^t A W_V e_i = \| c \|_\infty v^t W_V e_i = 0\), which contradicts the
fact that \(c \neq \mathbf{0}\).  By construction, \(v = A^t u\) with \(\| u \|_\infty = 1\).
Finally, to see that \(v^t W_V \neq \mathbf{0}\), assume to the contrary that
\(v^t W_V = \mathbf{0}\).  Arguing as above, this implies that \(c^t c = 0\),
which contradicts the fact that \(c \neq \mathbf{0}\).  For the "only if"
direction, suppose that \(A W_V = \mathbf{0}\) and \(v = A^t u\) is a coupling
vector.  Then \(v^t W_V = u^t A W_V = \mathbf{0}\), hence no coupling vector exists.
\(\blacksquare\)

*Lemma*: Fix a coupling vector \(v\). Then for each \(C > 0\)
there exist \(X, \tilde{X} \in \bR^{n \times d}\) such that
\(\| \tilde{X} \|_\infty = 1\) and
$$
    \| M(X, \tilde{X}) \|_\infty \geq
    \frac{C^2}{\| v \|_2^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \|_\infty > 0.
$$

*Proof*: By definition,
\(v = A^t u \) with \(\| u \|_\infty = 1\).  Clearly, we can choose
\(\tilde{X} \in \bR^{n \times d}\) such that \(\| \tilde{X} \|_\infty = 1\) and \(\tilde{X}^t e_i = u\),
where \(e_i\) is the \(i\)th standard basis vector in \(\bR^n\).
Now define
$$
    X = C e_j \frac{v^t}{\| v \|_2^2} \in \bR^{n \times d},
$$
where \(C > 0\) and \(i \neq j\) (note that such an index \(j\)
exists since \(n > 1\)).  Since \(X^t e_i = \mathbf{0}\), we have
\(P^t e_i = X A^t X^t e_i = \mathbf{0}\) and consequently the
\(i\)th row of \(M(X, \tilde{X})\) is
$$
\begin{align*}
    M(X, \tilde{X})_i
    &= \left(
        d \sigma(\mathbf{0}) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_i
    \right)^t X W_V \\
    &= \left(
        d \sigma(\mathbf{0}) \cdot X A^t \tilde{X}^t e_i
    \right)^t X W_V \\
    &= \left(
        d \sigma(\mathbf{0}) \cdot X A^t u
    \right)^t X W_V \\
    &= \left(
        d \sigma(\mathbf{0}) \cdot X v
    \right)^t X W_V \\
    &= \left(
    \left(
        \frac{1}{n} I_n - \frac{1}{n^2} \mathbf{1} \mathbf{1}^t
    \right) Xv
    \right)^t X W_V \\
    &= \left(
        \left(
        \frac{1}{n} I_n - \frac{1}{n^2} \mathbf{1} \mathbf{1}^t
        \right)
        C e_j \frac{v^t v}{\| v \|_2^2}
        \right)^t C e_j \frac{v^t}{\| v \|_2^2} W_V \\
    &= C^2 e_j^t
        \left(
        \frac{1}{n} I_n - \frac{1}{n^2} \mathbf{1} \mathbf{1}^t
        \right)
        e_j \frac{v^t}{\| v \|_2^2} W_V \\
    &= \frac{C^2}{\| v \|_2^2}
        \left(
        \frac{1}{n} - \frac{1}{n^2}
        \right)
        v^t W_V.
\end{align*}
$$
where \(I_n\) is the \(n\)-dimensional identity matrix.  For the
lower bound, observe that
$$
\begin{align*}
    \| M(X, \tilde{X}) \|_\infty
    &= \max_{1 \leq k \leq n} \| M(X, \tilde{X})_k \|_\infty \\
    &\geq \| M(X, \tilde{X})_i \|_\infty \\
    &= \left\| \frac{C^2}{\| v \|_2^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) v^t W_V \right\|_\infty \\
    &= \frac{C^2}{\| v \|_2^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \|_\infty.
\end{align*}
$$
To conclude, note that the latter expression is positive since \(v^t W_V \neq \mathbf{0}\).
\(\blacksquare\)

In what follows, we assume that all matrix spaces \(\bR^{p \times q}\) have the
\(\infty\)-norm.

<!--
A W_V != 0 ==> Att not Lipschitz
<==>
-->

*Theorem*: If \(A W_V \neq \mathbf{0}\), then \(\Att\) is not Lipschitz.

*Proof*: Suppose that \(A W_V \neq \mathbf{0}\).  Assume that \(\Att\)
has Lipschitz constant \(L\), so that
$$
\begin{align*}
    L &= \sup_{X \in \bR^{n \times d}} \| d \Att (X) \|_{\mathrm{op}} \\
    &= \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\|_\infty = 1}
     \| d \Att(X) \cdot \tilde{X} \|_\infty \\
    &= \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\|_\infty = 1} \| M(X, \tilde{X}) + \vec{\sigma}(P) \tilde{X} W_V \|_\infty \\
    &\geq \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\|_\infty = 1} | \| M(X, \tilde{X}) \|_\infty - \| \vec{\sigma}(P) \tilde{X} W_V \|_\infty | \\
    &\geq \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\|_\infty = 1} \| M(X, \tilde{X}) \|_\infty - \| \vec{\sigma}(P) \tilde{X}  W_V \|_\infty \\
    &\geq \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\|_\infty = 1} \| M(X, \tilde{X}) \|_\infty - \| W_V \|_\infty, \qquad (1)
\end{align*}
$$
where in the last line we have used the fact that
$$
\| \vec{\sigma}(P) \tilde{X} W_V \|_\infty \leq
\| \vec{\sigma}(P) \|_\infty \, \| \tilde{X} \|_\infty \, \| W_V \|_\infty = \| W_V \|_\infty.
$$
Now, since \(A W_V \neq \mathbf{0}\), we can fix a coupling vector \(v\).
Furthermore, for each \(C > 0\), there exist \(X, \tilde{X} \in \bR^{n \times d}\)
such that \(\| \tilde{X} \|_\infty = 1\) and
$$
    \| M(X, \tilde{X}) \|_\infty \geq
    \frac{C^2}{\| v \|_2^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \|_\infty > 0.
$$
Consequently, for each \(C > 0\), we have
$$
    \sup_{X \in \bR^{n \times d}} \| d \Att (X) \|_{\mathrm{op}}
    \geq
    \frac{C^2}{\| v \|_2^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \|_\infty - \| W_V \|_\infty.
$$
Since the right-hand side can be made arbitrarily large, \(\Att\) is not Lipschitz.
\(\blacksquare\)
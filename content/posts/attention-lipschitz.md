+++
title = 'Scaled dot-product self-attention is not Lipschitz'
date = 2025-01-08T08:59:04-07:00
draft = true
tag = ['attention', 'lipschitz', 'random-notes']
+++

## Introduction

In this post, we prove that the scaled dot-product self-attention
map is not Lipschitz (for unbounded inputs).  This fact is
already known in the literature, for example as [1, Theorem 3.1].
The proof presented here differs in two important ways.  First, the proof
is component-free, in the sense that we avoid computing
individual components of Jacobian matrices.
Second, we more precisely identify the parameters for which the
self-attention map is not Lipschitz.

<!--more-->

$$
    \newcommand{\Att}{\mathrm{Att}}
    \newcommand{\bR}{\mathbb{R}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Scaled dot-product self-attention

In this section, \(W_Q, W_K \in \bR^{d \times d_K}\), \(W_V \in \bR^{d \times d_V}\),
and \(n > 1\).

The *scaled dot-product self-attention map* [2] is defined by
$$
\begin{align*}
    \Att : \bR^{n \times d} &\to \bR^{n \times d_V} \\
    X &\mapsto \Att(X) = \vec{\sigma}
    \left(
        \frac{X W_Q (X W_K)^t}{\sqrt{d_K}}
    \right) X W_V,
\end{align*}
$$
where \(\vec{\sigma} : \bR^{n \times n} \to \bR^{n \times n}\) is the standard
softmax map applied row-wise.
We say that the matrices \(W_Q, W_K, W_V\) are the *parameters* of \(\Att\).
To keep the notation concise, we write
$$
    A = \frac{W_Q W_K^t}{\sqrt{d_K}} \qquad \mbox{and} \qquad
    P(X) = X A X^t.
$$
With this shorthand, we have
$$
    \Att(X) = \vec{\sigma}(P(X)) X W_V
$$
or more explicitly
$$
    \Att(X) = \begin{bmatrix}
        \sigma(P(X)^t e_1)^t \\
        \vdots \\
        \sigma(P(X)^t e_n)^t
    \end{bmatrix} X W_V,
$$
where \(\sigma : \bR^n \to \bR^n\) is the standard softmax map and \(e_i\)
is the \(i\)th standard basis vector in \(\bR^n\).  Note
that the transpose operations are necessary to ensure that (1) the inputs
to \(\sigma\) are column vectors and (2) the outputs of \(\sigma\)
are stacked up as row vectors.

By the chain rule, the total derivative of \(\Att\) is
$$
    d \Att(X) \cdot \tilde{X}
    = M(X, \tilde{X}) + \vec{\sigma}(P(X)) \tilde{X} W_V,
$$
where
$$
    M(X, \tilde{X}) = \begin{bmatrix}
        \left( d \sigma(P(X)^t e_1) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_1 \right)^t \\
        \vdots \\
        \left( d \sigma(P(X)^t e_n) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_n \right)^t \\
    \end{bmatrix} X W_V.
$$
In the next section, we will use the well-known fact that
$$
    d \sigma(x) = \mathrm{diag}(\sigma(x)) - \sigma(x) \sigma(x)^t,
$$
where \(\mathrm{diag}(x)\) is the diagonal matrix whose \((i,i)\)th entry is
\(x^i\).

## Main result

All norms in this section are Frobenius norms.

### Scaled dot-product self-attention is not Lipschitz

*Definition*: We say that \(v \in \bR^{d}\) is a *coupling vector for \(\Att\)* if it satisfies:
* \(v \neq \mathbf{0}\),
* \(v = A^t u\) with \(\| u \| = 1\), and
* \(v^t W_V \neq \mathbf{0}\).

<!--
"If"     : A W_V != 0 ==> There exists a coupling vector
"Only if": There exists a coupling vector ==> A W_V != 0
           <==> A W_V = 0 ==> There does not exist a coupling vector
-->

*Lemma*: There exists a coupling vector for \(\Att\) if and only if \(A W_V \neq \mathbf{0}\).

*Proof*: For the "if" direction, suppose that \(A W_V \neq \mathbf{0}\).
Choose \(i\) such that \(c = A W_V e_i' \neq \mathbf{0}\), where \(e_i'\) is
the \(i\)th standard basis vector in \(\bR^{d_V}\), and set
\(v = A^t u\), where \(u = c \| c \|^{-1}\).
To see that \(v \neq \mathbf{0}\), assume to the contrary that \(v = \mathbf{0}\).
Then
$$
    \| c \|^2 = c^t c = c^t A W_V e_i' = \| c \| v^t W_V e_i' = 0,
$$
which contradicts the
fact that \(c \neq \mathbf{0}\).  By construction, \(v = A^t u\) with \(\| u \| = 1\).
Finally, to see that \(v^t W_V \neq \mathbf{0}\), assume to the contrary that
\(v^t W_V = \mathbf{0}\).  Arguing as above, this implies that \(\| c \|^2 = 0\),
which contradicts the fact that \(c \neq \mathbf{0}\).  For the "only if"
direction, suppose that \(A W_V = \mathbf{0}\) and \(v = A^t u\) is a coupling
vector.  Then \(v^t W_V = u^t A W_V = \mathbf{0}\), hence no coupling vector exists.
\(\blacksquare\)

*Lemma*: If \(v\) is a coupling vector for \(\Att\), then for each \(C > 0\),
there exist \(X, \tilde{X} \in \bR^{n \times d}\) such that
\(\| \tilde{X} \| = 1\) and
$$
    \| M(X, \tilde{X}) \| \geq
    \frac{C^2}{\| v \|^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \| > 0.
$$

*Proof*: By definition,
\(v = A^t u \) with \(\| u \| = 1\).  For \(C > 0\), define
$$
    X = C e_i \frac{v^t}{\| v \|^2} \in \bR^{n \times d}
$$
and let \(\tilde{X} \in \bR^{n \times d}\) be the matrix
with \(u\) in the \(j\)th column and zeros elsewhere, where \(i \neq j\)
(note that such a \(j\)
exists since \(n > 1\)).  By construction, \(\| \tilde{X} \| = 1\).
Since \(X^t e_j = \mathbf{0}\), we have
\(P(X)^t e_j = X A^t X^t e_j = \mathbf{0}\) and thus the
\(j\)th row of \(M(X, \tilde{X})\) is
$$
\begin{align*}
    M(X, \tilde{X})_j
    &= \left(
        d \sigma(\mathbf{0}) \cdot (X A^t \tilde{X}^t + \tilde{X} A^t X^t) e_j
    \right)^t X W_V \\
    &= \left(
        d \sigma(\mathbf{0}) \cdot X A^t \tilde{X}^t e_j
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
        C e_i \frac{v^t v}{\| v \|^2}
        \right)^t C e_i \frac{v^t}{\| v \|^2} W_V \\
    &= C^2 e_i^t
        \left(
        \frac{1}{n} I_n - \frac{1}{n^2} \mathbf{1} \mathbf{1}^t
        \right)
        e_i \frac{v^t}{\| v \|^2} W_V \\
    &= \frac{C^2}{\| v \|^2}
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
    \| M(X, \tilde{X}) \|
    &\geq \| M(X, \tilde{X})_j \| \\
    &= \left\| \frac{C^2}{\| v \|^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) v^t W_V \right\| \\
    &= \frac{C^2}{\| v \|^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \|.
\end{align*}
$$
To conclude, note that the latter expression is positive. \(\blacksquare\)

<!--
A W_V != 0 ==> Att not Lipschitz
<==>
-->

*Theorem 1*: If \(A W_V \neq \mathbf{0}\), then \(\Att\) is not Lipschitz.

*Proof*: To prove that \(\Att\) is not Lipschitz, we can prove that the supremum
$$
    \sup_{X \in \bR^{n \times d}} \| d \Att (X) \|_{\mathrm{op}}
$$
does not exist.  To this end, observe that
$$
\begin{align*}
    \sup_{X \in \bR^{n \times d}} \| d \Att (X) \|_{\mathrm{op}}
    &= \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\| = 1}
     \| d \Att(X) \cdot \tilde{X} \| \\
    &= \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\| = 1} \| M(X, \tilde{X}) + \vec{\sigma}(P(X)) \tilde{X} W_V \| \\
    &\geq \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\| = 1} | \| M(X, \tilde{X}) \| - \| \vec{\sigma}(P(X)) \tilde{X} W_V \| | \\
    &\geq \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\| = 1} \| M(X, \tilde{X}) \| - \| \vec{\sigma}(P(X)) \tilde{X}  W_V \| \\
    &\geq \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\| = 1} \| M(X, \tilde{X}) \| - \sqrt{n} \| W_V \|, \qquad (1)
\end{align*}
$$
where in the last line we have used the fact that
$$
\| \vec{\sigma}(P(X)) \tilde{X} W_V \| \leq
\| \vec{\sigma}(P(X)) \| \, \| \tilde{X} \| \, \| W_V \| \leq \sqrt{n} \| W_V \|.
$$
Now, since \(A W_V \neq \mathbf{0}\), we can fix a coupling vector \(v\).
By the preceding lemma, for each \(C > 0\), there exist \(X, \tilde{X} \in \bR^{n \times d}\)
such that \(\| \tilde{X} \| = 1\) and
$$
    \| M(X, \tilde{X}) \| \geq
    \frac{C^2}{\| v \|^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \| > 0.
$$
Consequently, for each \(C > 0\) we have
$$
\begin{align*}
    \sup_{X \in \bR^{n \times d}} \| d \Att (X) \|_{\mathrm{op}}
    &\geq
    \sup_{X, \tilde{X} \in \bR^{n \times d}, \, \|\tilde{X}\| = 1} \| M(X, \tilde{X}) \| - \sqrt{n} \| W_V \| \\
    &\geq
    \frac{C^2}{\| v \|^2} \left( \frac{1}{n} - \frac{1}{n^2} \right) \| v^t W_V \| - \sqrt{n} \| W_V \|.
\end{align*}
$$
Since the right-hand side is arbitrarily large, the supremum does not exist.
\(\blacksquare\)

## Extension to multi-head attention

Suppose that we have scaled dot-product attention maps
$$
    \Att^i : \bR^{n \times d} \to \bR^{n \times d_V}, \quad 1 \leq i \leq h,
$$
where \(\Att^i\) has parameters \(W_Q^i, W_K^i\ \in \bR^{d \times d_K}\) and
\(W_V^i \in \bR^{d \times d_V}\). We define
$$
    A^i = \frac{W_Q^i (W_K^i)^t}{\sqrt{d_K}}.
$$
The next definition requires an additional parameter \(W_O \in \bR^{h d_V \times h d_V}\).

The *multi-head scaled dot-product self-attention map* [2] is defined by
$$
\begin{align*}
    \mathscr{M} : \bR^{n \times d} &\to \bR^{n \times h d_V} \\
    X &\mapsto \mathscr{M} (X) =
    [
    \Att^1(X) \, \cdots \, \Att^h(X)
    ] W_O
\end{align*}
$$

*Theorem*: If \(W_O\) is invertible and \(A^i W_V^i \neq \mathbf{0}\) for
some \(i\), then \(\mathscr{M}\) is not Lipschitz.

*Proof*: One can show that
$$
\begin{align*}
    \| d \mathscr{M}(X) \|_{\mathrm{op}}
    &\geq s_{\min}(W_O) \| d \Att^i(X) \|_{\mathrm{op}}
\end{align*}
$$
where \(s_{\min}(W_O) \neq 0\) is the minimum singular value of \(W_O\).
Invoking Theorem 1, we conclude that \(\mathscr{M}\) is not Lipschitz.
\(\blacksquare\)

One can relax the assumption that \(W_O\) is invertible in various ways, but this
significantly complicates the statement and proof of the theorem.

## References

1. *The Lipschitz Constant of Self-Attention*, Hyunjik Kim, George Papamakarios, Andriy Mnih.
Proceedings of the 38th International Conference on Machine Learning, PMLR 139:5562-5571, 2021.
2. *Attention is All you Need*, Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N Gomez, Łukasz Kaiser, Illia Polosukhin. Advances in Neural Information Processing Systems 30, 2017.

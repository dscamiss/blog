+++
title = 'An explicit formula for the gradient of standard attention'
date = 2024-09-19T11:03:04-07:00
draft = true
tag = ['attention', 'gradient', 'backpropagation', 'random-notes']
+++

$$
    \newcommand{\Attn}{\mathrm{Att}}
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\sL}{\mathscr{L}}
    \newcommand{\tr}{\mathrm{tr}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

We begin by defining the *parameter space* to be
$$
    \Theta = \bR^{n \times d} \times \bR^{n \times d} \times \bR^{n \times d}
$$
with generic element \(\theta = (Q, K, V)\).

The *standard attention map* is defined by
$$
\begin{align*}
    \Attn : \Theta &\to \bR^{n \times d} \\
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
where \(e_i\) is the \(i\)th Euclidean basis vector in \(\bR^n\).

The total derivative of \(\Attn\) at \(\theta\) is
$$
\begin{align*}
    d \Attn(\theta) \cdot \tilde{\theta} =
    d_Q \Attn(\theta) \cdot \tilde{Q} +
    d_K \Attn(\theta) \cdot \tilde{K} +
    d_V \Attn(\theta) \cdot \tilde{V},
\end{align*}
$$
where \(\tilde{\theta} = (\tilde{Q},\tilde{K},\tilde{V})\).

By the chain rule, the partial derivatives of \(\Attn\) are
$$
\begin{align*}
    d_Q \Attn(\theta) \cdot \tilde{Q}
    &= \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot K\tilde{Q}^t e_i]^t V \\
    d_K \Attn(\theta) \cdot \tilde{K}
    &= \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot \tilde{K}Q^t e_i]^t V \\
    d_V \Attn(\theta) \cdot \tilde{V}
    &= \sum_{i=1}^{n} e_i \sigma(p_i)^t \tilde{V},
\end{align*}
$$
where we have introduced the shorthand \(p_i = K Q^t e_i\).  Here, \(d \sigma(x)\) denotes the total derivative of
\(\sigma\) at \(x\), which we identify with the Jacobian matrix of \(\sigma\) at \(x\).

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

To compute \(d_Q \sL(\theta)\), observe that
$$
\begin{align*}
    \left\langle d_Q \Attn(\theta) \cdot \tilde{Q}, P \right\rangle_F
    &= \left\langle \sum_{i=1}^{n} e_i [d \sigma(p_i) \cdot K\tilde{Q}^t e_i]^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i [d \sigma(p_i) \cdot K\tilde{Q}^t e_i]^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i e_i^t \tilde{Q} K^t d \sigma(p_i)^t V, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \tr\left( V^t d \sigma(p_i) K \tilde{Q}^t e_i e_i^t P \right) \\
    &= \sum_{i=1}^{n} \tr\left( \tilde{Q}^t e_i e_i^t P V^t d \sigma(p_i) K \right) &\color{gray}\mbox{(Cyclic invariance)} \\
    &= \sum_{i=1}^{n} \left\langle \tilde{Q}, e_i e_i^t P V^t d \sigma(p_i) K \right\rangle_F \\
    &= \left\langle \tilde{Q}, \sum_{i=1}^{n} e_i e_i^t P V^t d \sigma(p_i) K \right\rangle_F,
\end{align*}
$$
where \(\langle \cdot, \cdot \rangle_F\) is the Frobenius inner product on \(\bR^{n \times d}\)
and \(\tr(\cdot)\) is trace.
$$
    d_Q \Attn(\theta)^* \cdot P = \sum_{i=1}^{n} e_i e_i^t P V^t d \sigma(p_i) K.
$$
Using this result, we obtain
$$
\begin{align*}
    d_Q \sL(\theta) \cdot \tilde{Q}
    &= \left\langle d_Q \sL(\theta) \cdot \tilde{Q}, 1 \right\rangle_\bR \\
    &= \left\langle d \ell(\Attn(\theta)) \cdot d_Q \Attn(\theta) \cdot \tilde{Q}, 1 \right\rangle_\bR \\
    &= \left\langle \tilde{Q}, d_Q \Attn(\theta)^* \circ d \ell(\Attn(\theta))^* \cdot 1 \right\rangle_F \\
    &= \left\langle \tilde{Q}, \sum_{i=1}^{n} e_i e_i^t (d \ell(\Attn(\theta))^* \cdot 1) V^t d \sigma(p_i) K \right\rangle_F \\
    &= \left\langle \tilde{Q}, \sum_{i=1}^{n} e_i e_i^t \Lambda(\theta) V^t d \sigma(p_i) K \right\rangle_F,
\end{align*}
$$
where we have introduced the shorthand
$$
    \Lambda(\theta) = d \ell(\Attn(\theta))^* \cdot 1.
$$
Thus we can [make the identification](https://en.wikipedia.org/wiki/Riesz_representation_theorem)
$$
    d_Q \sL(\theta) \equiv \sum_{i=1}^{n} e_i e_i^t \Lambda(\theta) V^t d \sigma(p_i) K.
$$

To compute \(d_K \sL(\theta)\), we proceed in a similar way.  Observe that
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
Using this result, we obtain
$$
    d_K \sL(\theta) \cdot \tilde{K} =
    \left\langle \tilde{K}, \sum_{i=1}^{n} d\sigma(p_i)^t V \Lambda(\theta)^t e_i e_i^t Q \right\rangle_F.
$$
As above, we can make the identification
$$
    d_K \sL(\theta) \equiv \sum_{i=1}^{n} d\sigma(p_i)^t V \Lambda(\theta)^t e_i e_i^t Q.
$$

Finally, to compute \(d_V \sL(\theta)\), simply observe that
$$
\begin{align*}
    \left\langle d_V \Attn(\theta) \cdot \tilde{V}, P \right\rangle_F
    &= \left\langle \sum_{i=1}^{n} e_i \sigma(p_i)^t \tilde{V}, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle e_i \sigma(p_i)^t \tilde{V}, P \right\rangle_F \\
    &= \sum_{i=1}^{n} \left\langle \tilde{V}, \sigma(p_i) e_i^t P \right\rangle_F \\
    &= \left\langle \tilde{V}, \sum_{i=1}^{n} \sigma(p_i) e_i^t P \right\rangle_F.
\end{align*}
$$
Using this result, we obtain
$$
\begin{align*}
    d_V \sL(\theta) \cdot \tilde{V}
    &= \left\langle \tilde{V}, \sum_{i=1}^{n} \sigma(p_i) e_i^t \Lambda(\theta) \right\rangle_F.
\end{align*}
$$
Once again, we can make the identification
$$
    d_V \sL(\theta) \equiv \sum_{i=1}^{n} \sigma(p_i) e_i^t \Lambda(\theta).
$$
Putting everything together, we have the identification
$$
\colorbox{magicmint}
{
$
    d \sL(\theta) \equiv
    \left(
    \Omega(\theta) K,
    \Omega(\theta)^t Q,
    \sum_{i=1}^{n} \sigma(p_i) e_i^t \Lambda(\theta)
    \right) \in \Theta,
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
To be completely explicit, note that we can write \(\Omega(\Theta)\) in "matrix form" as
$$
    \Omega(\theta) =
    \begin{bmatrix}
    e_1^t \Lambda(\theta) V^t d \sigma(p_1) \\
    \vdots \\
    e_n^t \Lambda(\theta) V^t d \sigma(p_n)
    \end{bmatrix}.
$$
Expanding each \(d\sigma(p_i)\), this becomes
$$
    \Omega(\theta) =
    \begin{bmatrix}
    e_1^t \Lambda(\theta) V^t (\Delta \sigma(p_1) - \sigma(p_1) \sigma(p_1)^t) \\
    \vdots \\
    e_n^t \Lambda(\theta) V^t (\Delta \sigma(p_n) - \sigma(p_n) \sigma(p_n)^t)
    \end{bmatrix}.
$$
For details concerning the computation of each \(d\sigma(p_i)\), see [this post]().

_Side note_: How smart are autograd implementations?  Do they recognize that
we can re-use the value of \(\Omega(\theta)\) in the computation of
\(d \sL(\theta)\)?


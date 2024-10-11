+++
title = 'Input-output sensitivity in LSTM networks'
date = 2024-10-06T14:23:38-07:00
draft = true
+++

## Introduction

The literature is full of claims that the LSTM architecture is well-adapted to learning input-output dependence
over long time lags, and there is a large amount of empirical evidence supporting
this claim.  Nevertheless, I couldn't find a proof of this claim, at least not in the form of
a direct analysis of input-output sensitivity.  In this post,
we get the ball rolling on a direct analysis. First, we derive a recurrence
relation which relates the input-output sensitivities over arbitrarily long time lags.
Then, we use the recurrence relation to show that a particular arrangement of hidden states
preserves input-output sensitivity.

<!--more-->

$$
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\sA}{\mathscr{A}}
    \newcommand{\sB}{\mathscr{B}}
    \newcommand{\sC}{\mathscr{C}}
    \newcommand{\sF}{\mathscr{F}}
    \newcommand{\sL}{\mathscr{L}}
    \newcommand{\sR}{\mathscr{R}}
    \newcommand{\olz}{\overline{z}}
    \newcommand{\oli}{\overline{i}}
    \newcommand{\olf}{\overline{f}}
    \newcommand{\olo}{\overline{o}}
    \newcommand{\smin}{\sigma_{\min}}
    \newcommand{\smax}{\sigma_{\max}}
    \newcommand{\bfone}{\bf{1}}
    \newcommand{\bfzero}{\bf{0}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Vanilla LSTM

We begin by defining the *parameter space* to be
$$
    \Theta = (\bR^{n_y \times n_x} \times \bR^{n_y \times n_y} \times \bR^{n_y})^4,
$$
with generic element
$$
    \theta = (W_z,R_z,b_z,W_i,R_i,b_i,W_f,R_f,b_f,W_o,R_o,b_o).
$$

The *vanilla LSTM* is the recurrent neural network (RNN) defined by
$$
\begin{align*}
    z_t &= g(W_z x_t + R_z y_{t-1} + b_z) \\
    i_t &= \sigma(W_i x_t + R_i y_{t-1} + b_i) \\
    f_t &= \sigma(W_f x_t + R_f y_{t-1} + b_f) \\
    o_t &= \sigma(W_o x_t + R_o y_{t-1} + b_o) \\
    c_t &= z_t \odot i_t + c_{t-1} \odot f_t \\
    y_t &= h(c_t) \odot o_t \\
    y_0 &= \bfzero,
\end{align*}
$$
where \(g, h\) are \(\tanh\) and \(\sigma\) is the sigmoid function, and
we adopt the convention that functions act componentwise on vectors.
For example, if \(x = (x^1,\dots,x^n)\), then
$$
\begin{align*}
    g(x) &= (g(x^1), \dots, g(x^n))^\dagger \\
    g'(x) &= (g'(x^1), \dots, g'(x^n))^\dagger.
\end{align*}
$$
Other choices for \(g, h, \sigma\) are possible, but we will be using properties
of these specific functions below, such as positivity and vanishing of derivatives
at infinity.

To be clear, the preceding definition means that each \(\square \in \{z_t, i_t, f_t, o_t, c_t, y_t\}\) is a map
$$
    \square : \Theta \times (\bR^{n_x})^t \to \bR^{n_y}
$$
defined in the natural way which retains causality (with respect to the input sequence).

For example, \(z_t\) is defined by
$$
\begin{align*}
    z_t(\theta, x_1, \dots, x_t)
    &= g(W_z x_t + R_z y_{t-1}(\theta, x_1, \dots, x_{t-1}) + b_z),
\end{align*}
$$
and the other maps are similar.

Following the notation of [1], we introduce
$$
\begin{align*}
    \olz_t &= W_z x_t + R_z y_{t-1} + b_z \\
    \oli_t &= W_i x_t + R_i y_{t-1} + b_i \\
    \olf_t &= W_f x_t + R_f y_{t-1} + b_f \\
    \olo_t &= W_o x_t + R_o y_{t-1} + b_o.
\end{align*}
$$
As above, each \(\square \in \{\olz_t, \oli_t, \olf_t, \olo_t\}\) is a map
$$
    \square : \Theta \times (\bR^{n_x})^t \to \bR^{n_y}
$$
defined in the natural way which retains causality.

We say that:
* \(z_t, i_t, f_t, o_t\) are the *hidden states* at time \(t\),
* \(c_t\) is the *cell state* at time \(t\),
* \(x_t\) is the *input* at time \(t\), and
* \(y_t\) is the *output* at time \(t\).

As stated above, the goal is to understand the input-output sensitivity of the LSTM.
We take the partial derivatives \(d_{x_s} y_t\) as our measurements of
input-output sensitivity.  Due to coupling in the LSTM, evaluating these partial derivatives requires
evaluating the partial derivatives of the hidden states and cell state.

## Hidden states

In this section, we evaluate the partial derivatives \(d_{x_s} \square\),
for \(\square \in \{ z_t, i_t, f_t, o_t \}\).

For \(s = t\), we have
$$
\begin{align*}
    d_{x_s} z_s &= \Delta g'(\olz_s) W_z \\
    d_{x_s} i_s &= \Delta \sigma'(\oli_s) W_i \\
    d_{x_s} f_s &= \Delta \sigma'(\olf_s) W_f \\
    d_{x_s} o_s &= \Delta \sigma'(\olo_s) W_o.
\end{align*}
$$
Here, \(\Delta x\) is the diagonal matrix whose \((j, j)\)-th component
is equal to \(x^j\).

For example,
$$
    \Delta g'(\olz_s) =
    \begin{bmatrix}
        g'(\olz_s^1) & 0 & \cdots & 0 \\
        0 & g'(\olz_s^2) & \cdots & 0 \\
        0 & 0 & \ddots & 0 \\
        0 & 0 & \cdots & g'(\olz_s^{n_h})
    \end{bmatrix}.
$$

For \(s < t\), we have
$$
\begin{align*}
    d_{x_s} z_t &= \Delta g'(\olz_t) R_z d_{x_s} y_{t-1} \\
    d_{x_s} i_t &= \Delta \sigma'(\oli_t) R_i d_{x_s} y_{t-1} \\
    d_{x_s} f_t &= \Delta \sigma'(\olf_t) R_f d_{x_s} y_{t-1} \\
    d_{x_s} o_t &= \Delta \sigma'(\olo_t) R_o d_{x_s} y_{t-1}.
\end{align*}
$$

## Cell state

In this section, we evaluate the partial derivatives \(d_{x_s} c_t\).

For \(s = t\), we have
$$
\begin{align*}
    d_{x_s} c_s \cdot \tilde{x}
    &= (d_{x_s} z_s \cdot \tilde{x}) \odot i_s +
    z_s \odot (d_{x_s} i_s \cdot \tilde{x}) +
    c_{s-1} \odot (d_{x_s} f_s \cdot \tilde{x}).
\end{align*}
$$

Using the expressions for \(d_{x_s} z_s\), \(d_{x_s} i_s\), and \(d_{x_s} f_s\) from the previous section,
this becomes
$$
\begin{align*}
    d_{x_s} c_s &= \sC^W_s,
\end{align*}
$$
where we have introduced
$$
\colorbox{lesserbox}
{
$
    \begin{align*}
        \sC^W_s = \Delta g'(\olz_s) \Delta i_s W_z +
        \Delta z_s \Delta \sigma'(\oli_s) W_i +
        \Delta c_{s-1} \Delta \sigma'(\olf_s) W_f.
    \end{align*}
$
}
$$

Similarly, for \(s < t\), we have
$$
\begin{align*}
    d_{x_s} c_t \cdot \tilde{x}
    &= (d_{x_s} z_t \cdot \tilde{x}) \odot i_t +
    z_t \odot (d_{x_s} i_t \cdot \tilde{x}) +
    (d_{x_s} c_{t-1} \cdot \tilde{x}) \odot f_t +
    c_{t-1} \odot (d_{x_s} f_t \cdot \tilde{x}).
\end{align*}
$$
Using the expressions for \(d_{x_s} z_t\), \(d_{x_s} i_t\), and \(d_{x_s} f_s\) from the previous
section, this becomes
$$
\begin{align*}
    d_{x_s} c_t &= \sC^R_t d_{x_s} y_{t-1} + \Delta f_t d_{x_s} c_{t-1},
\end{align*}
$$
where we have introduced
$$
\colorbox{lesserbox}
{
$
    \begin{align*}
        \sC^R_t &=
        \Delta g'(\olz_t) \Delta i_t R_z +
        \Delta z_t \Delta \sigma'(\oli_t) R_i +
        \Delta c_{t-1}  \Delta \sigma'(\olf_t) R_f.
    \end{align*}
$
}
$$

This gives a recurrence relation for the \(d_{x_s} c_t\)'s, whose solution is
$$
\begin{align*}
    d_{x_s} c_t
    &= \left( \prod_{k=s+1}^{t} \Delta f_k \right) \sC^W_s + \sum_{j=s}^{t-1} \left( \prod_{k=j+2}^{t} \Delta f_k \right) \sC^R_{j+1} d_{x_s} y_j,
\end{align*}
$$
where the "empty" matrix products are equal to the identity matrix.  This can
be proven by a straightforward induction argument.
For example, when \(s = 1\) and \(t = 3\), we have
$$
\begin{align*}
    d_{x_1} c_3
    &= \sC^R_3 d_{x_1} y_2 + \Delta f_3 d_{x_1} c_2 \\
    &= \sC^R_3 d_{x_1} y_2 + \Delta f_3 [\sC^R_2 d_{x_1} y_1 + \Delta f_2 d_{x_1} c_1] \\
    &= \sC^R_3 d_{x_1} y_2 + \Delta f_3 \sC^R_2 d_{x_1} y_1 + \Delta f_3 \Delta f_2 d_{x_1} c_1 \\
    &= \Delta f_3 \Delta f_2 \sC^W_1 + \sC^R_3 d_{x_1} y_2 + \Delta f_3 \sC^R_2 d_{x_1} y_1.
\end{align*}
$$

We can express the solution more compactly by introducing
$$
    \sF^{b}_a = \prod_{k=a}^{b} \Delta f_k,
$$
so that
$$
\colorbox{lesserbox}
{
$
    \begin{align*}
        d_{x_s} c_t
        &= \sF^{t}_{s+1} \sC^W_s + \sum_{j=s}^{t-1} \sF^{t}_{j+2} \sC^R_{j+1} d_{x_s} y_{j}.
    \end{align*}
    \qquad (1)
$
}
$$

## Output

In this section, we evaluate the partial derivatives \(d_{x_s} y_t\).

For \(s = t\), we have
$$
\begin{align*}
    d_{x_s} y_s \cdot \tilde{x}
    &= \Delta h'(c_s) \Delta o_s \sC^W_s + \Delta h(c_s) \Delta \sigma'(\olo_s) W_o.
\end{align*}
$$

For \(s < t\), we have
$$
\begin{align*}
    d_{x_s} y_t \cdot \tilde{x}
    &= (\Delta h'(c_t) d_{x_s} c_t \cdot \tilde{x}) \odot o_t + h(c_t) \odot (d_{x_s} o_t \cdot \tilde{x}).
\end{align*}
$$
Using the expression for \(d_{x_s} o_t\) derived above, this becomes
$$
\begin{align*}
    d_{x_s} y_t
    &= \Delta h'(c_t) \Delta o_t d_{x_s} c_t + \Delta h(c_t) \Delta \sigma'(\olo_t) R_o d_{x_s} y_{t-1}.
\end{align*}
$$
Equivalently,
$$
    d_{x_s} c_t =
    (\Delta h'(c_t) \Delta o_t)^{-1} d_{x_s} y_t - (\Delta h'(c_t) \Delta o_t)^{-1}
    \Delta \sigma'(\olo_t) \Delta h(c_t) R_o d_{x_s} y_{t-1}.
$$
Here we have used positivity of \(h'\) and \(\sigma\) to invert
\(\Delta h'(c_t)\) and \(\Delta o_t\), respectively.

For notational convenience, introduce
$$
\begin{align*}
    \sA_t &= \Delta h'(c_t) \Delta o_t \\
    \sB_t &= \Delta \sigma'(\olo_t) \Delta h(c_t) R_o.
\end{align*}
$$
Then we can write
$$
    d_{x_s} c_t = \sA_t^{-1} d_{x_s} y_t - \sA_t^{-1} \sB_t d_{x_s} y_{t-1}.
$$

Together with \((1)\), this gives a recurrence relation for the \(d_{x_s} y_t\)'s:
$$
\colorbox{magicmint}
{
$
    \begin{align*}
        d_{x_s} y_t
        &= \sA_t \sF^{t}_{s+1} \sC^W_s + \sA_t \sum_{j=s}^{t-1} \sF^{t}_{j+2} \sC^R_{j+1} d_{x_s} y_{j}
        + \sB_t d_{x_s} y_{t-1}.
        \qquad (2)
    \end{align*}
$
}
$$

## Preserving input-output sensitivity

In this section, we use the recurrence relation \((2)\) to
identify a particular arrangement of hidden states which preserves input-output sensitivity.
Note that we are not claiming that this arrangement is the *only* one that
preserves input-output sensitivity, but it is arguably the simplest one to identify.

We say that:
* \(z_\tau\) is *saturated* if each of its components satisfies \(z_\tau^j \approx 1\) or \(z_\tau^j \approx -1\),
* \(\square \in \{ i_\tau, f_\tau, o_\tau \}\) is *pos-saturated* if \(\square \approx \bfone\), and
* \(\square \in \{ i_\tau, f_\tau, o_\tau \}\) is *neg-saturated* if \(\square \approx \bfzero\).

Our assumptions in this section are:
* \(f_\tau\) is pos-saturated for \(s \leq \tau \leq t\),
* \(i_s\) is neg-saturated **or** \(i_s\) is pos-saturated and \(z_s\) is saturated,
* \(i_\tau\) is neg-saturated **or** \(i_\tau\) is pos-saturated and \(z_\tau\) is saturated, for \(s + 2 \leq \tau \leq t\),
* \(o_t\) is pos-saturated, and
* The minimum singular value of \(\Delta h'(c_t) \sC^R_{s+1}\) is greater than or equal to \(1\).

For simplicity's sake, in the remainder of this section we will treat the saturation
approximations as if they were equalities.

The first assumption implies that each "\(\sF^b_a\)" term is the identity matrix, so
$$
\begin{align*}
    d_{x_s} y_t &= \sA_t \sC^W_s + \sA_t \sum_{j=s}^{t-1} \sC^R_{j+1} d_{x_s} y_{j} + \sB_t d_{x_s} y_{t-1}.
\end{align*}
$$
The first three assumptions imply that \(\sC^W_s = \bfzero\) and \(\sC^R_\tau = \bfzero\) for \(s + 2 \leq \tau \leq t\), so
$$
\begin{align*}
    d_{x_s} y_t &= \sA_t \sC^R_{s+1} d_{x_s} y_s + \sB_t d_{x_s} y_{t-1}.
\end{align*}
$$
The fourth assumption implies that \(\sA_t = \Delta h'(c_t)\) and \(\sB_t = 0\), so
$$
\begin{align*}
    d_{x_s} y_t &= \Delta h'(c_t) \sC^R_{s+1} d_{x_s} y_s
\end{align*}
$$
Finally, the fifth assumption implies that
$$
\colorbox{magicmint}
{
$
    \begin{align*}
        \| d_{x_s} y_t \|
        \geq \| \Delta h'(c_t) \sC^R_{s+1} d_{x_s} y_s \|
        \geq \smin(\Delta h'(c_t) \sC^R_{s+1}) \| d_{x_s} y_s \|
        \geq \| d_{x_s} y_s \|,
    \end{align*}
$
}
$$
where \(\smin(A)\) is the minimum singular value of \(A\).

## Extensions



## References

[1] K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink and J. Schmidhuber, "LSTM: A Search Space Odyssey," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 10, pp. 2222--2232
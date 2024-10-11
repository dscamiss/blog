+++
title = 'Input-output sensitivity in LSTM networks'
date = 2024-10-06T14:23:38-07:00
draft = true
+++

## Introduction

The literature is full of claims that the LSTM architecture is well-adapted to learning input-output dependence
over long time lags, and there is a large amount of empirical evidence supporting
this claim.  Nevertheless, I couldn't find a proof of this claim, at least not in the form of
a *direct analysis of input-output sensitivity*.  In this post,
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
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Vanilla LSTM

We begin by defining the *parameter space* to be
$$
    \Theta = (\bR^{n_h \times n_x} \times \bR^{n_h \times n_y} \times \bR^{n_h})^4,
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
    y_0 &= 0_h,
\end{align*}
$$
where \(g, h\) are \(\tanh\) and \(\sigma\) is the sigmoid function.

To be clear, this definition means that each \(\square \in \{z_t, i_t, f_t, o_t, c_t, y_t\}\) is a map
$$
    \square : \Theta \times (\bR^{n_x})^t \to \bR^{n_h}
$$
defined in the natural way which retains causality (with respect to the input sequence).

For example,
$$
\begin{align*}
    z_t(\theta, x_1, \dots, x_t)
    &= g(W_z x_t + R_z y_{t-1}(\theta, x_1, \dots, x_{t-1}) + b_z).
\end{align*}
$$

To keep the notation under control, we introduce
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
    \square : \Theta \times (\bR^{n_x})^t \to \bR^{n_h}
$$
defined in the natural way which retains causality.

We say that:
* \(z_t, i_t, f_t, o_t\) are the *hidden states* at time \(t\),
* \(c_t\) is the *cell state* at time \(t\), and
* \(y_t\) is the *output* at time \(t\).

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
Here, \(\Delta x\) is the diagonal matrix whose \((j,j)\)th component is equal to
\(x^j\), and the derivatives \(g', \sigma'\) are applied component-wise.

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

## Output

In this section, we evaluate the partial derivatives \(d_{x_s} y_t\).

TODO: \(d_{x_s} y_s\)???

For \(s < t\), we have
$$
\begin{align*}
    d_{x_s} y_t \cdot \tilde{x}
    &= (\Delta h'(c_t) d_{x_s} c_t \cdot \tilde{x}) \odot o_t + h(c_t) \odot (d_{x_s} o_t \cdot \tilde{x}).
\end{align*}
$$
Using the expression for \(d_{x_s} o_t\) from above, this becomes
$$
\begin{align*}
    d_{x_s} y_t
    &= \Delta h'(c_t) \Delta o_t d_{x_s} c_t + \Delta h(c_t) \Delta \sigma'(\olo_t) R_o d_{x_s} y_{t-1}.
\end{align*}
$$
Equivalently,
$$
    d_{x_s} c_t =
    \Delta h'(c_t)^{-1} \Delta o_t^{-1} d_{x_s} y_t - \Delta h'(c_t)^{-1}  \Delta o_t^{-1}
    \Delta \sigma'(\olo_t) \Delta h(c_t) R_o d_{x_s} y_{t-1}.
$$
Here we have used \(h' > 0\) and \(\sigma > 0\) to invert
\(\Delta h'(c_t)\) and \(\Delta o_t\), respectively.

For notational convenience, set
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

Using the expressions for \(d_{x_s} z_s\), \(d_{x_s} i_s\), and \(d_{x_s} f_s\) from above, this becomes
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

For \(s < t\), we have
$$
\begin{align*}
    d_{x_s} c_t \cdot \tilde{x}
    &= (d_{x_s} z_t \cdot \tilde{x}) \odot i_t +
    z_t \odot (d_{x_s} i_t \cdot \tilde{x}) +
    (d_{x_s} c_{t-1} \cdot \tilde{x}) \odot f_t +
    c_{t-1} \odot (d_{x_s} f_t \cdot \tilde{x}).
\end{align*}
$$
Using the expressions for \(d_{x_s} z_t\), \(d_{x_s} i_t\), \(d_{x_s} c_{t-1}\), and \(d_{x_s} f_s\) from above, this becomes
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

This gives a recurrence relation for \(d_{x_s} c_t\), whose general solution is
$$
\begin{align*}
    d_{x_s} c_t
    &= \left( \prod_{k=s+1}^{t} \Delta f_k \right) \sC^W_s + \sum_{j=s}^{t-1} \left( \prod_{k=j+2}^{t} \Delta f_k \right) \sC^R_{j+1} d_{x_s} y_j,
\end{align*}
$$
where the matrix products are left-multiplicative and "empty" products are equal to \(I_{n_h}\).

For example, when \(s = 1\) and \(t = 3\), we have
$$
\begin{align*}
    d_{x_1} c_3
    &= \sC^R_3 d_{x_1} y_2 + \Delta f_3 d_{x_1} c_2 \\
    &= \sC^R_3 d_{x_1} y_2 + \Delta f_3 [\sC^R_2 d_{x_1} y_1 + \Delta f_2 d_{x_1} c_1] \\
    &= \sC^R_3 d_{x_1} y_2 + \Delta f_3 \sC^R_2 d_{x_1} y_1 + \Delta f_3 \Delta f_2 d_{x_1} c_1.
\end{align*}
$$

We can express the general solution more compactly by introducing
$$
    \sF^{b}_a = \prod_{k=a}^{b} \Delta f_k,
$$
so that
$$
\begin{align*}
    d_{x_s} c_t
    &= \sF^{t}_{s+1} \sC^W_s + \sum_{j=s}^{t-1} \sF^{t}_{j+2} \sC^R_{j+1} d_{x_s} y_{j}.
\end{align*}
$$

Using the expression for \(d_{x_s} c_t\) from above, we obtain
$$
\colorbox{magicmint}
{
$
    \begin{align*}
        d_{x_s} y_t
        &= \sA_t \sF^{t}_{s+1} \sC^W_s + \sA_t \sum_{j=s}^{t-1} \sF^{t}_{j+2} \sC^R_{j+1} d_{x_s} y_{j}
        + \sB_t d_{x_s} y_{t-1}.
    \end{align*}
$
}
$$
This gives a recurrence relation for the partial derivatives \(d_{x_s} y_t\).

## Preserving input sensitivity

In this section, we use the recurrence relation from the previous section to
identify a particular arrangement of hidden states which maintains input sensitivity.
Note that we are not claiming that this arrangement is the *only* one that
maintains input sensitivity, but it is perhaps the simplest one to identify.

Our assumptions in this section are:
* \(f_\tau\) is saturated for \(s \leq \tau \leq t\),
* \(z_s\) or \(i_s\) is saturated,
* \(z_\tau\) or \(i_\tau\) are saturated for \(s + 2 \leq \tau \leq t\),
* \(o_t\) is saturated, and
* The minimum singular value of \(\Delta h'(c_t) \sC^R_{s+1}\) is greater than or equal to \(1\).

The first assumption implies that each "\(\sF\)" term is the identity matrix, so
$$
\begin{align*}
    d_{x_s} y_t &= \sA_t \sC^W_s + \sA_t \sum_{j=s}^{t-1} \sC^R_{j+1} d_{x_s} y_{j} + \sB_t d_{x_s} y_{t-1}.
\end{align*}
$$
The first three assumptions imply that \(\sC^W_s = 0\) and \(\sC^R_\tau = 0\) for \(s + 2 \leq \tau \leq t\), so
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
\begin{align*}
    \| d_{x_s} y_t \|
    \geq \| \Delta h'(c_t) \sC^R_{s+1} d_{x_s} y_s \|
    \geq \smin(\Delta h'(c_t) \sC^R_{s+1}) \| d_{x_s} y_s \|
    \geq \| d_{x_s} y_s \|,
\end{align*}
$$
where \(\smin(A)\) is the minimum singular value of \(A\).

## Maintaining single-step sensitivity with a closed forget gate

In this section, we use the recurrence relation from the previous section
to derive two conditions under which
$$
\begin{align*}
    \| d_{x_s} y_t \| \geq \| d_{x_s} y_{t-1} \|.
\end{align*}
$$

In this section, \(s < t\) and we assume that \(f_t = 0_{n_h}\).

Using the result of the previous section, we have
$$
\begin{align*}
    d_{x_s} y_t
    &= (\sA_t \sC^R_{t} + \sB_t) d_{x_s} y_{t-1}.
\end{align*}
$$
The operator norm lower bound is
$$
    \| d_{x_s} y_t \| \geq \smin(\sA_t \sC^R_t + \sB_t) \| d_{x_s} y_{t-1} \|
$$
Using standard results, the minimum singular value satisfies
$$
\begin{align*}
    \smin(\sA_t \sC^R_{t} + \sB_t) \geq \smin(\sA_t \sC^R_t) - \smax(\sB_t) \\
    \smin(\sA_t \sC^R_{t} + \sB_t) \geq \smin(\sB_t) - \smax(\sA_t \sC^R_t) \\
\end{align*}
$$

**Case 1**: Maximum gap between \(\smin(\sA_t \sC^R_t) - \smax(\sB_t)\)

* Want \(\sB_t = 0\)
* Could have \(R_o\) but this is improbable
* Instead, take \(\olo_t\) components large
* This means \(o_t\) components equal 1

We have
$$
    \sA_t \sC^R_t =
    \Delta h'(c_t) \Delta g'(\olz_t) \Delta i_t R_z +
    \Delta h'(c_t) \Delta z_t \Delta \sigma'(\oli_t) R_i.
$$

**Case 2**: Maximum gap between \(\smin(\sB_t) - \smax(\sA_t \sC^R_t)\)

* Want \(\sA_t = 0\)
* Instead, take \(\olo_t\) components large
* This means \(o_t\) components equal 1

----

The idea should be this:
* Case 1: Drive \(c_t\) large (check pos/neg makes a difference?), then \(\sA_t = 0\) and \(\sB_t = \Delta \sigma'(\olo_t) I_{\pm} R_o\)
and

$$
    \smin(\Delta \sigma'(\olo_t) I_\pm R_o) = \smin(\Delta \sigma'(\olo_t) R_o)
$$

(But check this...)
* Case 2: Drive \(\olo_t\) large and positive, then \(\sB_t = 0\) and \(\sA_t = \Delta h'(c_t)\)
* Also, what's happening with \(\sA_t\) and \(\sB_t\) (in \(c_t,o_t\)) is independent of \(\sC^R_t\) (in \(z_t,i_t\)).

----

The next result is the "\(o_t\) is positively saturated" case.

**Proposition 1**: Suppose that \(o_t = (1,\dots,1)^t\) and
$$
    \smin(\Delta h'(c_t)) \smin(\Delta g'(\olz_t) \Delta i_t R_z + \Delta z_t \Delta \sigma'(\oli_t) R_i) \geq 1.
$$
Then
$$
\colorbox{magicmint}
{
$
    \| d_{x_s} y_t \| \geq \| d_{x_s} y_{t-1} \|.
$
}
$$

**Proof** The first assumption implies that \(\Delta o_t = I_{n_h}\) and \(\Delta \sigma'(\olo_t) = 0_{n_h}\). Therefore
$$
\begin{align*}
    \sA_t &= \Delta h'(c_t) \\
    \sB_t &= 0_{n_h \times n_h}.
\end{align*}
$$
Plugging into Bound 1 above, we have
$$
    \| d_{x_s} y_t \| \geq \smin(\Delta h'(c_t) \sC^R_t) \geq \smin(\Delta h'(c_t)) \smin(\sC^R_t).
$$
The second inequality uses invertibility of \(\Delta h'(c_t)\).  Finally, \(f_t = 0_{n_h}\)
implies that
$$
    \Delta \sigma'(\olf_t) = 0_{n_h}
$$
and therefore
$$
    \sC^R_t = \Delta g'(\olz_t) \Delta i_t R_z + \Delta z_t \Delta \sigma'(\oli_t) R_i.
$$
This completes the proof. \(\blacksquare\)

The next result is the "\(c_t\) is saturated" case.

**Proposition 2**: Suppose that
* The norm of \(c_t\) is large, so that \(h(c_t) \in \{ (1,\dots,1)^t, (-1,\dots,-1)^t \}\), and
* The minimum singular values of \(\Delta \sigma'(\olo_t)\) and \(R_o\) satisfy \(\smin(\Delta \sigma'(\olo_t)) \smin(R_o) \geq 1\).

Then
$$
    \| d_{x_s} y_t \| \geq \| d_{x_s} y_{t-1} \|.
$$

**Proof**: Then \(h'(c_t) = 0_{n_h}\) and \(\sA_t = 0_{n_h \times n_h}\).
Furthermore, \(\Delta h(c_t) \in \{ I_{n_h}, -I_{n_h} \}\) and we have
\(\smin(\sB_t) = \smin(\Delta \sigma'(\olo_t) R_o)\).
Since \(\Delta \sigma'(\olo_t)\) is invertible, we have
$$
\begin{align*}
    \smin(\sB_t) \geq \smin(\Delta \sigma'(\olo_t)) \smin(R_o) \geq 1
\end{align*}
$$
and the conclusion follows.
+++
title = 'Approximately-optimal learning rates in a special class of fully-connected neural networks'
date = 2024-10-25T15:29:24-07:00
draft = true
tag = ['fully-connected', 'learning-rate', 'random-notes']
+++

In this post, we explicitly compute approximations to the optimal learning rates,
for a special class of fully-connected neural networks.

The class consists of multi-layer fully-connected neural networks
with piecewise-linear activation functions.  We assume that the
loss function is norm-squared and the optimization scheme is mini-batch gradient descent.

The focus on piecewise-linear activation functions is motivated by two things:
First, such activation functions are widely used in practice (for example, standard
ReLU and leaky ReLU). Second, such activation functions lead to simpler second-order partial derivatives
of the loss function, and these partial derivatives are needed to compute the
approximately-optimal learning rates.

This post generalizes [Part 1]({{< ref "optimal_learning_rates_one_layer" >}}),
which dealt only with the single-layer case.

<!--more-->

$$
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\Id}{\mathrm{Id}}
    \newcommand{\sL}{\mathscr{L}}
    \newcommand{\sP}{\mathscr{P}}
    \newcommand{\sR}{\mathscr{R}}
    \newcommand{\tr}{\mathrm{trace}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Approximately-optimal learning rates in gradient descent

Consider a smooth function \(f : \Theta \to \bR\),
where \(\Theta\) is a finite-dimensional vector space.
Gradient descent attempts to minimize \(f\) by iterating on
$$
    \theta \leftarrow \theta - \alpha \nabla f(\theta).
$$
Here, \(\alpha \in \bR_{> 0}\) is the *learning rate*,
\(\nabla f(\theta)\) is the gradient of \(f\) at \(\theta\),
and \(\theta\) is given a suitable initial value.

We are interested in a dynamic learning rate policy, in which \(\alpha\)
is a function of \(\theta\).
In particular, suppose that \(\theta\) is fixed and we want to choose \(\alpha\) such that
$$
    f(\theta - \alpha \nabla f(\theta))
$$
is minimized.

In the next definition, \(df(\theta)\) and \(d^2 f(\theta)\) are the total derivative
and second-order total derivative of \(f\) at \(\theta\).
The definition is justified in light of the developments in [Part 1]({{< ref "optimal_learning_rates_one_layer" >}}).

*Definition*: Suppose that
$$
    d^2 f(\theta) \cdot (\nabla f(\theta), \nabla f(\theta)) > 0.
$$
We say that
$$
    \alpha_*(\theta) =
    \frac{
        df(\theta) \cdot \nabla f(\theta)
    }{
        d^2 f(\theta) \cdot (\nabla f(\theta), \nabla f(\theta))
    }.
$$
is the *approximately-optimal learning rate for \(f\) at \(\theta\)*.

As a special case, suppose that \(\Theta\) is the product vector space
$$
    \Theta = \Theta_1 \times \cdots \times \Theta_n
$$
with generic element \(\theta = (\theta_1,\dots,\theta_n)\).
Then we can write
$$
    \alpha_*(\theta) =
    \frac{
        \sum_{i=1}^{n} d_{\theta_i} f(\theta) \cdot \nabla_{\theta_i} f(\theta)
    }{
        \sum_{i,j=1}^{n} d^2_{\theta_i,\theta_j} f(\theta) \cdot (\nabla_{\theta_i} f(\theta), \nabla_{\theta_j} f(\theta))
    }.
$$
Here we have introduced the following notation:
* \(\nabla_{\theta_i} f(\theta)\) is the partial gradient of \(f\) with respect to \(\theta_i\) at \(\theta\), and
* \(d^2_{\theta_i,\theta_j} f(\theta)\) is the second-order partial derivative of \(f\) with respect to \((\theta_i,\theta_j)\) at \(\theta\).

## Fully-connected neural networks

Fix \(L \geq 1\) and define the *parameter space* to be
$$
    \Theta = \bR^{n_L \times n_{L-1}} \times \bR^{n_L} \times \cdots \times
    \bR^{n_1 \times n_{0}} \times \bR^{n_1}
$$
with generic element
$$
    \theta = (W_L,b_L,\dots,W_1,b_1).
$$

We will need the following entities, for each \(1 \leq \ell \leq L\):
* The \(\ell\)th *activation function* is a piecewise-linear function \(\sigma_\ell^0 : \bR \to \bR\),
* The \(\ell\)th *activation map* \(\sigma_\ell : \bR^{n_\ell} \to \bR^{n_\ell}\) is defined by
\begin{align*}
    \sigma_\ell(x) &= (\sigma_\ell^0(x^1), \dots, \sigma_\ell^0(x^{n_\ell}))^t,
\end{align*}
* The \(\ell\)th *intermediate computation map*
$$
\begin{align*}
    z_\ell : \Theta \times \bR^{n_0} &\to \bR^{n_\ell} \\
    (\theta,x) &\mapsto z_\ell(\theta, x) = W_\ell \sigma_{\ell-1}(z_{\ell-1}(\theta,x)) + b_\ell,
\end{align*}
$$
where \(\sigma_0 = \Id_{\bR^{n_0}}\) and \(z_0 \equiv x\), and
* The \(\ell\)th *activation map* \(a_\ell = \sigma_\ell \circ z_\ell\).  We also set \(a_0 \equiv x\).

Note that we can write \(z_\ell\) more compactly as
$$
\begin{align*}
    z_\ell(\theta, x) = W_\ell a_{\ell-1}(\theta,x) + b_\ell.
\end{align*}
$$

Consider the fully-connected neural network
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    \hat{y} = a_L.
\end{align*}
$
}
$$

In the rest of this post, we will assume that we are working
away from those points where the activation maps are not differentiable.
This is necessary since piecewise-linear functions are not
everywhere differentiable, in general.

Suppose that we have (mini-batch) training data
$$
\begin{align*}
    \{ (x_i, y_i) \}_{i=1}^{M}, \quad
    x_i \in \bR^{n_0}, \quad y_i \in \bR^{n_L}
\end{align*}
$$
and we are using the norm-squared loss function
$$
\begin{align*}
    \sL : \Theta \to \bR
\end{align*}
$$
defined by
$$
\begin{align*}
    \sL(\theta)
    &= \frac{1}{2M} \sum_{i=1}^{M} \langle \hat{y}(\theta, x_i) - y_i, \hat{y}(\theta, x_i) - y_i \rangle,
\end{align*}
$$
where \(\langle\cdot,\cdot\rangle\) is the Euclidean inner product on \(\bR^{n_L}\).

The approximately-optimal learning rate for \(\sL\) at \(\theta\) is equal to
$$
\begin{align}
  \label{eq:optimal_learning_rate_product_L}\tag{2}%
  \alpha_*(\theta) = \frac{N_W(\theta) + N_b(\theta)}{D_{W,W}(\theta) + D_{W,b}(\theta) + D_{b,W}(\theta) + D_{b,b}(\theta)},
\end{align}
$$
where we have introduced
$$
\begin{align*}
  N_W(\theta) &= \sum_{\ell=1}^{L} d_{W_\ell} \sL(\theta) \cdot \nabla_{W_\ell} \sL(\theta) \\
  N_b(\theta) &= \sum_{\ell=1}^{L} d_{b_\ell} \sL(\theta) \cdot \nabla_{b_\ell} \sL(\theta) \\
  D_{W,W}(\theta) &= \sum_{\lambda,\ell=1}^{L} d_{W_\lambda,W_\ell}^2 \sL(\theta) \cdot (\nabla_{W_\lambda} \sL(\theta), \nabla_{W_\ell} \sL(\theta)) \\
  D_{W,b}(\theta) &= \sum_{\lambda,\ell=1}^{L} d_{W_\lambda,b_\ell}^2 \sL(\theta) \cdot (\nabla_{W_\lambda} \sL(\theta), \nabla_{b_\ell} \sL(\theta)) \\
  D_{b,W}(\theta) &= \sum_{\lambda,\ell=1}^{L} d_{b_\lambda,W_\ell}^2 \sL(\theta) \cdot (\nabla_{b_\lambda} \sL(\theta), \nabla_{W_\ell} \sL(\theta)) \\
  D_{b,b}(\theta) &= \sum_{\lambda,\ell=1}^{L} d_{b_\lambda,b_\ell}^2 \sL(\theta) \cdot (\nabla_{b_\lambda} \sL(\theta), \nabla_{b_\ell} \sL(\theta)).
\end{align*}
$$

Using the symmetry of second-order partial derivatives, we can write

$$
\colorbox{lesserbox}
{
$
\begin{align}
  \alpha_*(\theta) = \frac{N_W(\theta) + N_b(\theta)}{D_{W,W}(\theta) + 2 D_{W,b}(\theta) + D_{b,b}(\theta)}.
\end{align}
$
}
$$

Next, we evaluate the individual terms in this expression.

To keep the notation relatively compact, we define
$$
\begin{align*}
    e_i &= \hat{y}(\theta, x_i) - y_i \\
    a_\ell(i) &= a_\ell(\theta, x_i) \\
    \Delta'_\ell(i) &= \Delta_\ell (\sigma_\ell'(z_\ell(\theta, x_i)),
\end{align*}
$$
where
$$
    \sigma_\ell'(x) = (\sigma_\ell'(x^1), \dots, \sigma_\ell'(x^{n_\ell}))^t
$$
and \(\Delta_\ell : \bR^{n_\ell} \to \bR^{n_\ell \times n_\ell}\) sends \(x\) to the
diagonal matrix whose \((i,i)\)th component is \(x^i\).

For the same reason, we also define
$$
\begin{align*}
  \sP_{\beta,\alpha}(i) &= \Delta'_\beta(i) W_\beta \Delta'_{\beta-1}(i) W_{\beta-1} \cdots \Delta'_\alpha(i) W_\alpha \\
  \sR_{\beta,\alpha}(i,j) &= \sP_{\beta,\alpha}(i) \Delta'_{\alpha-1}(i) \Delta'_{\alpha-1}(j) \sP_{\beta,\alpha}(j)^\dagger \\
  a_\alpha(i,j) &= a_{\alpha}(i)^\dagger a_{\alpha}(j)
\end{align*}
$$
for each \(\beta \geq \alpha\) such that the right-hand sides are well-defined.

Finally, as a special case, we set \(\sP_{L,L+1}(i) = \Id_{n_L}\) and \(\sR_{L,L+1}(i,j) = \Delta'_{L}(i) \Delta'_{L}(j)\).

## Numerator of \(\alpha_*(\theta)\)

In this section, we compute the numerator of \(\alpha_*(\theta)\).

### Numerator term \(N_W(\theta)\)

To compute \(N_W(\theta)\), first observe that
$$
\begin{align*}
    d_{W_\ell} \hat{y}(\theta,x_i) \cdot \tilde{W}_\ell =
    \sP_{L,\ell+1}(i) \Delta'_\ell(i) \tilde{W}_\ell a_{\ell - 1}(i)
\end{align*}
$$
with adjoint
$$
\begin{align*}
    d_{W_\ell} \hat{y}(\theta,x_i)^* \cdot v =
    \Delta'_\ell(i) \sP_{L,\ell+1}(i)^\dagger v a_{\ell - 1}(i)^\dagger.
\end{align*}
$$
Since
$$
\begin{align*}
    d_{W_\ell} \sL(\theta) \cdot \tilde{W}_\ell
    &= \left\langle \tilde{W}_\ell, \frac{1}{M} \sum_{i=1}^{M} d_{W_\ell} \hat{y}(\theta, x_i)^* \cdot e_i \right\rangle \\
    & =\left\langle \tilde{W}_\ell, \frac{1}{M} \sum_{i=1}^{M} \Delta'_\ell(i) \sP_{L,\ell+1}(i)^\dagger e_i a_{\ell - 1}(i)^\dagger \right\rangle,
\end{align*}
$$
the corresponding partial gradient is
$$
\begin{align*}
    \nabla_{W_\ell} \sL(\theta)
    = \frac{1}{M} \sum_{i=1}^{M} \Delta'_\ell(i) \sP_{L,\ell+1}(i)^\dagger e_i a_{\ell - 1}(i)^\dagger.
\end{align*}
$$
Using these observations, we have
$$
\begin{align*}
&{}d_{W_\ell} \sL(\theta) \cdot \nabla_{W_\ell} \sL(\theta) \\
&= \frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L,\ell+1}(i) \Delta'_\ell(i)
\left(\frac{1}{M} \sum_{j=1}^{M} \Delta'_\ell(j) \sP_{L,\ell+1}(j)^\dagger e_j a_{\ell - 1}(j)^\dagger\right) a_{\ell - 1}(i), e_i \right\rangle \\
&= \frac{1}{M^2} \sum_{i,j=1}^{M} \left\langle \sP_{L,\ell+1}(i) \Delta'_\ell(i)
\Delta'_\ell(j) \sP_{L,\ell+1}(j)^\dagger e_j a_{\ell - 1}(j)^\dagger a_{\ell - 1}(i), e_i \right\rangle \\
&= \frac{1}{M^2} \sum_{i,j=1}^{M} a_{\ell-1}(i,j) \left\langle \sR_{L,\ell+1}(i,j)
e_j, e_i \right\rangle.
\end{align*}
$$

### Numerator term \(N_b(\theta)\)

To compute \(N_b(\theta)\), first observe that
$$
\begin{align*}
    d_{b_\ell} \hat{y}(\theta,x_i) \cdot \tilde{b}_\ell =
    \sP_{L,\ell+1}(i) \Delta'_\ell(i) \tilde{b}_\ell
\end{align*}
$$
with adjoint
$$
\begin{align*}
    d_{b_\ell} \hat{y}(\theta,x_i)^* \cdot v =
    \Delta'_\ell(i) \sP_{L,\ell+1}(i)^\dagger v.
\end{align*}
$$
Since
$$
\begin{align*}
    d_{b_\ell} \sL(\theta) \cdot \tilde{b}_\ell
    &= \left\langle \tilde{b}_\ell, \frac{1}{M} \sum_{i=1}^{M} d_{b_\ell} \hat{y}(\theta, x_i)^* \cdot e_i \right\rangle \\
    &= \left\langle \tilde{b}_\ell, \frac{1}{M} \sum_{i=1}^{M} \Delta'_\ell(i) \sP_{L,\ell+1}(i)^\dagger e_i \right\rangle,
\end{align*}
$$
the corresponding partial gradient is
$$
\begin{align*}
    \nabla_{b_\ell} \sL(\theta)
    = \frac{1}{M} \sum_{i=1}^{M} \Delta'_\ell(i) \sP_{L,\ell+1}(i)^\dagger e_i.
\end{align*}
$$
Using these observations, we have
$$
\begin{align*}
&{}d_{b_\ell} \sL(\theta) \cdot \nabla_{b_\ell} \sL(\theta) \\
&= \frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L,\ell+1}(i) \Delta'_\ell(i)
\left(\frac{1}{M} \sum_{j=1}^{M} \Delta'_\ell(j) \sP_{L,\ell+1}(j)^\dagger e_j \right), e_i \right\rangle \\
&= \frac{1}{M^2} \sum_{i,j=1}^{M} \left\langle \sP_{L,\ell+1}(i) \Delta'_\ell(i)
\Delta'_\ell(j) \sP_{L,\ell+1}(j)^\dagger e_j, e_i \right\rangle \\
&= \frac{1}{M^2} \sum_{i,j=1}^{M} \left\langle \sR_{L,\ell+1}(i,j) e_j, e_i \right\rangle.
\end{align*}
$$

### Numerator

Putting everything together, the numerator of \(\alpha_*(\theta)\) is

$$
\colorbox{magicmint}
{
$
\begin{align*}
    N_W(\theta) + N_b(\theta) &=
    \frac{1}{M^2}
    \sum_{\ell = 1}^{L} \sum_{i,j=1}^{M} (1 + a_{\ell - 1}(i, j)) \left\langle \sR_{L,\ell + 1}(i,j) e_j, e_i \right\rangle.
\end{align*}
$
}
$$

## Denominator of \(\alpha_*(\theta)\)

In this section, we compute the denominator of \(\alpha_*(\theta)\).
Compared to computing the numerator, the bookkeeping here
is a bit involved, so we do not show full derivations.

### Denominator term \(D_{W,W}(\theta)\)

To compute \(D_{W,W}(\theta)\), first observe that
$$
\begin{align*} d_{W_\ell} \sL(\theta) \cdot \tilde{W}_\ell
= \frac{1}{M} \sum_{i=1}^{M} \left\langle d_{W_\ell} \hat{y}(\theta,x_i) \cdot \tilde{W}_\ell, e_i \right\rangle.
\end{align*}
$$
It follows that
$$
\begin{align}
d^2_{W_\lambda,W_\ell} \sL(\theta) \cdot (\tilde{W}_\lambda, \tilde{W}_\ell)
&= \frac{1}{M} \sum_{i=1}^{M} \left\langle d_{W_\lambda} \hat{y}(\theta,x_i) \cdot \tilde{W}_\lambda, d_{W_\ell} \hat{y}(\theta,x_i) \cdot \tilde{W}_\ell \right\rangle \\
&\qquad + \, \frac{1}{M} \sum_{i=1}^{M} \left\langle d^2_{W_\lambda,W_\ell} \hat{y}(\theta,x_i) \cdot (\tilde{W}_\lambda, \tilde{W}_\ell), e_i \right\rangle.
\end{align}
$$
Using results from the previous section, the first term on the right-hand side is
$$
\begin{align*}
\frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L, \lambda + 1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda a_{\lambda - 1}(i),
\sP_{L, \ell + 1}(i) \Delta'_\ell(i) \tilde{W}_\ell a_{\ell - 1}(i) \right\rangle.
\end{align*}
$$
Working casewise, one can show that
$$
\begin{align*}
&{} d^2_{W_\lambda, W_\ell} \hat{y}(\theta,x_i) \cdot (\tilde{W}_\lambda, \tilde{W}_\ell) \\
&\quad = \, \begin{cases}
\sP_{L,\lambda + 1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda \sP_{\lambda - 1, \ell + 1}(i) \Delta'_\ell(i) \tilde{W}_\ell a_{\ell-1}(i), & \lambda > \ell, \\
0_{n_L}, & \lambda = \ell, \\
\sP_{L, \ell + 1}(i) \Delta'_\ell(i) \tilde{W}_\ell \sP_{\ell - 1, \lambda + 1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda a_{\lambda-1}(i), & \lambda < \ell
\end{cases}
\end{align*}
$$
and consequently
$$
\begin{align*}
&{} d^2_{W_\lambda,W_\ell} \sL(\theta) \cdot (\tilde{W}_\lambda, \tilde{W}_\ell) \\
\quad &= \frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L,\lambda+1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda a_{\lambda - 1}(i), \sP_{L,\ell+1}(i) \Delta'_\ell(i) \tilde{W}_\ell a_{\ell - 1}(i) \right\rangle \\
&\qquad + \,
\begin{cases}
\frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L,\lambda + 1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda \sP_{\lambda - 1, \ell + 1}(i) \Delta'_\ell(i) \tilde{W}_\ell a_{\ell-1}(i), e_i \right\rangle, & \lambda > \ell, \\
0, & \lambda = \ell, \\
\frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L, \ell + 1}(i) \Delta'_\ell(i) \tilde{W}_\ell \sP_{\ell - 1, \lambda + 1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda a_{\lambda-1}(i), e_i \right\rangle, & \lambda < \ell.
\end{cases}\end{align*}
$$
Clearly this result shows the expected symmetry of second-order partial derivatives.

Plugging in the gradients and simplifying, we arrive at
$$
\colorbox{lesserbox}
{
$
\begin{align*}
&{} D_{W,W}(\theta) \\
&= \frac{1}{M^3} \sum_{\lambda,\ell=1}^{L} \sum_{i,j,k=1}^{M} a_{\lambda - 1}(i,j) a_{\ell - 1}(i,k)
\left\langle \sR_{L,\lambda+1}(i,j) e_j, \sR_{L,\ell+1}(i,k) e_k \right\rangle \\
&\qquad + \, \frac{2}{M^3} \sum_{\lambda > \ell} \sum_{i,j,k=1}^{M} a_{\ell-1}(i,k)
\left\langle \sR_{L,\lambda + 1}(i,j) e_j a_{\lambda - 1}(j)^\dagger \sR_{\lambda-1,\ell+1}(i,k)
\sP_{L,\lambda}(k)^\dagger e_{k}, e_i \right\rangle,
\end{align*}
$
}
$$
where the "\(\lambda > \ell\)" summation is over all \(1 \leq \lambda, \ell \leq L\) such that \(\lambda > \ell\).

### Denominator term \(D_{W,b}(\theta)\)

To compute \(D_{W,b}(\theta)\), first observe that
$$
\begin{align*}
d_{b_\ell} \sL(\theta) \cdot \tilde{b}_\ell =
\frac{1}{M} \sum_{i=1}^{M} \left\langle d_{b_\ell} \hat{y}(\theta,x_i) \cdot \tilde{b}_\ell, e_i \right\rangle.
\end{align*}
$$
It follows that
$$
\begin{align*}
d^2_{W_\lambda,b_\ell} \sL(\theta) \cdot (\tilde{W}_\lambda, \tilde{b}_\ell) &= \frac{1}{M} \sum_{i=1}^{M} \left\langle d_{W_\lambda} \hat{y}(\theta,x_i) \cdot \tilde{W}_\lambda, d_{b_\ell} \hat{y}(\theta,x_i) \cdot \tilde{b}_\ell \right\rangle \\
&\quad + \, \frac{1}{M} \sum_{i=1}^{M} \left\langle d^2_{W_\lambda,b_\ell} \hat{y}(\theta,x_i) \cdot (\tilde{W}_\lambda,\tilde{b}_\ell), e_i \right\rangle.
\end{align*}
$$
Using results from the previous section, the first term on the right-hand side is
$$
\begin{align*}
\frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L, \lambda + 1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda a_{\lambda-1}(i), \sP_{L, \ell + 1}(i) \Delta'_\ell(i) \tilde{b}_\ell \right\rangle
\end{align*}
$$
Working casewise, one can show that
$$
d^2_{W_\lambda, b_\ell} \hat{y}(\theta,x_i) \cdot (\tilde{W}_\lambda, \tilde{b}_\ell) =
\begin{cases}
\sP^{i}_{L,\lambda+1}
\Delta'_\lambda(i) \tilde{W}_\lambda \sP_{\lambda - 1, \ell + 1}(i)
\Delta'_\ell(i) \tilde{b}_\ell,
& \lambda > \ell, \\
0_{n_L}, & \lambda \leq \ell
\end{cases}
$$
and consequently
$$
\begin{align*}
&{} d^2_{W_\lambda,b_\ell} \sL(\theta) \cdot (\tilde{W}_\lambda, \tilde{b}_\ell) \\
\quad &= \frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L, \lambda + 1}(i) \Delta'_\lambda(i) \tilde{W}_\lambda a_{\lambda-1}(i), \sP_{L, \ell + 1}(i) \Delta'_\ell(i) \tilde{b}_\ell \right\rangle \\
&\qquad + \, \begin{cases} \displaystyle \frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L,\lambda+1}(i) \Delta'_\lambda (i) \tilde{W}_\lambda \sP_{\lambda - 1, \ell + 1}(i) \Delta'_\ell (i) \tilde{b}_\ell, e_i \right\rangle, & \lambda > \ell, \\ 0, & \lambda \leq \ell.
\end{cases}
\end{align*}
$$
In this case, it is not immediately obvious that this
result shows the expected symmetry of second-order partial derivatives,
but this can be directly verified.

Plugging in the gradients and simplifying, we arrive at
$$
\colorbox{lesserbox}
{
$
\begin{align*}
D_{W,b}(\theta) &= \frac{1}{M^3} \sum_{\lambda,\ell=1}^{L} \sum_{i,j,k=1}^{M} a_{\lambda - 1}(i,j)
\left\langle \sR_{L,\lambda+1}(i,j) e_j, \sR_{L,\ell+1}(i,k) e_k \right\rangle \\
&\qquad + \, \frac{1}{M^3} \sum_{\lambda > \ell} \sum_{i,j,k=1}^{M} \left\langle \sR_{L,\lambda+1}(i,j) e_j a_{\lambda - 1}(j)^\dagger \sR_{\lambda - 1, \ell + 1}(i,k) \sP_{L,\lambda}(k)^\dagger e_k, e_i \right\rangle,
\end{align*}
$
}
$$
where the "\(\lambda > \ell\)" summation is defined as above.

### Denominator term \(D_{b,b}(\theta)\)

To compute \(D_{b,b}(\theta)\), first observe that
$$
\begin{align*}
d_{b_\ell} \sL(\theta) \cdot \tilde{b}_\ell =
\frac{1}{M} \sum_{i=1}^{M} \left\langle d_{b_\ell} \hat{y}(\theta,x_i) \cdot \tilde{b}_\ell, e_i \right\rangle.
\end{align*}
$$
It follows that
$$
\begin{align*}
d^2_{b_\lambda,b_\ell} \sL(\theta) \cdot (\tilde{b}_\lambda, \tilde{b}_\ell)
&= \frac{1}{M} \sum_{i=1}^{M} \left\langle d_{b_\lambda} \hat{y}(\theta,x_i) \cdot \tilde{b}_\lambda, d_{b_\ell} \hat{y}(\theta,x_i) \cdot \tilde{b}_\ell \right\rangle \\
&\quad + \, \frac{1}{M} \sum_{i=1}^{M} \left\langle d^2_{b_\lambda,b_\ell} \hat{y}(\theta,x_i) \cdot (\tilde{b}_\lambda,\tilde{b}_\ell), e_i \right\rangle.
\end{align*}
$$
Using results from the previous section, the first term on the right-hand side is
$$
\begin{align*}
\frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L, \lambda + 1}(i) \Delta'_\lambda(i) \tilde{b}_\lambda,
\sP_{L, \ell + 1}(i) \Delta'_\ell(i) \tilde{b}_\ell \right\rangle.
\end{align*}
$$
One can show that
$$
    d^2_{b_\lambda,b_\ell} \hat{y}(\theta,x_i) \equiv 0_{n_L}
$$
and consequently
$$
\begin{align*}
d^2_{b_\lambda,b_\ell} \sL(\theta) \cdot (\tilde{b}_\lambda, \tilde{b}_\ell)
&= \frac{1}{M} \sum_{i=1}^{M} \left\langle \sP_{L, \lambda + 1}(i) \Delta'_\lambda(i) \tilde{b}_\lambda, \sP_{L, \ell + 1}(i) \Delta'_\ell(i) \tilde{b}_\ell \right\rangle.
\end{align*}
$$
Clearly this result shows the expected symmetry of second-order partial derivatives.

Plugging in the gradients and simplifying, we arrive at
$$
\colorbox{lesserbox}
{
$
\begin{align*}
D_{b,b}(\theta) &=
\frac{1}{M^3} \sum_{\lambda,\ell=1}^{L}
\sum_{i,j,k=1}^{M} \left\langle \sR_{L,\lambda+1}(i,j) e_j, \sR_{L,\ell+1}(i,k) e_k \right\rangle.
\end{align*}
$
}
$$

### Denominator

Putting everything together and simplifying, the denominator of \(\alpha_*(\theta)\) is

$$
\colorbox{magicmint}
{
$
\begin{align*}
    &{} D_{W,W}(\theta) + 2 D_{W,b}(\theta) + D_{b,b}(\theta) \\
    &= \frac{1}{M^3} \sum_{\lambda,\ell=1}^{L} \sum_{i,j,k=1}^{M} (1 + a_{\lambda - 1}(i,j)) (1 + a_{\ell - 1}(i,k)) \left\langle \sR_{L,\lambda+1}(i,j) e_j, \sR_{L,\ell+1}(i,k) e_k \right\rangle \\ &\quad + \, \frac{2}{M^3} \sum_{\lambda > \ell} \sum_{i,j,k=1}^{M} (1 + a_{\ell-1}(i,k)) \left\langle \sR_{L,\lambda + 1}(i,j) e_j a_{\lambda - 1}(j)^\dagger \sR_{\lambda-1,\ell+1}(i,k) \sP_{L,\lambda}(k)^\dagger e_{k}, e_i \right\rangle.
\end{align*}
$
}
$$

## Main result

The final expression for \(\alpha_*(\theta)\) is

$$
\colorbox{magicmint}
{
$
\begin{align*}
&{} \alpha_*(\theta) \\
&= \frac{M \sum_{\ell = 1}^{L} \sum_{i,j=1}^{M} (1 + a_{\ell - 1}(i,j)) \left\langle \sR_{L,\ell + 1}(i,j) e_j, e_i \right\rangle}{
    \begin{align*}
    &{} \sum_{\lambda,\ell=1}^{L} \sum_{i,j,k=1}^{M} (1 + a_{\lambda - 1}(i,j)) (1 + a_{\ell - 1}(i,k)) \left\langle \sR_{L,\lambda+1}(i,j) e_j, \sR_{L,\ell+1}(i,k) e_k \right\rangle \\ &\quad + \, 2 \sum_{\lambda > \ell} \sum_{i,j,k=1}^{M} (1 + a_{\ell-1}(i,k)) \left\langle \sR_{L,\lambda + 1}(i,j) e_j a_{\lambda - 1}(j)^\dagger \sR_{\lambda-1,\ell+1}(i,k) \sP_{L,\lambda}(k)^\dagger e_{k}, e_i \right\rangle
    \end{align*}}.
\end{align*}
$
}
$$

As a basic sanity check, we can compare this to the \(L = 1\) result from [Part 1]({{< ref "optimal_learning_rates_one_layer" >}}).
We have
* \(a_0 (i, j) = x_i^t x_j\),
* \(\sR_{1,2} (i, j) = \Delta'_1(i) \Delta'_1(j)\), and
* The "\(\lambda > \ell\)" summation in the denominator vanishes.

The expression for \(\alpha_*(\theta)\) therefore reduces to
$$
\begin{align*}
    \alpha_*(\theta) &=
    \frac{M \sum_{i,j=1}^{M} (1 + x_i^t x_j) \left\langle \Delta'_1(i) \Delta'_1(j) e_j, e_i \right\rangle}{
    \sum_{i,j,k=1}^{M} (1 + x_i^t x_j) (1 + x_i^t x_k) \left\langle \Delta'_1(i) \Delta'_1(j) e_j,
    \Delta'_1(i) \Delta'_1(k) e_k \right\rangle
    }.
\end{align*}
$$
This matches the result from Part 1, but with different notation for the diagonal matrices.

The computation of \(\alpha_*(\theta)\) is probably too heavy for the learning rate to
be updated at each iteration of gradient descent.  It is likely that a useful approximation
exists, and this might be an interesting topic for another post.
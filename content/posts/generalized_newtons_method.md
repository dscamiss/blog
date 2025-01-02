+++
title = "Theory and implementation of the generalized Newton's method"
date = 2024-08-22T13:24:04-07:00
draft = true
tag = ['hessian', 'learning-rate', 'gradient descent', 'stochastic gradient descent', 'sgd', 'random-notes']
+++

In this post, we review the generalized Newton's method (GeN) proposed in
[1].  Then, we give a high-level overview of a PyTorch implementation, which
runs exact and approximate versions of GeN.

<!--more-->

$$
    \newcommand{\bN}{\mathbb{N}}
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\sL}{\mathscr{L}}
    \newcommand{\tr}{\mathrm{trace}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## Generalized Newton's method

Consider a smooth function \(f : \Theta \to \bR\),
where \(\Theta\) is a finite-dimensional inner product space.
Broadly speaking, gradient descent attempts to minimize \(f\) by iterating
$$
    \theta_t \leftarrow \theta_{t-1} - \alpha_t \omega(\theta_{t-1}, \nabla f(\theta_{t-1})),
$$
where
* \(\alpha_t \in \bR_{\geq 0}\) is the *learning rate* at step \(t\),
* \(\omega : \Theta \times \Theta \to \Theta\) is the *optimizer*, and
* \(\nabla f(\theta)\) is the gradient of \(f\) at \(\theta\).

For example, in stochastic gradient descent (SGD) the optimizer is simply \(\omega(\theta_1, \theta_2) = \theta_2\),
and the gradient descent iteration becomes
$$
    \theta_t \leftarrow \theta_{t-1} - \alpha_t \nabla f(\theta_{t-1}).
$$

The *generalized Newton's method* introduced in [1] is a learning rate scheduler.  That is,
the method is a scheme for choosing \(\alpha_t\) at each gradient descent iteration, with
the goal of accelerating the convergence of gradient descent. To motivate the generalized
Newton's method, suppose that \(\theta\) is fixed and introduce the function
$$
\begin{align*}
    g_\theta : \bR_{\geq 0} &\to \bR \\
    \alpha &\mapsto g_\theta(\alpha) = f(\theta - \alpha \omega(\theta, \nabla f(\theta))).
\end{align*}
$$
To keep the notation under control, we will suppress the arguments to \(\omega\),
so that
$$
    g_\theta (\alpha) = f(\theta - \alpha \omega).
$$

By composition, \(g_\theta\) is smooth, and its second-order Taylor series expansion at \(0\) is
$$
    g_\theta(\alpha) = f(\theta) - \alpha df(\theta) \cdot \omega +
    \frac{\alpha^2}{2} d^2 f(\theta) \cdot (\omega, \omega) + O(\alpha^3)
$$
for \(\alpha\) sufficiently small, say \(|\alpha| < \delta(\theta)\).

We want to minimize \(g_\theta\).   To do so, we work with the second-order approximation
$$
    g_\theta(\alpha) \approx f(\theta) - \alpha df(\theta) \cdot \omega +
    \frac{\alpha^2}{2} d^2 f(\theta) \cdot (\omega, \omega)
$$
which we will assume is valid for \(|\alpha| < \delta(\theta)\).

From single-variable calculus, we know that if
$$
    d^2 f(\theta) \cdot (\omega, \omega) > 0,
$$
then the second-order approximating function attains its global minimum at
$$
    \alpha_*(\theta) =
    \frac{\mathrm{num}(\theta)}{\mathrm{den}(\theta)}
    =
    \frac{
        df(\theta) \cdot \omega
    }{
        d^2 f(\theta) \cdot (\omega, \omega)
    }.
$$
Following [1], we say that \(\alpha_*(\theta)\) the *optimal learning rate* at
\(\theta\).

*Note*: The name "optimal learning rate" is somewhat misleading, since the derivation of
\(\alpha_*(\theta)\) is based on a local approximation. Indeed, even though
\(g_\theta(\theta)\) may be quasi-parabolic globally, its Taylor series expansion at \(0\) may
not accurately capture the global parabolic shape.  In this way, \(\alpha_*(\theta)\)
can be far from the true optimal learning rate.

*Note*: For comparison, the above equation is Equation (3.1) in [1], although they use different
notation -- for example, they denote \(\alpha_*(\theta)\) by \(\eta^*_t\).

We can directly use \(\alpha_*(\theta)\) to drive a learning rate scheduler, for
example by setting
$$
    \alpha_t = \alpha_*(\theta_t).
$$
In this case, the gradient descent iteration becomes
$$
    \theta_t =
    \theta_{t-1} -
    \frac{
        df(\theta) \cdot \omega
    }{
        d^2 f(\theta) \cdot (\omega, \omega)
    }
    \omega.
$$
This learning rate scheduler is the *exact generalized Newton's method (exact GeN)*.

*Note*: Practically speaking, \(\theta\) represents trainable model parameters lumped into a single
parameter vector. We can rephrase the calculation of \(\alpha_*(\theta)\) in terms of the individual
parameter vectors, as follows.   First, suppose that \(\Theta\) is the product inner product space
$$
    \Theta = \Theta_1 \times \cdots \times \Theta_n
$$
with generic element \(\theta = (\theta_1,\dots,\theta_n)\).
Then we can write
$$
    \alpha_*(\theta) =
    \frac{
        \sum_{i=1}^{n} d_{\theta_i} f(\theta) \cdot \omega_i
    }{
        \sum_{i,j=1}^{n} d^2_{\theta_i,\theta_j} f(\theta) \cdot (\omega_i, \omega_j)
    }.
$$
Here we have introduced the following notation:
* \(\nabla_{\theta_i} f(\theta)\) is the partial gradient of \(f\) with respect to \(\theta_i\) at \(\theta\),
* \(d^2_{\theta_i,\theta_j} f(\theta)\) is the second-order partial derivative of \(f\) with respect to \((\theta_i,\theta_j)\) at \(\theta\), and
* \(\omega = (\omega_1,\dots,\omega_n)\), with the same component division as \(\theta\).

To avoid the computational burden of computing second-order partial derivatives
(in other words, computing full Hessians or Hessian-vector products),
the authors of [1] propose the following "backpropagation-free" differencing scheme.

**Step 1**: At the \(t\)-th gradient descent iteration, set
$$
\begin{align*}
    f_- &= f(\theta_t - \alpha_{t-1} \omega_t) \\
    f_0 &= f(\theta_t) \\
    f_+ &= f(\theta_t + \alpha_{t-1} \omega_t),
\end{align*}
$$
where \(\omega_t = \omega(\theta_t, \nabla f(\theta_t))\) and \(\alpha_0 > 0\) is chosen *a priori*.

**Step 2**: Fit a second-order polynomial to
$$
    \{ (-\alpha_{t-1}, f_-), (0, f_0), (\alpha_{t-1}, f_+) \}.
$$

**Step 3**: If the second-order polynomial fit is convex, then set
$$
    \Delta_t = \frac{\alpha_{t-1}}{2} \frac{f_+ - f_-}{f_+ - 2 f_0 + f_-}
$$
and
$$
    \alpha_t = \gamma \alpha_{t-1} + (1 - \gamma) \Delta_t,
$$
where the moving average coefficient \(0 \leq \gamma < 1\) is chosen *a priori*.
On the other hand, if the second-order polynomial fit is not convex, then set
\(\alpha_t\) to a small default learning rate.

This learning rate scheduler is the *approximate generalized Newton's method (approximate GeN)*.

*Note*: Approximate GeN is "backpropagation-free" in the sense that it does not use
_additional_ backpropagation steps to compute second-order partial derivatives.
Of course, backpropagation is still used to compute the parameter gradients
passed to each \(\omega_t\).

*Note*: As noted in [1], assuming that \(\alpha_*(\theta_t)\) varies slowly in \(t\), we can recompute
\(\alpha_t\) periodically to amortize the extra computation needed for approximate GeN.

*Note*: In an implementation of exact or approximate GeN, care should be taken to ensure that each
\(\alpha_t\) is in the interval where the second-order approximation
is declared (or determined to be) valid.  If \(\alpha_t\) is outside this interval,
then we can revert to a small default learning rate.


## Fully-connected neural networks

For \(k\)th-order partial derivatives, when \(i_1 = \cdots = i_k = i\) we will write
$$
    d^k_{\theta_i} f(\theta) \equiv d^k_{\theta_{i_1},\dots,\theta_{i_k}} f(\theta).
$$


We begin by defining the *parameter space* to be
$$
    \Theta = \bR^{n_1 \times n_0} \times \bR^{n_1}
$$
with generic element \(\theta = (W_1,b_1)\).

The *activation function* is a piecewise-linear function \(\sigma_0 : \bR \to \bR\).

The *activation map* \(\sigma: \bR^{n_1} \to \bR^{n_1}\) is defined by
$$
\begin{align*}
    \sigma(x) &= (\sigma_0(x^1), \dots, \sigma_0(x^{n_1}))^t.
\end{align*}
$$
Note that the component indices of \(x\) are written with superscripts.

Wherever it is well-defined, the total derivative of \(\sigma\) at \(x\) is
$$
    d\sigma(x) \cdot h = \Delta \sigma'(x) h,
$$
where
$$
    \sigma'(x) = (\sigma_0'(x^1), \dots, \sigma_0'(x^{n_1}))^t
$$
and \(\Delta : \bR^{n_1} \to \bR^{n_1 \times n_1}\) sends \(x\) to the
diagonal matrix whose \((i,i)\)th component is \(x^i\).

Similarly, wherever it is well-defined, the total derivative of \(\sigma'\) at \(x\) is
$$
    d\sigma'(x) \cdot h = \Delta \sigma''(x) h = 0_{n_1},
$$
since \(\sigma_0\) is piecewise-linear.

Consider the single-layer, fully-connected neural network
$$
\begin{align*}
    \hat{y} : \Theta \times \bR^{n_0} &\to \bR^{n_1}
\end{align*}
$$
defined by
$$
\begin{align*}
    \hat{y}(\theta, x) &= \sigma (W_1 x + b_1).
\end{align*}
$$
For notational convenience, we introduce the *intermediate computation map*
$$
\begin{align*}
    z_1 : \Theta \times \bR^{n_0} &\to \bR^{n_1}
\end{align*}
$$
defined by
$$
\begin{align*}
    z_1(\theta, x) &= W_1 x + b_1.
\end{align*}
$$

In the rest of this post, we will assume that we are working
away from those points where the activation map is not differentiable.
This is necessary since piecewise-linear functions are not
everywhere differentiable, in general.

We will need the first- and second-order partial derivatives of \(\hat{y}\) with
respect to \(W_1, b_1\).

The first-order partial derivatives of \(\hat{y}\) at \((\theta, x)\) are
$$
\begin{align*}
    d_{W_1} \hat{y}(\theta,x) \cdot W
    &= d \sigma(z_1(\theta,x)) \circ d_{W_1} z_1(\theta,x) \cdot W \\
    &= \Delta \sigma'(z_1(\theta,x)) W x
\end{align*}
$$
and
$$
\begin{align*}
    d_{b_1} \hat{y}(\theta,x) \cdot b
    &= d \sigma(z_1(\theta,x)) \circ d_{b_1} z_1(\theta,x) \cdot b \\
    &= \Delta \sigma'(z_1(\theta,x)) b.
\end{align*}
$$
The second-order partial derivatives of \(\hat{y}\) at \((\theta, x\)) are
$$
\begin{align*}
    d^2_{W_1} \hat{y}(\theta,x) \cdot (V,W)
    &= \Delta [d \sigma'(z_1(\theta,x)) \circ \cdots] W x = 0_{n_1}
\end{align*}
$$
and
$$
\begin{align*}
    d^2_{b_1} \hat{y}(\theta,x) \cdot (a,b)
    &= \Delta [d \sigma'(z_1(\theta,x)) \circ \cdots] b = 0_{n_1}
\end{align*}
$$
and finally
$$
\begin{align*}
    d^2_{W_1,b_1} \hat{y}(\theta,x) \cdot (W,b)
    &= \Delta [d \sigma'(z_1(\theta,x)) \circ \cdots] b = 0_{n_1}.
\end{align*}
$$

Turning to optimal learning rates, suppose that we have (mini-batch) training data
$$
\begin{align*}
    \{ (x_i, y_i) \}_{i=1}^{M}, \quad
    x_i \in \bR^{n_0}, \quad y_i \in \bR^{n_1}
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
where \(\langle\cdot,\cdot\rangle\) is the Euclidean inner product on \(\bR^{n_1}\).  To
make use of the results in the previous section,
\(\sL\) must be equal to its second-order Taylor series expansion at \(\theta\).  For
now, let's assume this is true; it will be proven below
after partial derivative computations.

Wherever it is well-defined, the optimal learning rate for \(\sL\) at \(\theta\) is
$$
\begin{align*}
    \alpha_*(\theta) &=
    \frac{\mathrm{num}(\theta)}{\mathrm{den}(\theta)},
\end{align*}
$$
where
$$
\begin{align*}
    \mathrm{num}(\theta) &=
        d_{W_1} \sL(\theta) \cdot \nabla_{W_1} \sL(\theta) +
        d_{b_1} \sL(\theta) \cdot \nabla_{b_1} \sL(\theta) \\
    \mathrm{den}(\theta) &=
        d^2_{W_1} \sL(\theta) \cdot (\nabla_{W_1} \sL(\theta), \nabla_{W_1} \sL(\theta)) \\
        &\qquad + \, 2 d^2_{W_1,b_1} \sL(\theta) \cdot (\nabla_{W_1} \sL(\theta), \nabla_{b_1} \sL(\theta)) \\
        &\qquad + \, d^2_{b_1} \sL(\theta) \cdot (\nabla_{b_1} \sL(\theta), \nabla_{b_1} \sL(\theta)).
\end{align*}
$$
Recall that for \(\alpha_*(\theta)\) to be "well-defined,"
we must have \(\mathrm{den}(\theta) > 0\).

To compute \(\alpha_*\), we start with the partial gradients.  For convenience, we set
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    e_i &= \hat{y}(\theta, x_i) - y_i \\
    \Delta'(z_{1,i}) &= \Delta \sigma'(z_1(\theta, x_i)).
\end{align*}
$
}
$$
Observe that, using the cyclic property of trace, we have
$$
\begin{align*}
    \langle v, d_{W_1} \hat{y}(\theta, x_i) \cdot W \rangle
    &= \tr(v^t \Delta' (z_{1,i}) Wx_i) \\
    &= \tr(x_i v^t \Delta' (z_{1,i}) W) \\
    &= \langle \Delta' (z_{1,i}) v x_i^t, W \rangle_F,
\end{align*}
$$
where \(\langle A, B \rangle_F = \tr(A^t B)\) is the Frobenius inner product.  This shows that
$$
\begin{align*}
    d_{W_1} \hat{y}(\theta, x_i)^* \cdot v = \Delta' (z_{1,i}) v x_i^t,
\end{align*}
$$
where the superscript "\(*\)" denotes adjoint.  Using this, we obtain
$$
\begin{align*}
    d_{W_1} \sL(\theta) \cdot W
    &= \frac{1}{M} \sum_{i=1}^{M} \langle d_{W_1} \hat{y}(\theta, x_i) \cdot W, e_i \rangle \\
    &= \left\langle
        W, \frac{1}{M} \sum_{i=1}^{M} d_{W_1} \hat{y}(\theta, x_i)^* \cdot e_i
    \right\rangle_F \\
    &= \left\langle
        W, \frac{1}{M} \sum_{i=1}^{M} \Delta'(z_{1,i}) e_i x_i^t
    \right\rangle_F.
\end{align*}
$$
Recalling that \(\nabla_{W_1} \sL(\theta)\) is the unique element of \(\bR^{n_1 \times n_0}\) satisfying
$$
    d_{W_1} \sL(\theta) \cdot W = \left\langle W, \nabla_{W_1} \sL(\theta) \right\rangle_F,
$$
we have
$$
\begin{align*}
    \nabla_{W_1} \sL(\theta) = \frac{1}{M} \sum_{i=1}^{M} \Delta'(z_{1,i}) e_i x_i^t
\end{align*}
$$
and the first term in \(\mathrm{num}(\theta)\) is
$$
\begin{align*}
    d_{W_1} \sL(\theta) \cdot \nabla_{W_1} \sL(\theta) &=
    \frac{1}{M^2} \sum_{i,j=1}^{M} \langle \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j x_j^t x_i, e_i \rangle.
\end{align*}
$$
In a similar way, we can compute
$$
\begin{align*}
    \nabla_{b_1} \sL(\theta) = \frac{1}{M} \sum_{i=1}^{M} \Delta'(z_{1,i}) e_i
\end{align*}
$$
and the second term in \(\mathrm{num}(\theta)\) is
$$
\begin{align*}
    d_{b_1} \sL(\theta) \cdot \nabla_{b_1} \sL(\theta) &=
    \frac{1}{M^2} \sum_{i,j=1}^{M} \langle \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j, e_i \rangle.
\end{align*}
$$
Combining terms, we have
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    \mathrm{num}(\theta) = \frac{1}{M^2} \sum_{i,j=1}^{M} (1 + x_i^t x_j)
    \langle \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j, e_i \rangle.
\end{align*}
$
}
$$
As for the denominator of \(\alpha_*(\theta)\), we begin by computing
$$
\begin{align*}
    d^2_{W_1} \sL(\theta) \cdot (V, W)
    &= \frac{1}{M} \sum_{i=1}^{M}
    \langle d^2_{W_1} \hat{y}(\theta, x_i) \cdot (V, W), e_i \rangle \\
    &\qquad+ \, \frac{1}{M} \sum_{i=1}^{M} \langle d_{W_1} \hat{y}(\theta, x_i) \cdot V, d_{W_1} \hat{y}(\theta, x_i) \cdot W \rangle \\
    &= \frac{1}{M} \sum_{i=1}^{M} \langle \Delta'(z_{1,i}) V x_i, \Delta'(z_{1,i}) W x_i \rangle.
\end{align*}
$$
Analogously, the other second-order partial derivatives are
$$
\begin{align*}
    d^2_{W_1,b_1} \sL(\theta) \cdot (W,b) &=
    \frac{1}{M} \sum_{i=1}^{M}
    \langle \Delta'(z_{1,i}) W x_i, \Delta'(z_{1,i}) b \rangle
\end{align*}
$$
and
$$
\begin{align*}
    d^2_{b_1} \sL(\theta) \cdot (a, b) &=
    \frac{1}{M} \sum_{i=1}^{M}
    \langle \Delta'(z_{1,i}) a, \Delta'(z_{1,i}) b \rangle.
\end{align*}
$$
Note that the above results imply \(d^3 \sL(\theta) \equiv 0\) and consequently
\(\sL\) is equal to its second-order Taylor series expansion at \(\theta\).
To see this, observe (for example) that
$$
\begin{align*}
    d^3_{W_1} \sL(\theta) \cdot (U,V,W) &=
    \frac{1}{M} \sum_{i=1}^{M} \langle \Delta [d\sigma'(z_{1,i}) \circ \cdots] V x_i, \cdots \rangle \\
    &\qquad + \, \frac{1}{M} \sum_{i=1}^{M} \langle \cdots, \Delta [d\sigma'(z_{1,i}) \circ \cdots] W x_i \rangle
    = 0.
\end{align*}
$$
Similarly, the other third-order partial derivatives are also identically equal to \(0\).

Plugging in the partial gradients, we obtain
$$
\begin{align*}
    &{} d^2_{W_1} \sL(\theta) \cdot (\nabla_{W_1} \sL(\theta), \nabla_{W_1} \sL(\theta)) \\
    &= \frac{1}{M^3} \sum_{i,j,k=1}^{M}
    \langle
        \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j x_j^t x_i,
        \Delta'(z_{1,i}) \Delta'(z_{1,k}) e_k x_k^t x_i
    \rangle
\end{align*}
$$
and
$$
\begin{align*}
    &{} d^2_{b_1} \sL(\theta) \cdot (\nabla_{b_1} \sL(\theta), \nabla_{b_1} \sL(\theta)) \\
    &= \frac{1}{M^3} \sum_{i,j,k=1}^{M}
    \langle
        \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j,
        \Delta'(z_{1,i}) \Delta'(z_{1,k}) e_k
    \rangle.
\end{align*}
$$
Finally,
$$
\begin{align*}
    &{} d^2_{W_1,b_1} \sL(\theta) \cdot (\nabla_{W_1} \sL(\theta), \nabla_{b_1} \sL(\theta)) \\
    &= \frac{1}{M^3} \sum_{i,j,k=1}^{M}
    \langle
        \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j x_j^t x_i,
        \Delta'(z_{1,i}) \Delta'(z_{1,k}) e_k
    \rangle.
\end{align*}
$$
Combining terms (and making a dummy index swap), the denominator of \(\alpha_*(\theta)\) is
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    \mathrm{den}(\theta) &=
    \frac{1}{M^3} \sum_{i,j,k=1}^{M}
    (1 + x_i^t x_j)
    (1 + x_i^t x_k)
    \langle
        \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j,
        \Delta'(z_{1,i}) \Delta'(z_{1,k}) e_k
    \rangle.
\end{align*}
$
}
$$
Putting everything together, the optimal learning rate for \(\sL\) at \(\theta\) is
$$
\colorbox{magicmint}
{
$
\alpha_*(\theta) =
\frac{
    M \sum_{i,j=1}^{M} (1 + x_i^t x_j)
        \langle \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j, e_i \rangle
}{
    \sum_{i,j,k=1}^{M}
        (1 + x_i^t x_j)
        (1 + x_i^t x_k)
        \langle
            \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j,
            \Delta'(z_{1,i}) \Delta'(z_{1,k}) e_k
        \rangle
}.
$
}
$$
For stochastic gradient descent (i.e., \(M = 1\)), this expression simplifies to
$$
\colorbox{lesserbox}
{
$
\alpha_*(\theta) =
\frac{
    \langle \Delta'(z_{1,1}) \Delta'(z_{1,1}) e_1, e_1 \rangle
}{
    (1 + x_1^t x_1)
    \langle
        \Delta'(z_{1,1}) \Delta'(z_{1,1}) e_1,
        \Delta'(z_{1,1}) \Delta'(z_{1,1}) e_1
    \rangle
}.
$
}
$$
Further, if the activation function is standard ReLU, then clearly
$$
\Delta'(z_{1,1}) \Delta'(z_{1,1}) = \Delta'(z_{1,1})
$$
and we have
$$
\colorbox{lesserbox}
{
$
\alpha_*(\theta) =
\frac{
    1
}{
    1 + x_1^t x_1
}.
$
}
$$
This directly relates the optimal learning rate to the norm of the input vector.

In another post, we will address the practical impact of using the
optimal learning rates.  If there is a tangible benefit
(i.e., faster training), then it might be worth pursuing a similar
analysis for multi-layer fully-connected neural networks.
One challenge in this direction is that we can no longer rely on the
exactness of the second-order Taylor series expansion (indeed, this fails to be
exact even in the two-layer case).
Nevertheless, the expansion might give
a useful approximation, or some other conditions can be imposed to make it
exact.

## Implementation

## References

[1] Zi Bu and Shiyun Xu, *Automatic gradient descent with generalized Newton's method*,
[arXiv:2407.02772](https://arxiv.org/abs/2407.02772) [cs.LG]

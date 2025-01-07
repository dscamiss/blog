+++
title = "The generalized Newton's method for learning rate selection"
date = 2024-12-25T13:24:04-07:00
draft = false
tag = ['hessian', 'learning-rate', 'learning-rate-selection', 'gradient-descent', 'sgd', 'random-notes']
+++

In this post, we review the generalized Newton's method (GeN) proposed in
[1].  Then, we explicitly compute the learning rates prescribed by the exact
version of GeN, for a simple problem instance.  Then, we give a high-level
overview of a PyTorch implementation which runs the exact version of GeN for
stochastic gradient descent.

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
the method is a prescription for how to choose \(\alpha_t\) at each gradient descent iteration, with
the goal of accelerating the convergence of gradient descent. To motivate the generalized
Newton's method, suppose that \(\theta\) is fixed and define
$$
\begin{align*}
    g_\theta : \bR &\to \bR \\
    \alpha &\mapsto g_\theta(\alpha) = f(\theta - \alpha \omega),
\end{align*}
$$
where we have introduced the shorthand
$$
\omega = \omega(\theta, \nabla f(\theta)).
$$
By composition, \(g_\theta\) is smooth, and its second-order Taylor series expansion at \(0\) is
$$
    g_\theta(\alpha) = f(\theta) - \alpha df(\theta) \cdot \omega +
    \frac{\alpha^2}{2} d^2 f(\theta) \cdot (\omega, \omega) + O(\alpha^3)
$$
for \(\alpha\) sufficiently small, say \(|\alpha| < \delta(\theta)\).

To minimize \(g_\theta\), we will work with the second-order approximating function
$$
    \alpha \mapsto f(\theta) - \alpha df(\theta) \cdot \omega +
    \frac{\alpha^2}{2} d^2 f(\theta) \cdot (\omega, \omega).
$$
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
Following [1], we say that \(\alpha_*(\theta)\) is the *optimal learning rate* for \(f\) at \(\theta\).

*Note*: The name "optimal learning rate" is somewhat misleading, since the derivation of
\(\alpha_*(\theta)\) is based on a local approximation. Indeed, even though
\(g_\theta\) may be quasi-parabolic globally, its Taylor series expansion at \(0\) may
not accurately capture the global parabolic shape.  In this way, \(\alpha_*(\theta)\)
can be far from the point at which the second-order approximating function attains
its global minimum.
![Inaccuracy in optimal learning rates](/fully_connected_example.png)
The above figure shows an example of this phenomenon, for a particular case where
\(f\) is a loss function.  The blue curve
is \(g_\theta\), and the green curve is the second-order approximating function.
The point at which the second-order approximating function attains
its global minimum is marked with the vertical line. \(\diamond\)

*Note*: For comparison purposes, the above equation is Equation (3.1) in [1], although they use different notation -- for example, they write \(\eta^*_t\) instead of \(\alpha_*(\theta)\).
\(\diamond\)

We can use the optimal learning rates to drive a learning rate scheduler, by setting
$$
    \alpha_t = \alpha_*(\theta_t).
$$
In this case, the gradient descent iteration becomes
$$
    \theta_t =
    \theta_{t-1} -
    \frac{
        df(\theta_t) \cdot \omega_t
    }{
        d^2 f(\theta_t) \cdot (\omega_t, \omega_t)
    }
    \omega_t,
$$
where we have introduced the shorthand
$$
    \omega_t = \omega(\theta_t, \nabla f(\theta_t)).
$$
This learning rate scheduler is the *exact generalized Newton's method (exact GeN)*.

To avoid the computational burden of computing second-order partial derivatives
(in other words, computing full Hessians or Hessian-vector products),
the authors of [1] propose the following "backpropagation-free" differencing scheme:
Instead of computing \(\alpha_*(\theta_t)\) at each gradient descent iteration,
compute
$$
    \Delta_t = \frac{\alpha_{t-1}}{2}
    \frac{
        g_{\theta_t}(-\alpha_{t-1}) - g_{\theta_t}(\alpha_{t-1})
    }{
        g_{\theta_t}(-\alpha_{t-1}) - 2 g_{\theta_t}(0) + g_{\theta_t}(\alpha_{t-1})
    }.
$$
If the denominator of \(\Delta_t\) is positive, then set
$$
    \alpha_t = \gamma \alpha_{t-1} + (1 - \gamma) \Delta_t,
$$
where the moving average coefficient \(0 \leq \gamma < 1\) is chosen *a priori*.
On the other hand, if the denominator of \(\Delta_t\) is not positive, then set
\(\alpha_t\) to a small default learning rate.

This learning rate scheduler is the *approximate generalized Newton's method (approximate GeN)*.
To see that the definition of \(\Delta_t\) is correct, observe that
$$
\begin{align*}
    \frac{g_{\theta}(-\alpha) - g_{\theta}(\alpha)}{2 \alpha} = df(\theta) \cdot \omega
\end{align*}
$$
and
$$
\begin{align*}
    \frac{g_{\theta}(-\alpha) - 2 g_{\theta}(0) + g_{\theta}(\alpha)}{\alpha^2} = d^2 f(\theta) \cdot (\omega, \omega).
\end{align*}
$$

*Note*: As described in [1], other schemes for calculating \(\Delta_t\) are possible -- for example, we could
compute a polynomial fit which incorporates more data points.

*Note*: Approximate GeN is "backpropagation-free" in the sense that it does not use
_additional_ backpropagation steps to compute second-order partial derivatives.
Of course, backpropagation is still used to compute the parameter gradients
passed to \(\omega_t\). \(\diamond\)

*Note*: As noted in [1], assuming that \(\alpha_*(\theta_t)\) is slowly varying
with \(t\), we can recompute
\(\alpha_t\) periodically to amortize the extra computation needed for exact or
approximate GeN. \(\diamond\)

*Note*: In an implementation of exact or approximate GeN, care should be taken to ensure that each
\(\alpha_t\) is in the interval where the second-order approximation
is declared (or somehow determined to be) valid.  If \(\alpha_t\) is outside this interval,
then a reasonable policy is to set \(\alpha_t\) to a small default learning rate. \(\diamond\)

*Note*: In the machine-learning context, \(\theta\) represents trainable model parameters lumped into a single
parameter vector. We can rephrase the calculation of \(\alpha_*(\theta)\) in terms of the individual
parameter vectors, as follows.   First, suppose that \(\Theta\) is the product inner product space
\(\Theta = \Theta_1 \times \cdots \times \Theta_n\)
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
* \(\nabla_{\theta_i} f(\theta)\) is the partial gradient of \(f\) with respect to \(\theta_i\) at \(\theta\), and
* \(d^2_{\theta_i,\theta_j} f(\theta)\) is the second-order partial derivative of \(f\) with respect to \((\theta_i,\theta_j)\) at \(\theta\).

We will see an example of this in the next section. \(\diamond\)

## Example: Exact GeN for single-layer fully-connected neural networks

In this section, we compute optimal learning rates prescribed by exact GeN.
The focus here is a simple problem instance, where we have a single-layer fully-connected neural network activated by a piecewise-linear function
(for example, ReLU), the objective is to minimize the norm-squared loss function,
and the optimizer is SGD.

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
Note that we have written \(d^2_{W_1} \hat{y}\) instead of \(d^2_{W_1, W_1} \hat{y}\)
and similarly for the \(b_1\) derivative.

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

Since the optimier is SGD, the optimal learning rate for \(\sL\) at \(\theta\) is
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
Recall that for \(\alpha_*(\theta)\) to be well-defined,
we must have \(\mathrm{den}(\theta) > 0\).

To compute \(\alpha_*(\theta)\), we start with the partial gradients.  For convenience, we set
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
For the case \(M = 1\), this expression simplifies to
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

## Implementation of exact GeN for SGD

This snippet computes first-order approximation coefficients.

```python
def norm_of_tensor_dict(tensor_dict: _TensorDict, p: float = 2.0) -> _Scalar:
    """Helper function to sum the norms of each tensor in a dictionary.

    Args:
        tensor_dict: Dictionary containing only tensors.
        ord: Order of the norm (default = 2.0).

    Returns:
        Scalar tensor with 2-norm.
    """
    tensors = tensor_dict.values()
    return sum(linalg.vector_norm(tensor, p) ** 2.0 for tensor in tensors)


def first_order_approximation_coeffs(
    model: nn.Module, criterion: CriterionType, x: Real[Tensor, "..."], y: Real[Tensor, "..."]
) -> tuple[_ScalarTwoTuple, _TensorDict]:
    """Compute coefficients of first-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        x: Input tensor.
        y: Output tensor (target).

    Returns:
        Tuple containing:
            - Tuple of scalar tensors with approximation coefficients.
            - Dictionary with model parameter gradients.  This can be ignored,
              since it is only to avoid code duplication in the second-order
              approximation code.
    """
    # Extract parameters from `model` to pass to `torch.func.functional_call()`
    params_dict = dict(model.named_parameters())

    # Wrapper function for parameter-dependent loss
    def parameterized_loss(params_dict):
        y_hat = functional_call(model, params_dict, (x,))
        return criterion(y_hat, y)

    with torch.no_grad():
        # Polynomial coefficients
        coeff_0 = parameterized_loss(params_dict)
        coeff_1 = torch.as_tensor(0.0)

        # Compute parameter gradients
        grad_params_dict = grad(parameterized_loss)(params_dict)

        # Compute first-order coefficient
        coeff_1 = norm_of_tensor_dict(grad_params_dict)

    return (coeff_0, -coeff_1), grad_params_dict
```

This snippet builds on the previous one, to compute second-order approximation coefficients.

```python
def second_order_approximation_coeffs(
    model: nn.Module, criterion: CriterionType, x: Real[Tensor, "..."], y: Real[Tensor, "..."]
) -> _ScalarThreeTuple:
    """Compute coefficients of second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        x: Input tensor.
        y: Output tensor (target).

    Returns:
        Tuple of scalar tensors with approximation coefficients.
    """

    # Wrapper function for parameter-dependent loss
    # - This version is compatible with `make_functional()`, which is needed
    #   for the call to `torch.autograd.functional.vhp()`.  PyTorch issues a
    #   warning about using `make_functional()`, but there seems to be no
    #   analogue of `torch.autograd.functional.vhp()` which can be used with
    #   `torch.func.functional_call()`.
    def parameterized_loss(*params):
        model_func, _ = make_functional(model)
        y_hat = model_func(params, x)
        return criterion(y_hat, y)

    with torch.no_grad():
        coeffs, grad_params_dict = first_order_approximation_coeffs(model, criterion, x, y)
        coeff_2 = torch.as_tensor(0.0)

        # Compute second-order coefficient
        params = tuple(model.parameters())
        grad_params = tuple(grad_params_dict.values())
        _, prod = vhp(parameterized_loss, params, grad_params)

        for i, grad_param in enumerate(grad_params):
            coeff_2 += torch.dot(grad_param.flatten(), prod[i].flatten())

    # Note: Minus was already applied to first-order coefficient
    return (coeffs[0], coeffs[1], coeff_2 / 2.0)
```

Now we can subclass `torch.optim.lr_scheduler.LRScheduler` to implement
exact GeN.

```python
class ExactGeNForSGD(LRScheduler):
    """Exact GeN for SGD.

    Args:
        optimizer: Optimizer.
        last_epoch: Number of last epoch.
        model: Network model.
        criterion: Loss criterion function.
        lr_min: Minimum learning rate to use.
        lr_max: Maximum learning rate to use.
    """

    _DEFAULT_LR = 1e-3

    def __init__(  # noqa: DCO010
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int,
        model: nn.Module,
        criterion: CriterionType,
        lr_min: float,
        lr_max: float,
    ) -> None:
        super().__init__(optimizer, last_epoch)

        self.model = model
        self.criterion = criterion
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_lrs = self.base_lrs.copy()

    # Pylint complains that redefinition of step() has a different signature
    def step(  # pylint: disable=arguments-renamed
        self, x: Optional[Real[Tensor, "..."]] = None, y: Optional[Real[Tensor, "..."]] = None
    ) -> list[float]:
        """Update learning rate(s) in the optimizer.

        Args:
            x: Input tensor.
            y: Output tensor (target).

        Returns:
            List of learning rates for each parameter group.
        """
        lrs = self.get_lr(x, y)

        # Update learning rates in the optimizer
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

        return lrs

    def get_lr(
        self, x: Optional[Real[Tensor, "..."]] = None, y: Optional[Real[Tensor, "..."]] = None
    ) -> list[float]:
        """Compute learning rate(s) for a particular batch.

        Args:
            x: Input tensor.
            y: Output tensor (target).

        Returns:
            List of learning rates for each parameter group.
        """
        # Handle initial step (in this case, `x` and `y` are not available)
        if x is None and y is None:
            lr = self._DEFAULT_LR
        else:
            # Get coefficients of second-order approximation
            coeffs = second_order_approximation_coeffs(self.model, self.criterion, x, y)
            coeffs = [coeff.item() for coeff in coeffs]

            if coeffs[2] <= 0.0:
                # Approximation is concave --> use default learning rate
                lr = self._DEFAULT_LR
            else:
                # Approximation is convex --> use alpha_star
                alpha_star = -coeffs[1] / (2.0 * coeffs[2])
                lr = min(self.lr_max, max(alpha_star, self.lr_min))

        # Update current learning rate(s)
        num_groups = len(self.optimizer.param_groups)
        self.current_lrs = [lr for _ in range(num_groups)]

        return self.current_lrs

    def get_last_lr(self) -> list[float]:  # noqa
        return self.current_lrs
```

The next snippet shows how to use exact GeN for SGD in a standard training loop.

```python
# Make SGD optimizer
optimizer = optim.SGD(model.parameters())

# Make exact GeN for SGD
scheduler = ExactGeNForSGD(
    optimizer, -1, model, criterion, config.lr_min, config.lr_max
)

<...>

model.train()

for epoch in range(config.num_epochs):
    for x, y in dataloader:
        # Move data to device
        x = x.to(device)
        y = y.to(device)

        # Zero model parameter gradients
        optimizer.zero_grad()

        # Run forward pass
        y_hat = model(x)

        # Compute loss
        loss = criterion(y_hat, y)

        # Run backward pass
        loss.backward()

        # Adjust learning rate(s) in optimizer
        scheduler.step(x, y)

        # Adjust model parameters using new learning rate(s)
        optimizer.step()
```

## References

[1] Zi Bu and Shiyun Xu, *Automatic gradient descent with generalized Newton's method*,
[arXiv:2407.02772](https://arxiv.org/abs/2407.02772) [cs.LG]

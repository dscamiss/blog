+++
title = 'Optimal learning rates in a special class of fully-connected neural networks: The single-layer case'
date = 2024-08-22T13:24:04-07:00
draft = false
tag = ['fully-connected', 'learning-rate', 'random-notes']
+++

$$
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\sL}{\mathscr{L}}
    \newcommand{\tr}{\mathrm{trace}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

In this post, we explicitly compute the optimal learning rates
for a special class of fully-connected neural networks.

The class consists of single-layer fully-connected neural networks
with piecewise-linear activation functions.  We assume that the
optimization scheme is mini-batch gradient descent.

The focus on piecewise-linear activation functions is motivated by two things:
First, such activation functions are widely used in practice (for example, standard
ReLU and leaky ReLU). Second, such activation functions lead to simpler second-order partial derivatives
of the loss function, and these partial derivatives are needed to compute the optimal learning rates.

## Optimal learning rates in gradient descent

Consider a smooth function \(f : \Theta \to \bR\),
where \(\Theta\) is a finite-dimensional vector space.
Gradient descent attempts to minimize \(f\) by iterating on
$$
    \theta \leftarrow \theta - \alpha df(\theta)^*.
$$
Here, \(\alpha \in \bR_{> 0}\) is a fixed *learning rate*,
\(df(\theta)\) is the total derivative of \(f\) at \(\theta\),
the superscript \(*\) denotes adjoint, and \(\theta\) is
given a suitable initial value.

Suppose that \(\theta \in \Theta\) is fixed.  We want to choose \(\alpha\) such that
$$
    f(\theta - \alpha df(\theta)^*)
$$
is minimized.  Equivalently, we want to minimize the composite function
$$
    f \circ g : \bR_{> 0} \to \bR,
$$
where the map \(g : \bR_{> 0} \to \Theta\) is defined by
$$
    g(\alpha) = \theta - \alpha df(\theta)^*.
$$

Now assume that \(f\) is equal to its second-order Taylor series expansion at \(\theta\):
$$
    f(\theta + h)
    = f(\theta) + df(\theta) \cdot h + \frac{1}{2} d^2 f(\theta) \cdot (h, h).
$$
Here, \(d^2 f(\theta)\) is the second-order total derivative of \(f\) at \(\theta\).

Under this assumption,
$$
\begin{align*}
    (f \circ g)(\alpha)
    &= f(\theta) - \alpha df(\theta) \cdot df(\theta)^* +
    \frac{\alpha^2}{2} d^2 f(\theta) \cdot (df(\theta)^*, df(\theta)^*).
\end{align*}
$$
By differentiating both sides, we obtain
$$
    (f \circ g)'(\alpha) = -df(\theta) \cdot df(\theta)^* + \alpha d^2 f(\theta) \cdot (df(\theta)^*, df(\theta)^*)
$$
and
$$
    (f \circ g)''(\alpha) = d^2 f(\theta) \cdot (df(\theta)^*, df(\theta)^*).
$$
From single-variable calculus, we know that if
$$
    d^2 f(\theta) \cdot (df(\theta)^*, df(\theta)^*) > 0,
$$
then \(f \circ g\) is strictly convex on \(\bR_{> 0}\) and
$$
    \alpha_*(\theta) =
    \frac{\mathrm{num}(\theta)}{\mathrm{den}(\theta)}
    =
    \frac{
        df(\theta) \cdot df(\theta)^*
    }{
        d^2 f(\theta) \cdot (df(\theta)^*, df(\theta)^*)
    }
$$
is its global minimizer.  We say that \(\alpha_*(\theta)\) is the
*optimal learning rate for \(f\) at \(\theta\)*.

As a special case, suppose that \(\Theta\) is a product space
$$
    \Theta = \Theta_1 \times \cdots \times \Theta_n
$$
with generic element \(\theta = (\theta_1,\dots,\theta_n)\).
Then we can write
$$
    \alpha_*(\theta) =
    \frac{
        \sum_{i=1}^{n} d_{\theta_i} f(\theta) \cdot d_{\theta_i} f(\theta)^*
    }{
        \sum_{i,j=1}^{n} d^2_{\theta_i,\theta_j} f(\theta) \cdot (d_{\theta_i} f(\theta)^*, d_{\theta_j} f(\theta)^*)
    }.
$$
Here we have introduced the following notation for partial derivatives:
* \(d_{\theta_i} f(\theta)\) is the partial derivative of \(f\) with respect to \(\theta_i\) at \(\theta\), and
* \(d^2_{\theta_i,\theta_j} f(\theta)\) is the second-order partial derivative of \(f\) with respect to \((\theta_i,\theta_j)\) at \(\theta\).

For \(k\)th-order partial derivatives, when \(i_1 = \cdots = i_k = i\) we will write
$$
    d^k_{\theta_i} f(\theta) \equiv d^k_{\theta_{i_1},\dots,\theta_{i_k}} f(\theta).
$$

## Fully-connected neural networks

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

Now consider a single-layer, fully-connected neural network
$$
\begin{align*}
    f : \Theta &\to \bR^{n_1}
\end{align*}
$$
defined by
$$
\begin{align*}
    f(\theta, x) &= \sigma (W_1 x + b_1).
\end{align*}
$$
For notational convenience, we introduce the "intermediate computation" map
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

In the rest of this section, we will assume that we are working
away from those points where the activation map is not differentiable.
This is necessary since piecewise-linear functions are not
everywhere differentiable, in general.

We will need the first- and second-order partial derivatives of \(f\) with
respect to \(W_1, b_1\).

The first-order partial derivatives of \(f\) at \((\theta, x)\) are
$$
\begin{align*}
    d_{W_1} f(\theta,x) \cdot W
    &= d \sigma(z_1(\theta,x)) \circ d_{W_1} z_1(\theta,x) \cdot W \\
    &= \Delta \sigma'(z_1(\theta,x)) W x
\end{align*}
$$
and
$$
\begin{align*}
    d_{b_1} f(\theta,x) \cdot b
    &= d \sigma(z_1(\theta,x)) \circ d_{b_1} z_1(\theta,x) \cdot b \\
    &= \Delta \sigma'(z_1(\theta,x)) b.
\end{align*}
$$
The second-order partial derivatives of \(f\) at \((\theta, x\)) are
$$
\begin{align*}
    d^2_{W_1} f(\theta,x) \cdot (V,W)
    &= \Delta [d \sigma'(z_1(\theta,x)) \circ \cdots] W x = 0_{n_1}
\end{align*}
$$
and
$$
\begin{align*}
    d^2_{b_1} f \cdot (a,b)
    &= \Delta [d \sigma'(z_1(\theta,x)) \circ \cdots] b = 0_{n_1}
\end{align*}
$$
and finally
$$
\begin{align*}
    d^2_{W_1,b_1} f \cdot (W,b)
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
    &= \frac{1}{2M} \sum_{i=1}^{M} \langle f(\theta, x_i) - y_i, f(\theta, x_i) - y_i \rangle,
\end{align*}
$$
where \(\langle\cdot,\cdot\rangle\) is the Euclidean inner product.  To
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
        d_{W_1} \sL(\theta) \cdot d_{W_1} \sL(\theta)^* +
        d_{b_1} \sL(\theta) \cdot d_{b_1} \sL(\theta)^* \\
    \mathrm{den}(\theta) &=
        d^2_{W_1} \sL(\theta) \cdot (d_{W_1} \sL(\theta)^*, d_{W_1} \sL(\theta)^*) \\
        &\qquad + \, 2 d^2_{W_1,b_1} \sL(\theta) \cdot (d_{W_1} \sL(\theta)^*, d_{b_1} \sL(\theta)^*) \\
        &\qquad + \, d^2_{b_1} \sL(\theta) \cdot (d_{b_1} \sL(\theta)^*, d_{b_1} \sL(\theta)^*).
\end{align*}
$$
Recall that for \(\alpha_*(\theta)\) to be "well-defined,"
we must have \(\mathrm{den}(\theta) > 0\).

To compute \(\alpha_*\), we start with the adjoints.  For convenience, we will use the shorthand
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    e_i &= f(\theta, x_i) - y_i \\
    \Delta'(z_{1,i}) &= \Delta \sigma'(z_1(\theta, x_i)).
\end{align*}
$
}
$$
Observe that, using the cyclic property of trace, we have
$$
\begin{align*}
    \langle v, d_{W_1} f(\theta, x_i) \cdot W \rangle
    &= \tr(v^t \Delta' (z_{1,i}) Wx_i) \\
    &= \tr(x_i v^t \Delta' (z_{1,i}) W) \\
    &= \langle \Delta' (z_{1,i}) v x_i^t, W \rangle_F,
\end{align*}
$$
where \(\langle A, B \rangle_F = \tr(A^t B)\) is the Frobenius inner product.  This shows that
$$
\begin{align*}
    d_{W_1} f(\theta, x_i)^* \cdot v = \Delta' (z_{1,i}) v x_i^t.
\end{align*}
$$
Using this observation, we obtain
$$
\begin{align*}
    d_{W_1} \sL(\theta) \cdot W
    &= \frac{1}{M} \sum_{i=1}^{M} \langle d_{W_1} f(\theta, x_i) \cdot W, e_i \rangle \\
    &= \left\langle
        W, \frac{1}{M} \sum_{i=1}^{M} d_{W_1} f(\theta, x_i)^* \cdot e_i
    \right\rangle_F \\
    &= \left\langle
        W, \frac{1}{M} \sum_{i=1}^{M} \Delta'(z_{1,i}) e_i x_i^t
    \right\rangle_F.
\end{align*}
$$
We can therefore [make the identification](https://en.wikipedia.org/wiki/Riesz_representation_theorem)
$$
\begin{align*}
    d_{W_1} \sL(\theta)^* \equiv \frac{1}{M} \sum_{i=1}^{M} \Delta'(z_{1,i}) e_i x_i^t
\end{align*}
$$
and the first term in \(\mathrm{num}(\theta)\) is
$$
\begin{align*}
    d_{W_1} \sL(\theta) \cdot d_{W_1} \sL(\theta)^* &=
    \frac{1}{M^2} \sum_{i,j=1}^{M} \langle \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j x_j^t x_i, e_i \rangle.
\end{align*}
$$
In a similar way, we can make the identification
$$
\begin{align*}
    d_{b_1} \sL(\theta)^* = \frac{1}{M} \sum_{i=1}^{M} \Delta'(z_{1,i}) e_i
\end{align*}
$$
and the second term in \(\mathrm{num}(\theta)\) is
$$
\begin{align*}
    d_{b_1} \sL(\theta) \cdot d_{b_1} \sL(\theta)^* &=
    \frac{1}{M^2} \sum_{i,j=1}^{M} \langle \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j, e_i \rangle.
\end{align*}
$$
Combining terms, we have
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    \mathrm{num}(\theta) = \frac{1}{M^2} \sum_{i,j=1}^{M} (1 + x_j^t x_i)
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
    \langle d^2_{W_1} f_i(\theta, x_i) \cdot (V, W), e_i \rangle \\
    &\qquad+ \, \frac{1}{M} \sum_{i=1}^{M} \langle d_{W_1} f_i(\theta, x_i) \cdot V, d_{W_1} f_i(\theta, x_i) \cdot W \rangle \\
    &= \frac{1}{M} \sum_{i=1}^{M} \langle \Delta'(z_{1,i}) V x_i, \Delta'(z_{1,i}) W x_i \rangle.
\end{align*}
$$
Similar computations yield
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
Note that the above results imply \(d^3 \sL(\theta) \equiv 0\) and consequently that
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

Plugging in the adjoints, we obtain
$$
\begin{align*}
    &{} d^2_{W_1} \sL(\theta) \cdot (d_{W_1} \sL(\theta)^*, d_{W_1} \sL(\theta)^*) \\
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
    &{} d^2_{b_1} \sL(\theta) \cdot (d_{b_1} \sL(\theta)^*, d_{b_1} \sL(\theta)^*) \\
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
    &{} d^2_{W_1,b_1} \sL(\theta) \cdot (d_{W_1} \sL(\theta)^*, d_{b_1} \sL(\theta)^*) \\
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
    (1 + x_j^t x_i)
    (1 + x_k^t x_i)
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
    M \sum_{i,j=1}^{M} (1 + x_j^t x_i)
        \langle \Delta'(z_{1,i}) \Delta'(z_{1,j}) e_j, e_i \rangle
}{
    \sum_{i,j,k=1}^{M}
        (1 + x_j^t x_i)
        (1 + x_k^t x_i)
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
+++
title = 'A novel Newton-like method for learning rate selection'
date = 2025-04-16T13:59:17-07:00
draft = true
tag = ['hessian', 'learning-rate', 'learning-rate-selection', 'gradient-descent', 'sgd', 'random-notes']
+++

In this post, we propose a novel Newton-like method for learning
rate selection.  The method provides a learning rate selection
process that works as a wrapper for any optimizer
(such as SGD, Adam, AdamW, and so on).  The method is
perfectly general, with no constraints imposed on the optimizer.

The PyTorch implementation is available [here](https://github.com/dscamiss/newt/).

<!--more-->

$$
    \newcommand{\bN}{\mathbb{N}}
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\bZ}{\mathbb{Z}}
    \newcommand{\rmvec}{\mathrm{vec}}
    \newcommand{\sL}{\mathscr{L}}
    \newcommand{\tr}{\mathrm{trace}}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
$$

## In general

Given a smooth function \(f : \bR^n \to \bR\),
the gradient descent iterations are
$$
    \theta_{t+1} \leftarrow \theta_t - \alpha_t \omega(\theta_t, \nabla f(\theta_t)).
$$
Here \(\alpha_t \in \bR_{\geq 0}\) is the *learning rate* at step \(t\),
and \(\omega : \bR^n \times \bR^n \to \bR^n\) is the *optimizer*.

Define the "value per learning rate" functions \(g_t : \bR_{\geq 0} \to \bR\) by
$$
    g_t(\alpha) = f(\theta_t - \alpha_t \omega(\theta_t, \nabla f(\theta_t))).
$$
Following [1], [2], we propose to minimize \(f\) by the
*Newton-like* iterations
$$
\left\{
\begin{align*}
    \theta_{t+1} &\leftarrow \theta_t - \alpha_t \omega(\theta_t, \nabla f(\theta_t)) \\
    \alpha_{t+1} &\leftarrow \alpha_t - \frac{g'_t(\alpha_t)}{g''_t(\alpha_t)}.
\end{align*}
\right.
$$
In the rest of this post, we will simply write
$$
    \omega_t = \omega(\theta_t, \nabla f(\theta_t)).
$$
To write the Newton-like iterations more explicitly, observe that
$$
\begin{align*}
    g'_t(\alpha_t)
    &= - df(\theta_t - \alpha_t \omega_t) \cdot \omega_t \\
    &= - \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle. \\
\end{align*}
$$
Similarly, we have
$$
\begin{align*}
    g''_t(\alpha_t)
    &= d^2 f(\theta_t - \alpha_t \omega_t) \cdot (\omega_t, \omega_t) \\
    &= \omega_t^T H(f)(\theta_t - \alpha_t \omega_t) \omega_t,
\end{align*}
$$
where \(H(f)(\theta)\) is the Hessian matrix of \(f\) evaluated at \(\theta\).
The Newton-like iterations are
$$
\colorbox{lesserbox}
{
$
\left\{
\begin{align*}
    \theta_{t+1} &\leftarrow \theta_t - \alpha_t \omega_t \\
    \alpha_{t+1} &\leftarrow \alpha_t +
    \frac{
        \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }{
        \omega_t^T H(f)(\theta_t - \alpha_t \omega_t) \omega_t
    }.
\end{align*}
\right.
$
}
$$

We would like to avoid the impracticality of
computing the Hessian-vector products.  To do so, we will use an approximation.

## Retsinas et al. approximation

The authors of [1] derive their approximation of the Hessian-vector
products starting with a first-order Taylor series approximation of the
partial derivatives of \(f\):
$$
\begin{align*}
    d_i f (\theta_t)
    &=
    d_i f (\theta_t - \alpha_t \omega_t + \alpha_t \omega_t) \\
    &\approx
    d_i f (\theta_t - \alpha_t \omega_t)
    +
    \alpha_t d(d_i f)(\theta_t - \alpha_t \omega_t) \cdot \omega_t.
\end{align*}
$$
We can write these approximations more explicitly as
$$
\begin{align*}
    d_i f (\theta_t)
    &\approx
    d_i f (\theta_t - \alpha_t \omega_t)
    +
    \alpha_t
    \begin{bmatrix}
        d^2_{i,1} f(\theta_t - \alpha_t \omega_t)
        & \cdots &
        d^2_{i,n} f(\theta_t - \alpha_t \omega_t)
    \end{bmatrix}
    \omega_t.
\end{align*}
$$
Stacking these quantities, we have
$$
\begin{align*}
    \nabla f (\theta_t)
    \approx
    \nabla f (\theta_t - \alpha_t \omega_t)
    + \alpha_t H(f)(\theta_t - \alpha_t \omega_t) \omega_t,
\end{align*}
$$
where \(H(f)(\theta)\) is the Hessian matrix of \(f\) evaluated at \(\theta\).

Rearranging and multiplying on the left by \(\omega_t^T\), we obtain
$$
\begin{align*}
    \omega_t^T H(f)(\theta_t - \alpha_t  \omega_t) \omega_t
    \approx
    \frac{1}{\alpha_t}
    \langle \nabla f (\theta_t) - \nabla f (\theta_t - \alpha_t \omega_t), \omega_t \rangle,
    \quad
    \alpha_t > 0.
\end{align*}
$$
This reproduces Equation (7) in [1].

Using this approximation, the \(\alpha\) iteration becomes
$$
\colorbox{lesserbox}
{
$
    \alpha_{t+1} \leftarrow \alpha_t +
    \frac{
        \alpha_t \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }
    {
        \langle \nabla f (\theta_t) - \nabla f (\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }.
$
}
$$
Adding a slow-adaptation parameter \(\gamma \in (0, 1]\) gives Equation (8) in [1]:
$$
    \alpha_{t+1} \leftarrow \alpha_t
    \left(
    1 + \gamma
    \frac{
        \alpha_t \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }
    {
        \langle \nabla f (\theta_t) - \nabla f (\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }
    \right).
$$

We are interested in how this compares to \(\alpha\) iterations
which are derived using different approximations of the Hessian-vector
product.

## Alternative approximation

Our starting point is a totally standard second-order Taylor series
approximation of \(f\):
$$
\begin{align*}
    f(\theta_t)
    &= f(\theta_t - \alpha_t \omega_t + \alpha_t \omega_t) \\
    &\approx
    f(\theta_t - \alpha_t \omega_t)
    + \alpha_t df(\theta_t - \alpha_t \omega_t) \cdot \omega_t
    + \frac{\alpha_t^2}{2} d^2 f(\theta_t - \alpha_t \omega_t) \cdot
    (\omega_t, \omega_t) \\
    &=
    f(\theta_t - \alpha_t \omega_t)
    + \alpha_t \langle \nabla f (\theta_t - \alpha_t \omega_t), \omega_t \rangle
    + \frac{\alpha_t^2}{2} \omega_t^T H(f)(\theta_t - \alpha_t \omega_t) \omega_t.
\end{align*}
$$
Rearranging, we obtain
$$
\begin{align*}
    &\omega_t^T H(f)(\theta_t - \alpha_t \omega_t) \omega_t \\
    &\qquad \approx \frac{2}{\alpha_t^2} (f(\theta_t) - f(\theta_t - \alpha_t \omega_t))
    - \frac{2}{\alpha_t}
    \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle,
    \quad \alpha_t > 0.
\end{align*}
$$
Using this approximation, the \(\alpha\) iteration becomes
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    \alpha_{t+1} &\leftarrow
    \alpha_t +
    \frac{
        \alpha_t^2 \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }{
    2 (f(\theta_t) - f(\theta_t - \alpha_t \omega_t))
        - 2 \alpha_t \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }.
\end{align*}
$
}
$$
How do the two approximations compare?  Setting them equal, we see that
$$
\begin{align*}
    &\frac{1}{\alpha_t} \langle \nabla f(\theta_t), \omega_t \rangle
    - \frac{1}{\alpha_t} \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t
    \rangle \\
    &\qquad =
    \frac{2}{\alpha_t^2} (f(\theta_t) - f(\theta_t - \alpha_t \omega_t))
    - \frac{2}{\alpha_t}
    \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
\end{align*}
$$
if and only if
$$
\begin{align*}
    \frac{\langle \nabla f(\theta_t), \omega_t \rangle
    + \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t
    \rangle}{2}
    =
    \frac{f(\theta_t) - f(\theta_t - \alpha_t \omega_t)}{\alpha_t}.
\end{align*}
$$
From this it seems the developments in [1] involve an
implicit first-order Taylor series approximation to the *lookahead value difference*
\(f(\theta_t) - f(\theta_t - \alpha_t \omega_t)\).
To understand the size of the error term incurred by using this
approximation, observe that
$$
    \frac{f(\theta_t) - f(\theta_t - \alpha_t \omega_t)}{\alpha_t} =
    \langle \nabla f(\theta_t), \omega_t \rangle
    - \frac{\alpha_t}{2} \omega_t^T H(f)(\theta_t) \omega_t
    + O(\alpha_t^3).
$$
On the other hand,
$$
    \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    =
    \langle \nabla f(\theta_t), \omega_t \rangle -
    \alpha_t \omega_t^T H(f)(\theta_t) \omega_t + O(\alpha_t^2)
$$
which implies that
$$
\begin{align*}
    \frac{\langle \nabla f(\theta_t), \omega_t \rangle
    + \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t
    \rangle}{2}
    =
    \langle \nabla f(\theta_t), \omega_t \rangle -
    \frac{\alpha_t}{2} \omega_t^T H(f)(\theta_t) \omega_t + O(\alpha_t^2).
\end{align*}
$$
We conclude that
$$
\colorbox{lesserbox}
{
$
\begin{align*}
    \frac{\langle \nabla f(\theta_t), \omega_t \rangle
    + \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t
    \rangle}{2}
    =
    \frac{f(\theta_t) - f(\theta_t - \alpha_t \omega_t)}{\alpha_t}
    + O(\alpha_t^2).
\end{align*}
$
}
$$
This means that the developments in [1] incur an extra \(O(\alpha_t^2)\) error term by approximating the lookahead value difference.

There seems to be no advantage to this,
since in any case we *must* compute \(f(\theta_t)\) and
\(f(\theta_t - \alpha_t \omega_t)\), the latter as the
"forward" part of the
 \(\nabla f(\theta - \alpha_t \omega_t)\) computation.

*Thus we expect the Newton-like method based on the approximation
in this section improves the results of [1], in terms of convergence
behavior during training.*

Adding a slow-adaptation parameter \(\gamma \in (0, 1]\) gives the
final \(\alpha\) iteration:
$$
\colorbox{magicmint}
{
$
\begin{align*}
    \alpha_{t+1} &\leftarrow
    \alpha_t \left( 1 +
    \gamma \frac{
        \alpha_t^2 \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }{
    2 (f(\theta_t) - f(\theta_t - \alpha_t \omega_t))
        - 2 \alpha_t \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    } \right).
\end{align*}
$
}
$$

## ML context

In the context of machine learning, there is (at least
in principle) a single loss function, but typically a proxy loss function
is used in each gradient descent iteration. Furthermore,
the proxy loss function is allowed to vary with each iteration (i.e., since the training batch data changes with each iteration).
To account for this, we rewrite the Newton-like iteration from the previous
section as
$$
\colorbox{magicmint}
{
$
\left\{
\begin{align*}
    \theta_{t+1} &\leftarrow \theta_t - \alpha_t \omega_t \\
    \alpha_{t+1} &\leftarrow
    \alpha_t \left( 1 +
    \gamma \frac{
        \alpha_t^2 \langle \nabla L_t(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    }{
    2 (L_t(\theta_t) - L_t(\theta_t - \alpha_t \omega_t))
        - 2 \alpha_t \langle \nabla L_t(\theta_t - \alpha_t \omega_t), \omega_t \rangle
    } \right),
\end{align*}
\right.
$
}
$$
where \(L_t\) is the proxy loss function at iteration \(t\) and
\(\omega_t = \omega(\theta_t, \nabla L_t(\theta_t))\).

## Implementation

* The wrapped optimizer needs to have "parameter update" tracking; this is not
something that is currently available in PyTorch.
* We clamp the learning rate between \(\alpha_\min\) and \(\alpha_\max\) to
prevent it from vanishing or growing without bound during training.
The upper bound on \(\alpha\) is put in place because the approximations we use are inherently local, and may be misleading about the shape of the loss function away
from the current parameter values.  In other words, we want to avoid making a
big step across the loss landscape based on local information.
 * To avoid division by zero, we add a small
factor \(\epsilon\) in the denominator of the \(\alpha\) iteration.

## References

1. G. Retsinas, G. Sfikas, P. Filntisis and P. Maragos, "Newton-Based Trainable Learning Rate," ICASSP 2023
2. G. Retsinas, G. Sfikas, P. Filntisis and P. Maragos, "Trainable Learning Rate",
2022
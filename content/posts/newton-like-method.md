+++
title = 'A Newton-like method for learning rate selection'
date = 2025-04-16T13:59:17-07:00
draft = true
tag = ['hessian', 'learning-rate', 'learning-rate-selection', 'gradient-descent', 'sgd', 'random-notes']
+++

In this post, we propose an alternative Newton-like method for learning
rate selection.

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

Consider a smooth function \(f : \Theta \to \bR\), where \(\Theta\) is is a product of Euclidean spaces with the Frobenius inner product. Gradient descent attempts to minimize \(f\) by iterating
$$
    \theta_{t+1} \leftarrow \theta_t - \alpha_t \omega(\theta_t, \nabla f(\theta_t)),
$$
where \(\alpha_t \in \bR_{\geq 0}\) is the *learning rate* at step \(t\),
and \(\omega : \Theta \times \Theta \to \Theta\) is the *optimizer*.

To simplify the notation, we will write
$$
    \omega_t = \omega(\theta_t, \nabla f(\theta_t)).
$$
Define the functions \(g_t : \bR_{\geq 0} \to \bR\) by
$$
    g_t(\alpha) = f(\theta_t - \alpha \omega_t).
$$
We propose to minimize \(f\) by standard gradient descent
iterations on \(\theta\) and Newton iterations on \(\alpha\).
That is, we propose to minimize \(f\) with the
*Newton-like iteration*
$$
\left\{
\begin{align*}
    \theta_{t+1} &\leftarrow \theta_t - \alpha_t \omega_t \\
    \alpha_{t+1} &\leftarrow \alpha_t - \frac{g'_t(\alpha_t)}{g''_t(\alpha_t)}.
\end{align*}
\right.
$$
The numerator in the \(\alpha\) iteration is
$$
\begin{align*}
    g'_t(\alpha_t) &=
    -\alpha_t df(\theta_t - \alpha_t \omega_t) \cdot \omega_t \\
    &= -\alpha_t \langle \nabla f(\theta_t - \alpha_t \omega_t), \omega_t \rangle. \\
\end{align*}
$$
The denominator in the \(\alpha\) iteration is
$$
\begin{align*}
    g''_t(\alpha_t) &= d^2 f(\theta_t - \alpha_t \omega_t) \cdot (\omega_t, \omega_t) \\
    &= \rmvec(\omega_t)^T H(f)(\theta_t - \alpha_t \omega_t)
    \rmvec(\omega_t),
\end{align*}
$$
where \(\rmvec\) is the vectorization map and \(H(f)(\theta)\)
is the Hessian of \(f\) at \(\theta\).  To avoid computing
the expensive Hessian-vector products, we will use a second-order
Taylor series approximation.  We assume that
$$
\begin{align*}
    f(\theta + \delta) \approx
    f(\theta) + df(\theta) \cdot \delta +
    \frac{1}{2} d^2 f(\theta) \cdot (\delta, \delta)
\end{align*}
$$
for each \(\theta\) and \(\delta\) sufficiently small.
For \(\theta = \theta_t\) and \(\delta = -2\alpha_t \omega_t\), the
approximation is
$$
\begin{align*}
    f(\theta_t - 2\alpha_t \omega_t)
    \approx
    f(\theta_t - \alpha_t \omega_t)
    - \alpha_t df(\theta_t - \alpha_t \omega_t) \cdot \omega_t
    + \frac{\alpha_t^2}{2} d^2 f(\theta_t - \alpha_t \omega_t) \cdot
    (\omega_t, \omega_t).
\end{align*}
$$
Assuming \(\alpha_t > 0\) and rearranging terms, we obtain
$$
\begin{align*}
    d^2 f(\theta_t - \alpha_t \omega_t) \cdot (\omega_t, \omega_t)
    &\approx \frac{2}{\alpha_t^2} (f(\theta_t - 2 \alpha_t \omega_t) - f(\theta_t - \alpha_t \omega_t))
    + \frac{2}{\alpha_t} df(\theta_t - \alpha_t \omega_t) \cdot \omega_t.
\end{align*}
$$
To further simplify the notation, we will write
$$
\begin{align*}
    \theta_t[k] = \theta_t + k \omega_t, \quad k \in \bZ,
\end{align*}
$$
so that
$$
\begin{align*}
    g_t'(\alpha_t) &= -\alpha_t \langle \nabla f(\theta_t[-1]), \omega_t \rangle \\
    g_t''(\alpha_t)
    &=
    \frac{2}{\alpha_t^2} (f(\theta_t[-2]) - f(\theta_t[-1]))
    + \frac{2}{\alpha_t} \langle \nabla f(\theta_t[-1]), \omega_t \rangle.
\end{align*}
$$
The Newton-like iteration becomes
$$
\left\{
\begin{align*}
    \theta_{t+1} &\leftarrow \theta_t - \alpha_t \omega_t \\
    \alpha_{t+1} &\leftarrow
    \alpha_t +
    \frac{
        \alpha_t^3 \langle \nabla f(\theta_t[-1]), \omega_t \rangle
    }{
    2 (f(\theta_t[-2]) - f(\theta_t[-1]))
        + 2 \alpha_t \langle \nabla f(\theta_t[-1]), \omega_t \rangle
    }.
\end{align*}
\right.
$$

## ML context

In the context of machine learning, there is a single
loss function, but typically a proxy loss function
is used in each gradient descent iteration. Furthermore,
the proxy loss function is allowed to vary with each gradient
descent iteration (i.e., since the training batch data changes).
To account for this, we rewrite the Newton-like iteration as
$$
\left\{
\begin{align*}
    \theta_{t+1} &\leftarrow \theta_t - \alpha_t \omega_t \\
    \alpha_{t+1} &\leftarrow
    \alpha_t +
    \frac{
        \alpha_t^3 \langle \nabla L_t(\theta_t[-1]), \omega_t \rangle
    }{
    2 (L_t(\theta_t[-2]) - L_t(\theta_t[-1]))
        + 2 \alpha_t \langle \nabla L_t(\theta_t[-1]), \omega_t \rangle
    },
\end{align*}
\right.
$$
where \(L_t\) is the proxy loss function at iteration \(t\) and
\(\omega_t = \omega(\theta_t, \nabla L_t(\theta_t))\).

Each Newton-like iteration requires two additional forward passes to
compute \(L_t(\theta_t[-2]\) and \(L_t(\theta_t[-1]))\), and
one additional backward pass to compute \(\nabla L_t(\theta_t[-1])\).

## No-lookahead version

One (very pragmatic) way of working around this is to

* Make the \(\alpha\) iteration a slowly-updating moving average
* Use cached loss and gradient data from previous iterations

$$
\begin{align*}
    \alpha_{t+1} &\leftarrow
    \alpha_t +
    \beta \frac{
        \alpha_t^3 \langle \nabla L_{t-1}(\theta_{t-1}[-1]), \omega_{t-1} \rangle
    }{
    2 (L_{t-1}(\theta_t[-2]) - L_{t-1}(\theta_t[-1]))
        + 2 \alpha_t \langle \nabla L_t(\theta_t[-1]), \omega_t \rangle
    },
\end{align*}
$$

Change second-order approximation as in "Notes" section.
Then describe the "no-lookahead" version of the method.

# Notes

$$
\begin{align*}
    f(\theta_t)
    &=
    f(\theta_t - \alpha_t \omega_t + \alpha_t \omega_t) \\
    &\approx
    f(\theta_t - \alpha_t \omega_t)
    + \alpha_t df(\theta_t - \alpha_t \omega_t) \cdot \omega_t
    + \frac{\alpha_t^2}{2} d^2 f(\theta_t - \alpha_t \omega_t) \cdot
    (\omega_t, \omega_t).
\end{align*}
$$
Implies
$$
\begin{align*}
    d^2 f(\theta_t - \alpha_t \omega_t) \cdot
    (\omega_t, \omega_t)
    &\approx
    \frac{2}{\alpha_t^2} (f(\theta_t)
    - f(\theta_t - \alpha_t \omega_t))
    - \frac{2}{\alpha_t} df(\theta_t - \alpha_t \omega_t) \cdot \omega_t
\end{align*}
$$
Alpha update is
$$
\begin{align*}
    \alpha_{t+1} &\leftarrow
    \alpha_t +
    \frac{
        \alpha_t^3 \langle \nabla f(\theta_t[-1]), \omega_t \rangle
    }{
        2 (f(\theta_t)
        - f(\theta_t[-1]))
        - 2 \alpha_t \langle \nabla f(\theta_t[-1]), \omega_t \rangle
    }.
\end{align*}
$$
ML context it's
$$
\begin{align*}
    \alpha_{t+1} &\leftarrow
    \alpha_t +
    \frac{
        \alpha_t^3 \langle \nabla L_t(\theta_t[-1]), \omega_t \rangle
    }{
        2 (L_t(\theta_t)
        - L_t(\theta_t[-1]))
        - 2 \alpha_t \langle \nabla L_t(\theta_t[-1]), \omega_t \rangle
    }.
\end{align*}
$$
Approximate
$$
\begin{align*}
    RHS &=
    \alpha_t +
    \beta
    \frac{
        \alpha_t^3 \langle \nabla L_t(\theta_{t+1}), \omega_t \rangle
    }{
        2 (L_t(\theta_t)
        - L_t(\theta_{t+1}))
        - 2 \alpha_t \langle \nabla L_t(\theta_{t+1}), \omega_t \rangle
    } \\
    &\approx
    \alpha_t +
    \beta
    \frac{
        \alpha_t^3 \langle \nabla L_{t+1}(\theta_{t+1}), \omega_t \rangle
    }{
        2 (L_t(\theta_t)
        - L_{t+1}(\theta_{t+1}))
        - 2 \alpha_t \langle \nabla L_{t+1}(\theta_{t+1}), \omega_t \rangle
    } \\
    &\approx
    \alpha_t +
    \beta
    \frac{
        \alpha_t^3 \langle \nabla L_t(\theta_t), \omega_t \rangle
    }{
        2 (L_{t-1}(\theta_{t-1})
        - L_t(\theta_t))
        - 2 \alpha_t \langle \nabla L_t(\theta_t), \omega_t \rangle
    }
\end{align*}
$$

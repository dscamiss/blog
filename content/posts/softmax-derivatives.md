+++
title = 'Higher-order total derivatives of softmax'
date = 2024-08-08T10:25:20-07:00
draft = false
+++

In this post, we will compute the second-order and third-order total derivatives of
the softmax map.  At some point, I thought that the higher-order derivatives
could be used to cheaply compute approximations of the softmax map, but that's
a story for another post.

<!--more-->

$$
    \newcommand{\bR}{\mathbb{R}}
    \newcommand{\Diag}{\Delta}
    \newcommand{\Exp}{\mathrm{Exp}}
    \definecolor{magicmint}{rgb}{0.67, 0.94, 0.82}
    \definecolor{lesserbox}{rgb}{0.85, 0.95, 1.0}
$$

## Softmax

To get started, recall that the *softmax map* \(\sigma : \bR^n \to \bR^n\) sends \(x\) to
$$
    \sigma(x) =
    \begin{bmatrix}
    \displaystyle
    \frac{\exp(x^1)}{\sum_{i=1}^{n} \exp(x^i)} \\
    \vdots \\
    \displaystyle
    \frac{\exp(x^n)}{\sum_{i=1}^{n} \exp(x^i)}
    \end{bmatrix},
$$
where \(x^i\) is the \(i\)th component of \(x\).
Clearly, \(\sigma(x)\) is smooth and satisfies
$$
    \langle \sigma(x), 1_n \rangle = 1,
$$
where \(\langle \cdot, \cdot \rangle\) is the Euclidean inner product
on \(\bR^n\) and \(1_n\) is the all-ones vector.

## Total derivatives

### First-order total derivative

We begin by rewriting \(\sigma\) in a more convenient form as
$$
    \sigma(x) = \frac{\Exp(x)}{\langle \Exp(x), 1_n \rangle},
$$
where \(\Exp : \bR^n \to \bR^n\) is the component-wise application of \(\exp\).
That is,
$$
    \Exp(x) =
    \begin{bmatrix}
    \exp(x^1) \\
    \vdots \\
    \exp(x^n)
    \end{bmatrix}.
$$
The total derivative of \(\Exp\) at \(x\) is
$$
\begin{align}
    d\Exp(x) \cdot h = \Exp(x) \odot h,
\end{align}
$$
where \(\odot\) is the component-wise product.  To see this, observe that
$$
    \Exp(x) =
    \begin{bmatrix}
        \exp \circ\, \pi^1(x) \\
        \vdots \\
        \exp \circ\, \pi^n(x)
    \end{bmatrix},
$$
where \(\pi^i : \bR^n \to \bR\) is projection onto the \(i\)th factor.  By the chain rule,
we have
$$
\begin{align*}
    d\Exp(x) \cdot h
    =
    \begin{bmatrix}
        \exp(x^1) \pi^1(h) \\
        \vdots \\
        \exp(x^n) \pi^n(h)
    \end{bmatrix}
    =
    \Exp(x) \odot h.
\end{align*}
$$
Using this fact, together with the quotient rule, the total derivative of \(\sigma\)
at \(x\) is
$$
\begin{align*}
    d\sigma(x) \cdot h
    &= \frac{\langle \Exp(x), 1_n \rangle d\Exp(x) \cdot h
    - \langle d \Exp(x) \cdot h, 1_n \rangle \Exp(x)
    }{\langle \Exp(x), 1_n \rangle^2} \\
    &= \frac{\langle \Exp(x), 1_n \rangle \Exp(x) \odot h
    - \langle \Exp(x) \odot h, 1_n \rangle \Exp(x)
    }{\langle \Exp(x), 1_n \rangle^2} \\
    &= \frac{\Exp(x) \odot h}{\langle \Exp(x), 1_n \rangle} -
    \frac{\langle \Exp(x) \odot h, 1_n \rangle \Exp(x)}{\langle \Exp(x), 1_n \rangle^2} \\
    &= \frac{\Exp(x)}{\langle \Exp(x), 1_n \rangle} \odot h
    - \left\langle \frac{\Exp(x)}{\langle \Exp(x), 1_n \rangle}, h \right\rangle
    \frac{\Exp(x)}{\langle \Exp(x), 1_n \rangle} \\
    &= \sigma(x) \odot h - \langle \sigma(x), h \rangle \sigma(x).
\end{align*}
$$
Reiterated on a single line:
$$
\colorbox{magicmint}{
    $d\sigma(x) \cdot h = \sigma(x) \odot h - \langle \sigma(x), h \rangle \sigma(x)$.
}
$$
Finally, note that we can write
$$
    d \sigma(x) \cdot h = \sigma(x) \odot h - \sigma(x) \sigma(x)^t h,
$$
which implies that the Jacobian matrix of \(\sigma\) at \(x\) is
$$
\colorbox{lesserbox}{
    $J\sigma(x) = \Diag \sigma(x) - \sigma(x) \sigma(x)^t$.
}
$$
Here, \(\Delta : \bR^n \to \bR^{n \times n}\) sends \(x\) to the diagonal
matrix whose \((i,i)\)th entry is \(x^i\).

### Second-order total derivative

Moving to the second-order total derivative, recall that
$$
    d^2 \sigma(x) \cdot (h_1, h_2) =
    d[x \mapsto d\sigma(x) \cdot h_1](x) \cdot h_2.
$$
Using the expression for \(d\sigma(x)\) from the previous section, we obtain
$$
\begin{align*}
    d^2 \sigma(x) \cdot (h_1, h_2)
    &= (d\sigma(x) \cdot h_2) \odot h_1 \\
    &\qquad - \, \langle d\sigma(x) \cdot h_2, h_1 \rangle \sigma(x) \\
    &\qquad - \, \langle \sigma(x), h_1 \rangle d\sigma(x) \cdot h_2.
\end{align*}
$$
The first term on the right-hand side is
$$
\begin{align*}
    (d\sigma(x) \cdot h_2) \odot h_1
    &= (\sigma(x) \odot h_2 - \langle \sigma(x), h_2 \rangle \sigma(x)) \odot h_1 \\
    &= \sigma(x) \odot h_1 \odot h_2 - \langle \sigma(x), h_2 \rangle \sigma(x) \odot h_1.
\end{align*}
$$
The second term is
$$
\begin{align*}
    - \langle d\sigma(x) \cdot h_2, h_1 \rangle \sigma(x)
    &= - \langle \sigma(x) \odot h_2 - \langle \sigma(x), h_2 \rangle \sigma(x), h_1 \rangle \sigma(x) \\
    &= - \langle \sigma(x) \odot h_2, h_1 \rangle \sigma(x) +
    \langle \sigma(x), h_2 \rangle \langle \sigma(x), h_1 \rangle \sigma(x) \\
    &= - \langle \sigma(x), h_1 \odot h_2 \rangle \sigma(x) +
    \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle \sigma(x).
\end{align*}
$$
Finally, the third term is
$$
\begin{align*}
    - \langle \sigma(x), h_1 \rangle d\sigma(x) \cdot h_2
    &= - \langle \sigma(x), h_1 \rangle (\sigma(x) \odot h_2 - \langle \sigma(x), h_2 \rangle \sigma(x)) \\
    &= - \langle \sigma(x), h_1 \rangle \sigma(x) \odot h_2 +
        \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle \sigma(x).
\end{align*}
$$
In total, we have
$$
    \colorbox{magicmint}{
        $
        \begin{align*}
            d^2 \sigma(x) \cdot (h_1, h_2)
            &= \sigma(x) \odot h_1 \odot h_2 \\
            &\qquad - \, \langle \sigma(x), h_2 \rangle \sigma(x) \odot h_1 \\
            &\qquad - \, \langle \sigma(x), h_1 \rangle \sigma(x) \odot h_2 \\
            &\qquad - \, \langle \sigma(x), h_1 \odot h_2 \rangle \sigma(x) \\
            &\qquad + \, 2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle \sigma(x).
        \end{align*}
        $
    }
$$
Note that the right-hand side of the above expression is symmetric and bilinear.

### Third-order total derivative

Moving to the third-order total derivative, recall that
$$
    d^3 \sigma(x) \cdot (h_1, h_2, h_3) =
    d[x \mapsto d^2\sigma(x) \cdot (h_1, h_2)](x) \cdot h_3.
$$
Using the expression for \(d^2 \sigma(x)\) from the previous section, we obtain
$$
\begin{align*}
    d^3 \sigma(x) \cdot (h_1, h_2, h_3)
    &= [d\sigma(x) \cdot h_3] \odot h_1 \odot h_2 \\
    &\qquad - \, \langle d\sigma(x) \cdot h_3, h_2 \rangle \sigma(x) \odot h_1 \\
    &\qquad - \, \langle \sigma(x), h_2 \rangle [d\sigma(x) \cdot h_3] \odot h_1 \\
    &\qquad - \, \langle d\sigma(x) \cdot h_3, h_1 \rangle \sigma(x) \odot h_2 \\
    &\qquad - \, \langle \sigma(x), h_1 \rangle [d\sigma(x) \cdot h_3] \odot h_2 \\
    &\qquad - \, \langle d\sigma(x) \cdot h_3, h_1 \odot h_2 \rangle \sigma(x) \\
    &\qquad - \, \langle \sigma(x), h_1 \odot h_2 \rangle d\sigma(x) \cdot h_3 \\
    &\qquad + \, 2 \langle d\sigma(x) \cdot h_3, h_1 \rangle \langle \sigma(x), h_2 \rangle \sigma(x) \\
    &\qquad + \, 2 \langle \sigma(x), h_1 \rangle \langle d\sigma(x) \cdot h_3, h_2 \rangle \sigma(x) \\
    &\qquad + \, 2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle d\sigma(x) \cdot h_3.
\end{align*}
$$
The first term on the right-hand side is
$$
\begin{align*}
    \sigma(x) \odot h_1 \odot h_2 \odot h_3
    - \langle \sigma(x), h_3 \rangle \sigma(x) \odot h_1 \odot h_2.
\end{align*}
$$
The second term is
$$
\begin{align*}
    - \langle \sigma(x), h_2 \odot h_3 \rangle \sigma(x) \odot h_1
    + \langle \sigma(x), h_2 \rangle \langle \sigma(x), h_3 \rangle \sigma(x) \odot h_1.
\end{align*}
$$
The third term is
$$
\begin{align*}
    - \langle \sigma(x), h_2 \rangle
    \sigma(x) \odot h_1 \odot h_3
    + \langle \sigma(x), h_2 \rangle \langle \sigma(x), h_3 \rangle
    \sigma(x) \odot h_1.
\end{align*}
$$
The fourth term is
$$
\begin{align*}
    - \langle \sigma(x), h_1 \odot h_3 \rangle \sigma(x) \odot h_2
    + \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_3 \rangle \sigma(x) \odot h_2.
\end{align*}
$$
The fifth term is
$$
\begin{align*}
    - \langle \sigma(x), h_1 \rangle
    \sigma(x) \odot h_2 \odot h_3
    + \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_3 \rangle
    \sigma(x) \odot h_2.
\end{align*}
$$
The sixth term is
$$
\begin{align*}
    - \langle \sigma(x), h_1 \odot h_2 \odot h_3 \rangle \sigma(x)
    + \langle \sigma(x), h_3 \rangle \langle \sigma(x), h_1 \odot h_2 \rangle \sigma(x).
\end{align*}
$$
The seventh term is
$$
\begin{align*}
    - \langle \sigma(x), h_1 \odot h_2 \rangle
    \sigma(x) \odot h_3
    + \langle \sigma(x), h_1 \odot h_2 \rangle
    \langle \sigma(x), h_3 \rangle \sigma(x).
\end{align*}
$$
The eighth term is
$$
\begin{align*}
    2 \langle \sigma(x), h_1 \odot h_3 \rangle \langle \sigma(x), h_2 \rangle \sigma(x)
    - 2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle \langle \sigma(x), h_3 \rangle \sigma(x).
\end{align*}
$$
The ninth term is
$$
\begin{align*}
    2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \odot h_3 \rangle \sigma(x)
    - 2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle \langle \sigma(x), h_3 \rangle \sigma(x).
\end{align*}
$$
Finally, the tenth term is
$$
\begin{align*}
    2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle
    \sigma(x) \odot h_3
    - 2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle
    \langle \sigma(x), h_3 \rangle \sigma(x).
\end{align*}
$$
In total, we have
$$
    \colorbox{magicmint}{
        $
        \begin{align*}
            d^3 \sigma(x) \cdot (h_1, h_2, h_3)
            &= \sigma(x) \odot h_1 \odot h_2 \odot h_3 \\
            &\qquad + \, 2 \langle \sigma(x), h_2 \rangle \langle \sigma(x), h_3 \rangle \sigma(x) \odot h_1 \\
            &\qquad + \, 2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_3 \rangle \sigma(x) \odot h_2 \\
            &\qquad + \, 2 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle \sigma(x) \odot h_3 \\
            &\qquad - \, \langle \sigma(x), h_3 \rangle \sigma(x) \odot h_1 \odot h_2 \\
            &\qquad - \, \langle \sigma(x), h_2 \rangle \sigma(x) \odot h_1 \odot h_3 \\
            &\qquad - \, \langle \sigma(x), h_1 \rangle \sigma(x) \odot h_2 \odot h_3 \\
            &\qquad - \, \langle \sigma(x), h_2 \odot h_3 \rangle \sigma(x) \odot h_1 \\
            &\qquad - \, \langle \sigma(x), h_1 \odot h_3 \rangle \sigma(x) \odot h_2 \\
            &\qquad - \, \langle \sigma(x), h_1 \odot h_2 \rangle \sigma(x) \odot h_3 \\
            &\qquad - \, \langle \sigma(x), h_1 \odot h_2 \odot h_3 \rangle \sigma(x) \\
            &\qquad + \, 2 \langle \sigma(x), h_2 \odot h_3 \rangle \langle \sigma(x), h_1 \rangle \sigma(x) \\
            &\qquad + \, 2 \langle \sigma(x), h_1 \odot h_3 \rangle \langle \sigma(x), h_2 \rangle \sigma(x) \\
            &\qquad + \, 2 \langle \sigma(x), h_1 \odot h_2 \rangle \langle \sigma(x), h_3 \rangle \sigma(x) \\
            &\qquad - \, 6 \langle \sigma(x), h_1 \rangle \langle \sigma(x), h_2 \rangle \langle \sigma(x), h_3 \rangle \sigma(x).
        \end{align*}
        $
    }
$$
One can check that the right-hand side of the above expression is symmetric and trilinear.

## Taylor series approximations

The results above can be used to produce
useful approximations to \(\sigma\).  Recall that
for a smooth map \(f : \bR^n \to \bR^n\), the *\(p\)-th order
Taylor series approximation to \(f\) at \(x\)* is
$$
    f(x + h) \approx \sum_{k=0}^{p} \frac{1}{k!} d^k f(x) \cdot [h]^k,
$$
where \(d^0 f = f\) and \([h]^k\) is the \(k\)-tuple whose components are all equal to \(h\).

### First-order approximation

The first-order Taylor series approximation to \(\sigma\) at \(x\) is
$$
\colorbox{lesserbox}{
    $
    \begin{align*}
        \sigma(x + h) \approx \sigma(x) + \sigma(x) \cdot h - \langle \sigma(x), h \rangle \sigma(x).
    \end{align*}
    $
}
$$

### Second-order approximation

First observe that
$$
\begin{align*}
    d^2 \sigma(x) \cdot (h, h)
    &= \sigma(x) \odot h \odot h
    - 2 \langle \sigma(x), h \rangle \sigma(x) \odot h \\
    &\qquad - \, \langle \sigma(x), h \odot h \rangle \sigma(x)
    + 2 \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \sigma(x).
\end{align*}
$$
The second-order Taylor series approximation to \(\sigma\) at \(x\) is
$$
\colorbox{lesserbox}{
    $
    \begin{align*}
        \sigma(x + h) &\approx \sigma(x) + \sigma(x) \cdot h - \langle \sigma(x), h \rangle \sigma(x) \\
        &\qquad + \, \frac{1}{2} \sigma(x) \odot h \odot h
        - \langle \sigma(x), h \rangle \sigma(x) \odot h \\
        &\qquad - \, \frac{1}{2} \, \langle \sigma(x), h \odot h \rangle \sigma(x)
        + \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \sigma(x).
    \end{align*}
    $
}
$$

### Third-order approximation

First observe that
$$
    \begin{align*}
        d^3 \sigma(x) \cdot (h, h, h)
        &= \sigma(x) \odot h \odot h \odot h
        + 6 \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \sigma(x) \odot h \\
        &\qquad - \, 3 \langle \sigma(x), h \rangle \sigma(x) \odot h \odot h
        - 3 \langle \sigma(x), h \odot h \rangle \sigma(x) \odot h \\
        &\qquad - \, \langle \sigma(x), h \odot h \odot h \rangle \sigma(x)
        + 6 \langle \sigma(x), h \odot h \rangle \langle \sigma(x), h \rangle \sigma(x) \\
        &\qquad - \, 6 \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \sigma(x).
    \end{align*}
$$
The third-order Taylor series approximation to \(\sigma\) at \(x\) is
$$
\colorbox{lesserbox}{
    $
    \begin{align*}
        \sigma(x + h) &\approx \sigma(x) + \sigma(x) \cdot h - \langle \sigma(x), h \rangle \sigma(x) \\
        &\qquad + \, \frac{1}{2} \sigma(x) \odot h \odot h
        - \langle \sigma(x), h \rangle \sigma(x) \odot h \\
        &\qquad - \, \frac{1}{2} \, \langle \sigma(x), h \odot h \rangle \sigma(x)
        + \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \sigma(x) \\
        &\qquad + \, \frac{1}{6} \sigma(x) \odot h \odot h \odot h
        + \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \sigma(x) \odot h \\
        &\qquad - \, \frac{1}{2} \langle \sigma(x), h \rangle \sigma(x) \odot h \odot h
        - \frac{1}{2} \langle \sigma(x), h \odot h \rangle \sigma(x) \odot h \\
        &\qquad - \, \frac{1}{6} \langle \sigma(x), h \odot h \odot h \rangle \sigma(x)
        + \langle \sigma(x), h \odot h \rangle \langle \sigma(x), h \rangle \sigma(x) \\
        &\qquad - \, \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \langle \sigma(x), h \rangle \sigma(x).
    \end{align*}
    $
}
$$
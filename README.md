# NOUQ
Error estimate of functional type for neural operators

We consider elliptic equation in the domain $\Gamma \subseteq [0, 1]^{D}$
```math
\begin{equation}
    \begin{split}
    -\sum_{ij=1}^{D}\frac{\partial}{\partial x_i}\left(a_{ij}(x) \frac{\partial u}{\partial x_j} \right) + b^{2}(x) u(x) = f(x),\,\left.u\right|_{\partial\Gamma} = 0,\,a_{ij}(x) \geq c > 0.
    \end{split}
\end{equation}
```

First, we define random trigonometric polynomials
```math
    \mathcal{P}(N_1, N_2, \alpha) = \left\{f(x) = \mathcal{R}\left(\sum_{m=0}^{N_1}\sum_{n=0}^{N_2}\frac{c_{mn}\exp\left(2\pi i(mx_1 + nx_2)\right)}{(1+m+n)^\alpha}\right):\mathcal{R}(c), \mathcal{I}(c)\simeq \mathcal{N}(0, I)\right\}.
```
For the first equation, we use Cholesky factorization to define matrix $a$ and random trigonometric polynomials for $b$ and $f$:
```math
\begin{equation}
    \begin{split}
    &a(x) = \begin{pmatrix}
        \alpha(x) & 0 \\
        \gamma(x) & \beta(x)
    \end{pmatrix}
    \begin{pmatrix}
        \alpha(x) & \gamma(x) \\
        0 & \beta(x)
    \end{pmatrix},\\
    &\,\alpha(x),\beta(x) \simeq 0.1\mathcal{P}(5, 5, 2) + 1;\,\gamma(x),\,b(x),\,f(x) \simeq \mathcal{P}(5, 5, 2).
    \end{split}
\end{equation}
```

The second equation has a discontinuous scalar diffusion coefficient:
```math
\begin{equation}
    \begin{split}
    a(x) = \alpha(x)I,\,\alpha(x) = \begin{cases}
        10,\,p_1(x) \geq 0;\\
        1,\, ~~p_1(x) < 0,
    \end{cases}\,b(x) = 0,\,f(x) = 1,\,p_1(x) \simeq \mathcal{P}(5, 5, 2).
    \end{split}
\end{equation}
```

Third equation is similar to the second one but with more diverse $b$ and $f$:
```math
\begin{equation}
    \begin{split}
    a(x) = \alpha(x)I,\,\alpha(x) = \begin{cases}
        10,\,p_1(x) \geq 0;\\
        1,\, ~~p_1(x) < 0,
    \end{cases}\,b(x),\,f(x),\,p_1(x) \simeq \mathcal{P}(5, 5, 2).
    \end{split}
\end{equation}
```
Last equation is similar to the first one but with $b=0$:
```math
\begin{equation}
    \begin{split}
    &a(x) = \begin{pmatrix}
        \alpha(x) & 0 \\
        \gamma(x) & \beta(x)
    \end{pmatrix}
    \begin{pmatrix}
        \alpha(x) & \gamma(x) \\
        0 & \beta(x)
    \end{pmatrix},\\
    &\,\alpha(x),\beta(x) \simeq 0.1\mathcal{P}(5, 5, 2) + 1;\,\gamma(x), f(x) \simeq \mathcal{P}(5, 5, 2);\,b(x) = 0.
    \end{split}
\end{equation}
``` 
Datasets can be found at these links:

* [1 datasets for 2D Elliptic equation](https://disk.yandex.ru/d/v40iOyKxILEhUA)
* [2 datasets for 2D Elliptic equation](https://disk.yandex.ru/d/fAs6WXF1IoTcUQ)
* [3 datasets for 2D Elliptic equation](https://disk.yandex.ru/d/3pJUZcR0VnGREQ)
* [4 datasets for 2D Elliptic equation](https://disk.yandex.ru/d/m2xuTEHUCxRTgw)


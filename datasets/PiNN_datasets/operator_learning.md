# Operator learning with Astral loss

We already used Astral loss for PiNNs, how the same loss can be used for operator learning? For simplicity, I am going to illustrate eeverything with diffusion equation given below.

## Diffusion equation
Stationary diffusion equation in the conservative form
``` math
\begin{equation}
  \begin{split}
    -&\frac{\partial}{\partial x}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial x}\right) -\frac{\partial}{\partial y}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial y}\right)  = f(x, y),\\
    &u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0,\\
    &x, y\in(0, 1)\times(0, 1),
  \end{split}
\end{equation}
```
where
``` math
\begin{equation}
  \begin{split}
      &\sigma = \frac{z - \min z}{\max z - \min z}, z\sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right);\\
      &u = v\sin(\pi x)\sin(\pi y), v \sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right),
  \end{split}
\end{equation}
```
and the source term $f(x, y)$ is generated from diffusion coefficient $\sigma(x, y)$ and the exact solution $u(x, y)$.

**Error majorant:**
``` math
\begin{equation}
    \begin{split}
        &E[u -v] = \sqrt{\int dx dy\, \sigma(x, y) \text{grad}\,(u(x, y) - v(x, y))\cdot \text{grad}\,(u(x, y) - v(x, y))}, \\
        &E[u -v] \leq \sqrt{C_F^2(1+\beta) \int dx dy\,\left(f(x, y) + \text{div}\,w(x, y)\right)^2 + \frac{1 + \beta}{\beta \sigma(x, y)}\int dx dy\,\left(\sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)\cdot \left(\sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)} \\
        &C_F = 1 \big/\left(\inf_{x, y} 2\pi\sqrt{\sigma(x, y)}\right).
    \end{split}
\end{equation}
```

## Scheme 1: unsupervised operator learning

As we can see from expression above, upper bound depends on $2$ known functions and $3$ unknown functions, $1$ known scalar value and one unknown scalar value:
1. Known functions are right-hand side $f(x, y)$ and diffusion coefficient $\sigma(x, y).$
2. Known scalar $C_F$ -- Friedrichs constant.
3. Unknown functions are solution $v(x, y)$ and fluxes $w_1(x, y), w_2(x, y).$
4. Unknown scalar $\beta$ -- scalar parameter that control relative importance of two terms in error majorant.
Lets denote majorant as $L\left[f, \sigma, C_f, v, w, \beta\right]$.

For PiNNs we solve the following optimisation problem:
```math
\theta = \arg\min_{\theta} \mathbb{E}_{x}L\left[f, \sigma, C_F, v, w, \beta\right],\text{ s.t.}\left\{v(x), w(x), \beta\right\} = NN(x, \theta),
```
that is, for given data we predict unknown quantities with neural network and optimize weights $\theta$ of this neural network.

For Operator Learning there is a mild genralization -- we have context, i.e., additional input to the neural network and additional averaging:
```math
\theta = \arg\min_{\theta} \mathbb{E}_{\sigma, f}\mathbb{E}_{x}L\left[f, \sigma, C_F, v, w, \beta\right],\text{ s.t.}\left\{v(x), w(x), \beta\right\} = NN(x, f, \sigma, \theta),
```
where I omitted averaging with respect to $C_F$ since it is completely defined by $\sigma(x, y)$.

## Scheme 2: supervised operator learning (I)
The most straightforward way to apply operator learning in a supervised setting is to simply collect a dataset (e.g., using PiNNs):

1. Features: $f$, $\sigma$, $x$.
2. Targets: $v$, $w$, $\beta$.

After that one optimizes a usual $L_2$ loss:

```math
L[f, \sigma, v, w, \beta] = \left\|\widetilde{v} - v\right\| + \left\|\widetilde{w} - w\right\| + \left\|\widetilde{\beta} - \beta\right\|,\text{ s.t.}\left\{\widetilde{v}(x), \widetilde{w}(x), \widetilde{\beta}\right\} = NN(x, f, \sigma, \theta).
```

## Scheme 3: supervised operator learning (II)
There is, however, one possible complication in the scheme above. When we predict $\widetilde{v}$, it differs from $v$ available in the collected dataset. This present a problem in our case since fluxes $w$ are (approximately) optimal for $v$ not for $\widetilde{v}$. Is this effect significant? I suspect it is, but do not know for sure. Is there a way to check that? The plan below provide an answer.

1. Colelct dataset with features $f$, $\sigma$, $x$ and target $v$, i.e., this is an ordinary operator learning problem when we predict approximate solution from PDE data.
2. Train neural operator $N_1$ on this dataset and predict approximate solutions $\widetilde{v}$
3. Collect second dataset with features $f$, $\sigma$, $x$, $\widetilde{v}$ and targets $w$, $\beta$. This can be done with physics-informed neural network and Astral loss but now with approximate solution fixed to $\widetilde{v}$.
4. Train neural operator $N_2$ on this dataset.

This scheme differs from **supervised operator learning (I)** in a way one obtain $w$, $\beta$. In **supervised operator learning (I)** we obtain them for wrong approximate solution $v$ (wrong in a sense that we predict different targets $\widetilde{v}$), and in **supervised operator learning (II)** we select $w$, $\beta$ for approximate solution $\widetilde{v}$ that we actually predict. This way one may hope to obtain a more accurate upper bound.

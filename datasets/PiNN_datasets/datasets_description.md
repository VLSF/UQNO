# Datasets for PiNN training

There are four datasets: diffusion equation inside the square, diffusion equation in the L-shaped domain, convection-diffusion equation, Maxwell's equation.

Scripts used for dataset generation:
1. [Diffusion equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Diffusion.py) 
2. [Diffusion equation in the L-shaped domain](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/L_shaped.py)
3. [Convection-diffusion equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Convection_Diffusion.py)
4. [Maxwell's equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Maxwell.py)

Datasets are available for download:
1. [Diffusion equation](https://disk.yandex.ru/d/ofuDDtCXYDiDpg)
2. [Diffusion equation in the L-shaped domain](https://disk.yandex.ru/d/2fnSN1M-CanPPw)
3. [Convection-diffusion equation](https://disk.yandex.ru/d/ZMdRFig3KaezeQ)
4. [Maxwell's equation](https://disk.yandex.ru/d/VsS0MrxlSPvl4g)

One can download from GoogleColab / Jupyter Notebook using the following script

```python
import requests
from urllib.parse import urlencode

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/ofuDDtCXYDiDpg' # public link

final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

download_response = requests.get(download_url)
with open('Diffusion.npz', 'wb') as f:
    f.write(download_response.content)
```

Variable ```public_key``` above contains a ling to the Diffusion equation dataset. All datasets are ```.npz``` archives.

Mathematical details are given below.

## Diffusion equation

Here we convider stationary diffusion equation in the conservative form
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

## Diffusion equation in the L-shaped domain

We consider, again, a stationary diffusion equation
``` math
\begin{equation}
    \begin{split}
    -&\text{div}\,\text{grad}\,u(x, y) + \left(b(x, y)\right)^2 u(x, y) = f(x, y),\\
    &\left.u(x, y)\right|_{(x, y) \in \partial \Gamma} = 0,\\
    &x, y \in \Gamma = [0, 1]^2 \setminus [0.5, 1]^2,
    \end{split}
\end{equation}
```
where 
``` math
\begin{equation}
  \begin{split}
      &b, f \sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right).
  \end{split}
\end{equation}
```

## Convection-diffusion equation

For $D_x = 1$ convection-diffusion equation reads
``` math
\begin{equation}
    \begin{split}
        &\frac{\partial u(x, t)}{\partial t} - \frac{\partial^2 u(x, t)}{\partial x^2} + a\frac{\partial u(x, t)}{\partial x} = f(x), \\
        &u(x, 0) = \phi(x),\\
        &u(0, t) = u(1, t) = 0.
    \end{split}
\end{equation}
```
To solve it we consider the eigenvalue problem
``` math
\begin{equation}
    \begin{split}
        &- \frac{d^2 \psi_{\lambda}(x)}{d x^2} + a\frac{d \psi_{\lambda}(x)}{d x} = \lambda^2 \psi_{\lambda}(x), \\
        &\psi_{\lambda}(0) = \psi_{\lambda}(1) = 0,
    \end{split}
\end{equation}
```
with solution
``` math
\begin{equation}
    \psi_{k}(x) = e^{\frac{ax}{2}} \sin(\pi k x),\,\lambda_{k}^2 = \left(\pi k \right)^2 + \frac{a^2}{4}.
\end{equation}
```
Now we suppose that all PDE data is in the convenient format
``` math
\begin{equation}
    \begin{split}
        &f(x) = \sum_{k=0}^{\infty} f_k \psi_{k}(x),\\
        &\phi(x) = \sum_{k=0}^{\infty} \phi_k \psi_{k}(x),
    \end{split}
\end{equation}
```
and use standard separation of variables ansatz
``` math
\begin{equation}
    u(x, t) = \sum_{k=0}^{\infty} u_{k}(t) \psi_k(x).
\end{equation}
```

Ansatz leads to simle ODEs for each $c_{k}(t)$
``` math
\begin{equation}
    \begin{split}
      &\dot{u}_{k}(t) + \lambda_{k}^2 u_{k}(t) = f_k,\\
      &u_{k}(0) = \phi_{k}
    \end{split}
\end{equation}
```
with solution
``` math
\begin{equation}
    u_{k}(t) = \phi_k e^{-\lambda_{k}^2t} + \frac{f_k t}{1 + \lambda_{k}^2 t},
\end{equation}
```
so one can see that initial conditions decays and $u(x, t)$ approaches solution of the stationary problem.

To generate a family of PDEs we are going to draw $f_{k}$ and $\psi_k$ from normal distributions $N\left(0, \left(1 + (\pi k/ L)^s\right)^{-p}\right)$ for some positive $L, s, p$.


## Maxwell's equation

We consider $D=2$ Maxwell's equation in the square $x, y \in (0, 1)$

``` math
\begin{equation}
  \begin{split}
    &\frac{\partial}{\partial y}\left(\mu\left(\frac{\partial E_{y}}{\partial x} - \frac{\partial E_{x}}{\partial y}\right)\right) + E_{x} = f_{x},\\
    &-\frac{\partial}{\partial x}\left(\mu\left(\frac{\partial E_{y}}{\partial x} - \frac{\partial E_{x}}{\partial y}\right)\right) + E_{y} = f_{y},\\
    &\left.E_{x}\right|_{y=0} = \left.E_{x}\right|_{y=1} = 0,\\
    &\left.E_{y}\right|_{x=0} = \left.E_{y}\right|_{x=1} = 0.
  \end{split}
\end{equation}
```

For that we generate scalar field $A$ from $N\left(0, \left(I - \Delta\right)^{-k}\right)$ with homogeneous Neumann boundary conditions and use it to form exact solution $E_{x} = \partial_y A,\,E_{y} = -\partial_{x} A$. Next we sample $\mu$ from the same normal distribution and use all these fields to find solenoidal $f_{x}$ and $f_{y}$. This is done below.

## Anisotropic diffusion equation

Similar to Diffusion equation but has additional anisotropy parameter $\epsilon$:

``` math
\begin{equation}
  \begin{split}
    -&\frac{\partial}{\partial x}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial x}\right) - \epsilon^2\frac{\partial}{\partial y}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial y}\right)  = f(x, y),\\
    &u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0,\\
    &x, y\in(0, 1)\times(0, 1),
  \end{split}
\end{equation}
```
This parameter is also used to generate exact solution
``` math
\begin{equation}
  \begin{split}
      &\sigma = \frac{z - \min z}{\max z - \min z}, z\sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right);\\
      &u = v\sin(\pi x)\sin(\pi y), v \sim \mathcal{N}\left(0, \left(I - c\left(\partial_x^2 + \epsilon^2\partial_y^2\right)\right)^{-k}\right),
  \end{split}
\end{equation}
```
and the source term $f(x, y)$ is generated from diffusion coefficient $\sigma(x, y)$ and the exact solution $u(x, y)$.
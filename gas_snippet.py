#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import astropy.constants as c
import astropy.units as u

try:
    import widget
    has_widget = True
except ImportError:
    has_widget = False
    print('Widget not installed, cannot show time dependence!')
    print('You can download it from\n')
    print('    https://github.com/birnstiel/widget\n')
    print('and install it with this command from within the repository\n')
    print('    pip install .')

# define constants

mu = 2.3
m_p = c.m_p.cgs.value
au = c.au.cgs.value
year = u.year.to(u.s)
k_b = c.k_B.cgs.value
m_star = c.M_sun.cgs.value
Grav = c.G.cgs.value


def main():

    # define grids

    nr = 1000
    nt = 200

    xi = np.logspace(-1, 3, nr + 1) * au
    x = 0.5 * (xi[1:] + xi[:-1])

    time = np.logspace(-1, 6, nt - 1) * year

    # define initial condition

    time = np.hstack((0, time))
    sig_0 = 200. * (x / au)**-1
    sig_0[x > 100 * au] = 1e-100
    alpha = np.ones_like(x) * 1e-2
    T = 200 * (x / au)**-0.5

    # iteration

    sig_g = np.zeros([nt, nr])
    sig_g[0] = sig_0
    sig = sig_0.copy()

    for it in range(nt - 1):
        dt = time[it + 1] - time[it]

        sig = gas_timestep(dt, x, alpha, T, sig)
        sig_g[it + 1] = sig

    # plotting

    if has_widget:
        widget.plotter(x / au, sig_g, times=time / year,
                       xlog=True, ylog=True, ylim=[1e-4, 1e4],
                       xlabel='$r$ [au]', ylabel='$\Sigma_\mathrm{g}$ [g cm$^{-2}$]')
    else:
        f, ax = plt.subplots()
        ax.loglog(x / au, sig_g[-1])
        ax.set_ylim(1e-4, 1e4)
        ax.set_xlabel('$r$ [au]')
        ax.set_ylabel('$\Sigma_\mathrm{gas}$ [g/cm$^3$]')

    plt.show()


def gas_timestep(dt, x, alpha, T, sig_g):
    """
    Does an implicit time step for the gas surface density
    """
    nr = len(x)

    nu_gas = alpha * k_b * T / mu / m_p * np.sqrt(x**3 / Grav / m_star)
    u_gas_old = sig_g * x
    u_gas = u_gas_old[:]
    v_gas = np.zeros(nr)
    D_gas = 3.0 * np.sqrt(x)
    g_gas = nu_gas / np.sqrt(x)
    h_gas = np.ones(nr)
    K_gas = np.zeros(nr)
    L_gas = np.zeros(nr)
    p_L = -(x[1] - x[0]) * h_gas[1] / (x[1] * g_gas[1])
    q_L = 1. / x[0] - 1. / x[1] * g_gas[0] / g_gas[1] * h_gas[1] / h_gas[0]
    r_L = 0.0

    u_gas = impl_donorcell_adv_diff_delta(nr, x, D_gas, v_gas, g_gas, h_gas, K_gas, L_gas,
                                          u_gas, dt, p_L, 0.0, q_L, 1.0, r_L, 1e-100 * x[nr - 1])
    sig_g = u_gas / x
    sig_g = np.maximum(sig_g, 1e-100)
    return sig_g


def impl_donorcell_adv_diff_delta(n_x, x, Diff, v, g, h, K, L, u_in, dt, pl, pr, ql, qr, rl, rr):
    """
    Implicit donor cell advection-diffusion scheme with piecewise constant values

    NOTE: The cell centers can be arbitrarily placed - the interfaces are assumed
    to be in the middle of the "centers", which makes all interface values
    just the arithmetic mean of the center values.

        Perform one time step for the following PDE:

           du    d  /    \    d  /              d  /       u   \ \
           -- + -- | u v | - -- | h(x) Diff(x) -- | g(x) ----  | | = K + L u
           dt   dx \    /    dx \              dx \      h(x) / /

        with boundary conditions

            dgu/h |            |
          p ----- |      + q u |       = r
             dx   |x=xbc       |x=xbc

    Arguments:
    ----------
    n_x : int
        number of grid points

    x : array-like
        the grid

    Diff : array-like
        value of Diff @ cell center

    v : array-like
        the values for v @ interface (array[i] = value @ i-1/2)

    g : array-like
        the values for g(x)

    h : array-like
        the values for h(x)

    K : array-like
        the values for K(x)

    L : array-like
        the values for L(x)

    u : array-like
        the current values of u(x)

    dt : float
        the time step


    Output:
    -------

    u : array-like
        the updated values of u(x) after timestep dt

    """
    from numpy import zeros
    A = zeros(n_x)
    B = zeros(n_x)
    C = zeros(n_x)
    D = zeros(n_x)
    D05 = zeros(n_x)
    h05 = zeros(n_x)
    rhs = zeros(n_x)
    #
    # calculate the arrays at the interfaces
    #
    for i in range(1, n_x):
        D05[i] = 0.5 * (Diff[i - 1] + Diff[i])
        h05[i] = 0.5 * (h[i - 1] + h[i])
    #
    # calculate the entries of the tridiagonal matrix
    #
    for i in range(1, n_x - 1):
        vol = 0.5 * (x[i + 1] - x[i - 1])
        A[i] = -dt / vol *  \
            (
            max(0., v[i]) +
            D05[i] * h05[i] * g[i - 1] / ((x[i] - x[i - 1]) * h[i - 1])
            )
        B[i] = 1. - dt * L[i] + dt / vol * \
            (
            max(0., v[i + 1]) -
            min(0., v[i]) +
            D05[i + 1] * h05[i + 1] * g[i] / ((x[i + 1] - x[i]) * h[i]) +
            D05[i] * h05[i] * g[i] / ((x[i] - x[i - 1]) * h[i])
            )
        C[i] = dt / vol *  \
            (
            min(0., v[i + 1]) -
            D05[i + 1] * h05[i + 1] * g[i + 1] / ((x[i + 1] - x[i]) * h[i + 1])
            )
        D[i] = -dt * K[i]
    #
    # boundary Conditions
    #
    A[0] = 0.
    B[0] = ql - pl * g[0] / (h[0] * (x[1] - x[0]))
    C[0] = pl * g[1] / (h[1] * (x[1] - x[0]))
    D[0] = u_in[0] - rl

    A[-1] = - pr * g[-2] / (h[-2] * (x[-1] - x[-2]))
    B[-1] = qr + pr * g[-1] / (h[-1] * (x[-1] - x[-2]))
    C[-1] = 0.
    D[-1] = u_in[-1] - rr

    #
    # the delta-way
    #
    for i in range(1, n_x - 1):
        rhs[i] = u_in[i] - D[i] - \
            (A[i] * u_in[i - 1] + B[i] * u_in[i] + C[i] * u_in[i + 1])
    rhs[0] = rl - (B[0] * u_in[0] + C[0] * u_in[1])
    rhs[-1] = rr - (A[-1] * u_in[-2] + B[-1] * u_in[-1])

    #
    # solve for u2
    #
    u2 = tridag(A, B, C, rhs, n_x)
    #
    # update u
    # u = u2   # old way
    #
    u_out = u_in + u2  # delta way

    return u_out


def tridag(a, b, c, r, n):
    """
    Solves a tridiagnoal matrix equation

        M * u  =  r

    where M is tridiagonal, and u and r are vectors of length n.

    Arguments:
    ----------

    a : array
        lower diagonal entries

    b : array
        diagonal entries

    c : array
        upper diagonal entries

    r : array
        right hand side vector

    n : int
        size of the vectors

    Returns:
    --------

    u : array
        solution vector
    """
    import numpy as np

    gam = np.zeros(n)
    u = np.zeros(n)

    if b[0] == 0.:
        raise ValueError('tridag: rewrite equations')

    bet = b[0]

    u[0] = r[0] / bet

    for j in np.arange(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]

        if bet == 0:
            raise ValueError('tridag failed')
        u[j] = (r[j] - a[j] * u[j - 1]) / bet

    for j in np.arange(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


if __name__ == '__main__':
    main()

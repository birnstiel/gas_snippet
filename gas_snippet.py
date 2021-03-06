#!/usr/bin/env python
import numpy as np
import warnings
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

try:
    from numba import njit, jit
except ImportError:
    warnings.warn('numba not available, calculation will be ~8x slower; better install numba!')

    def njit(ob):
        return ob

    jit = njit

# define constants

mu = 2.3
m_p = c.m_p.cgs.value
au = c.au.cgs.value
year = u.year.to(u.s)
k_b = c.k_B.cgs.value
m_star = c.M_sun.cgs.value
Grav = c.G.cgs.value


def smoothstep(x, w):
    """
    Produces an "smoothed heaviside" function

    y = 1/2*exp(x/w)       if x <= 0
    y = 1-1/2*exp(-x/w)    if x >  0

    Arguments
    ---------
    x : array-like
        input x-array

    w : float
        width of the transition

    Output
    ------
    y : array-like
        the function at every given x value

    """
    isscalar = np.isscalar(x)

    x = np.array([1]) * x
    mask = x >= 0
    y = 0.5 * np.exp(np.minimum(x / w, 709.7))
    y[mask] = 1. - 0.5 * np.exp(-x[mask] / w)

    if isscalar:
        return y[0]
    else:
        return y


def calc_alpha(r, sig, alpha_dead=1e-4, alpha_active=1e-2, sig_dz=200., dsigma=20.):
    """
    Returns an alpha array, given the gas surface density and the radial grid.

    Arguments:
    ----------

    r : array
        radial grid [cm]

    sig : array
        gas surface density on grid r [g/cm^2]

    Keywords:
    ---------

    alpha_dead, alpha_active : float
        the active and dead values of alpha

    sig_dz : float
        the surface density at which alpha drops towards alpha_dead

    """
    return alpha_active - (alpha_active - alpha_dead) * smoothstep((sig - sig_dz), dsigma)


def smooth_alpha(alpha, stencil=5):
    """
    Smoothes alpha by a running weighted average

    Arguments:
    ----------

    alpha: array
        turbulence parameter on grid r

    Output:
    -------

    alpha : array
        smoothed turbulence parameter
    """
    n_ghost = stencil // 2
    a_out = np.hstack((
        alpha[0] * np.ones(n_ghost),
        alpha,
        alpha[-1] * np.ones(n_ghost)))

    n = len(a_out)

    return np.sum([a_out[stencil - 1 - s: n - s] for s in range(stencil)], 0) / stencil


def gas_timestep_active_dead(dt, x, T, sig_g, sig_thresh=200.0,
                             alpha_active=1e-2, alpha_dead=1e-4, dsigma=20.0):
    """
    Does a layered accretion gas time step, distinguishing between active
    and dead layer

    Arguments:
    ----------

    dt : float
        length of time step [s]

    x, T, sig_g : arrays
        radial grid x and on this grid: temperature T [K] and surface density
        sig_g [g/cm^2]

    sig_thresh : float
        at which gas surface density to transition from active to dead

    alpha_active, alpha_dead : float
        active and dead-zone alpha values

    Output:
    -------

    sig, alpha

    sig : array
        updated gas surface density

    alpha : array
        mid-plane gas surface density
    """

    # find the active and dead surface densities

    sigma_active = np.minimum(sig_g, sig_thresh)
    sigma_dead = np.maximum(sig_g - sigma_active, 1e-100)

    # assign the active and dead alpha values

    alpha_a = alpha_active * np.ones_like(x)
    alpha_d = calc_alpha(x, sig_g, alpha_dead=alpha_dead,
                         alpha_active=alpha_active, sig_dz=sig_thresh, dsigma=dsigma)
    alpha_d = smooth_alpha(alpha_d, stencil=3)

    # evolve the two surface densities separately

    sig_dead = gas_timestep(dt, x, alpha_d, T, sigma_dead)
    sig_active = gas_timestep(dt, x, alpha_a, T, sigma_active)

    # calculate the total surface density and mid-plane alpha

    sig = sig_dead + sig_active
    alpha = np.minimum(alpha_a, alpha_d)

    return sig, alpha


def gas_timestep_active_dead_avg(
        dt, x, T, sig_g, sig_thresh=200.0, alpha_active=1e-2, alpha_dead=1e-4,
        dsigma=20.0):
    """
    Does a layered accretion gas time step, distinguishing between active
    and dead layer

    Arguments:
    ----------

    dt : float
        length of time step [s]

    x, T, sig_g : arrays
        radial grid x and on this grid: temperature T [K] and surface density
        sig_g [g/cm^2]

    sig_thresh : float
        at which gas surface density to transition from active to dead

    alpha_active, alpha_dead : float
        active and dead-zone alpha values

    Output:
    -------

    sig, alpha

    sig : array
        updated gas surface density

    alpha : array
        mid-plane gas surface density
    """

    # find the active and dead surface densities

    sigma_active = np.minimum(sig_g, sig_thresh)
    sigma_dead = np.maximum(sig_g - sigma_active, 1e-100)

    # assign the active and dead alpha values

    alpha_a = alpha_active * np.ones_like(x)
    alpha_d = calc_alpha(x, sig_g, alpha_dead=alpha_dead,
                         alpha_active=alpha_active, sig_dz=sig_thresh, dsigma=dsigma)
    alpha_d = smooth_alpha(alpha_d, stencil=3)

    # calculate an averaged alpha

    alpha_mean = (alpha_a * sigma_active + alpha_d * sigma_dead) / (sig_g)

    # evolve the two surface densities separately

    sig = gas_timestep(dt, x, alpha_mean, T, sig_g)

    return sig, alpha_mean


def gas_timestep(dt, x, alpha, T, sig_g):
    """
    Does an implicit time step for the gas surface density:

    Arguments:
    ----------

    dt : float
        time step [s]

    x, alpha, T, sig_g : arrays
        radial grid x, and on the same grid, turbulence alpha, temperature T,
        and gas surface density sig_g

    Output:
    -------

    sig : float
        updated surface density after time step dt.
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
    # p_L = -(x[1] - x[0]) * h_gas[1] / (x[1] * g_gas[1])
    # q_L = 1. / x[0] - 1. / x[1] * g_gas[0] / g_gas[1] * h_gas[1] / h_gas[0]
    # r_L = 0.0

    p_L = 1.0
    q_L = - (g_gas[1] / h_gas[1] - g_gas[0] / h_gas[0]) / (x[1] - x[0])
    r_L = g_gas[0] / h_gas[0] * (u_gas[1] - u_gas[0]) / (x[1] - x[0])

    u_gas = impl_donorcell_adv_diff_delta(nr, x, D_gas, v_gas, g_gas, h_gas, K_gas, L_gas,
                                          u_gas, dt, p_L, 0.0, q_L, 1.0, r_L, 1e-100 * x[nr - 1])
    sig_g = u_gas / x
    sig_g = np.maximum(sig_g, 1e-100)
    return sig_g


@njit
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


@jit
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
    A = np.zeros(n_x)
    B = np.zeros(n_x)
    C = np.zeros(n_x)
    D = np.zeros(n_x)
    D05 = np.zeros(n_x)
    h05 = np.zeros(n_x)
    rhs = np.zeros(n_x)
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
            max(0., v[i]) + D05[i] * h05[i] * g[i - 1] / ((x[i] - x[i - 1]) * h[i - 1])
            )
        B[i] = 1. - dt * L[i] + dt / vol * \
            (
            max(0., v[i + 1]) - min(0., v[i]) + D05[i + 1] * h05[i + 1] * g[i] / ((x[i + 1] - x[i]) * h[i]) + D05[i] * h05[i] * g[i] / ((x[i] - x[i - 1]) * h[i])
            )
        C[i] = dt / vol *  \
            (
            min(0., v[i + 1]) - D05[i + 1] * h05[i + 1] * g[i + 1] / ((x[i + 1] - x[i]) * h[i + 1])
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

    u_out = u_in + u2  # delta way

    return u_out


def main():

    # define grids

    nr = 1000
    nt = 2000

    # xi = np.logspace(-1, 3, nr + 1) * au
    # x = 0.5 * (xi[1:] + xi[:-1])

    x = np.logspace(-1, 3, nr + 2) * au
    xi = 0.5 * (x[1:] + x[:-1])
    x = x[1:-1]

    time = np.logspace(-1, 6, nt - 1) * year

    # define initial condition

    time = np.hstack((0, time))
    sig_0 = 200. * (x / au)**-1
    sig_0[x > 100 * au] = 1e-100
    alpha = calc_alpha(x, sig_0)
    T = 200 * (x / au)**-0.5

    # iteration

    sig_g = np.zeros([nt, nr])
    sig_g[0] = sig_0
    alpha_out = np.zeros_like(sig_g)
    alpha_out[0] = alpha
    sig = sig_0.copy()

    for it in range(nt - 1):
        dt = time[it + 1] - time[it]

        # sig = gas_timestep(dt, x, alpha, T, sig)
        sig, alpha = gas_timestep_active_dead_avg(dt, x, T, sig, dsigma=5.0)
        sig_g[it + 1] = sig
        alpha_out[it + 1] = alpha

    # plotting

    if has_widget:
        widget.plotter(x / au, sig_g, data2=[alpha_out], times=time / year,
                       xlog=True, ylog=True, ylim=[1e-4, 1e4],
                       xlabel='$r$ [au]', ylabel=r'$\Sigma_\mathrm{g}$ [g cm$^{-2}$]')
    else:
        f, ax = plt.subplots()
        ax.loglog(x / au, sig_g[-1])
        ax.set_ylim(1e-4, 1e4)
        ax.set_xlabel('$r$ [au]')
        ax.set_ylabel(r'$\Sigma_\mathrm{gas}$ [g/cm$^3$]')

    plt.show()

    return locals()


if __name__ == '__main__':
    res = main()

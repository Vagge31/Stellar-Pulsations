from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt

"""Newtonian linear adiabatic stellar pulsation solver (shooting method).

Works on top of a dimensionless background model (x = r/R in [0,1], plus
interpolants for density, gravity, sound speed squared and the
Brunt-Vaisala frequency squared, all built from a Lane-Emden solution, see
`solvers.background.solve_lane_emden` and lane-emden.ipynb) to find the
adiabatic normal-mode eigenfrequencies and eigenfunctions of the star, for
both radial (l=0) and non-radial (l>0) oscillations.

The non-radial equations are written in the Cowling approximation, i.e. the
perturbation to the gravitational potential is neglected, so only two
first-order ODEs (rather than four) are needed for the perturbed radial
displacement and pressure.
"""

def pulsation_system(x, y, omega2_hat, l, bg_rho, bg_g, bg_c2, bg_N2):
    """Right-hand side of the adiabatic pulsation ODEs at dimensionless radius x.

    Parameters
    ----------
    x : float
        Dimensionless radius, r/R, in (0, 1).
    y : array_like of length 2
        State vector. For l=0 (radial): y=(zeta, dzeta/dx), the Lagrangian
        radial displacement relative to r (zeta = dr/r) and its derivative.
        For l>0 (non-radial, Cowling approximation): y=(y1, y2), the
        rescaled radial displacement and the Eulerian pressure perturbation.
    omega2_hat : float
        Trial dimensionless squared oscillation eigenfrequency.
    l : int
        Angular/spherical-harmonic degree of the mode (l=0 is radial).
    bg_rho, bg_g, bg_c2, bg_N2 : callables
        Interpolated dimensionless background profiles (density, local
        gravity, sound speed squared, Brunt-Vaisala frequency squared) as
        functions of x.

    Returns
    -------
    list of length 2
        [dy1/dx, dy2/dx].
    """

    rho = bg_rho(x)
    g = bg_g(x)
    c2 = bg_c2(x)
    N2 = bg_N2(x)
    gamma = 5/3

    if l == 0:
        # Radial (l=0) system: linear adiabatic radial pulsation equation,
        # written as a first-order system in (zeta, zeta').
        y1, y2 = y
        dy1_dx = y2
        dy2_dx = (gamma * g/c2 + 2/x)*y2 - (1/c2)*(omega2_hat + 4*g/x)*y1
        return [dy1_dx, dy2_dx]
    else:
        # Non-radial (l>0) system in the Cowling approximation.
        y1, y2 = y
        dy1_dx = ((c2 * l *(l+1))/(x**2 *omega2_hat) - 1)*(y2/(rho * c2)) - (2/x) * y1 + (g/c2)*y1
        dy2_dx = rho*(omega2_hat-N2)*y1 - (g/c2)*y2
        return [dy1_dx, dy2_dx]


def solve_system(omega2_hat, l, bg_rho, bg_g, bg_c2, bg_N2):
    """Shoot outward from the center and return the surface boundary-condition residual.

    Integrates `pulsation_system` from a small x0 near the center (using the
    regular series-expansion initial conditions appropriate for the given l)
    out to x=xend near the surface, then evaluates how well the outer
    (zero surface pressure perturbation) boundary condition is satisfied for
    the trial eigenfrequency omega2_hat. This residual is what
    `get_eigenfrequency` drives to zero via root-finding: omega2_hat is a
    true eigenfrequency exactly when the boundary conditions at both the
    center and the surface can be satisfied simultaneously.

    Parameters
    ----------
    omega2_hat : float
        Trial dimensionless squared eigenfrequency.
    l : int
        Angular degree (l=0 is radial).
    bg_rho, bg_g, bg_c2, bg_N2 : callables
        Background profile interpolants, see `pulsation_system`.

    Returns
    -------
    float
        Residual of the surface boundary condition (its root in omega2_hat
        gives an eigenfrequency); np.nan if the integration failed.
    """
    gamma = 5/3
    x0 = 1e-4
    if l == 0:
        # Regular near-center behaviour of the radial system (zeta, zeta').
        y1_start = x0**3
        y2_start = 3*x0**2

    else:
        rho_c = bg_rho(x0)
        # Regular near-center power-law behaviour (~x^(l-1), ~x^l) required
        # for a well-behaved (non-singular) non-radial solution at x=0.
        y1_start = x0**(l-1)
        y2_start = (rho_c*omega2_hat/l)*x0**l

    y0 = [y1_start, y2_start]

    xend = 0.99

    sol = solve_ivp(
        pulsation_system,
        [x0, xend],
        y0,
        args=(omega2_hat, l, bg_rho, bg_g, bg_c2, bg_N2),
        method='RK45'
    )
    if not sol.success:
        return np.nan

    y1_surf = sol.y[0, -1]
    y2_surf = sol.y[1, -1]

    rho_surf = bg_rho(xend)
    g_surf = bg_g(xend)

    if l == 0:
        # Vanishing Lagrangian pressure perturbation at the free surface.
        residual = y2_surf-(4+omega2_hat/g_surf)*(y1_surf/gamma)

    else:
        # Vanishing Eulerian pressure perturbation at the free surface.
        residual = y2_surf - y1_surf*rho_surf*g_surf

    return residual

def get_eigenfrequency(l, omega_min, omega_max, bg_rho, bg_g, bg_c2, bg_N2):
    """Find a mode's dimensionless squared eigenfrequency by bisection.

    Uses Brent's method (scipy.optimize.brentq) to find the root of
    `solve_system` (the surface boundary-condition residual) in the bracket
    [omega_min, omega_max]. The bracket must be chosen so the residual
    changes sign across it (e.g. from an initial scan over trial
    frequencies), i.e. it must contain exactly one eigenfrequency.

    Parameters
    ----------
    l : int
        Angular degree of the mode.
    omega_min, omega_max : float
        Bracketing interval for the trial squared eigenfrequency.
    bg_rho, bg_g, bg_c2, bg_N2 : callables
        Background profile interpolants, see `pulsation_system`.

    Returns
    -------
    float or None
        The eigenfrequency omega^2 if a root was bracketed, otherwise None
        (with a diagnostic message printed).
    """

    try:
        eig_freq = brentq(
            solve_system,
            omega_min,
            omega_max,
            args=(l, bg_rho, bg_g, bg_c2, bg_N2),
            xtol=1e-6
        )
        return eig_freq
    except ValueError as e:
        print(f"Αποτυχία εύρεσης ρίζας στο διάστημα [{omega_min}, {omega_max}].")
        print("Σφάλμα:", e)
        return None



def get_eigenfunction(omega2_hat, l, bg_rho, bg_g, bg_c2, bg_N2):
    """Integrate the pulsation system at a fixed frequency to get the eigenfunction.

    Same shooting integration as `solve_system`, but for a frequency already
    known to be an eigenfrequency (e.g. from `get_eigenfrequency`), returning
    the full dense solution so the eigenfunction (y1, y2) can be evaluated
    and plotted throughout the star.

    Parameters
    ----------
    omega2_hat : float
        Eigenfrequency (squared, dimensionless) at which to evaluate the
        eigenfunction.
    l : int
        Angular degree of the mode.
    bg_rho, bg_g, bg_c2, bg_N2 : callables
        Background profile interpolants, see `pulsation_system`.

    Returns
    -------
    OdeResult
        solve_ivp result with dense_output=True, giving y1(x), y2(x) on
        [x0, xend].
    """

    x0 = 1e-4

    if l == 0:
        y1_start = x0**3
        y2_start = 3*x0**2
    else:
        rho_c = bg_rho(x0)
        y1_start = x0**(l-1)
        y2_start = (rho_c*omega2_hat/l)*x0**l

    y0 = [y1_start, y2_start]

    xend = 0.999

    sol = solve_ivp(
        pulsation_system,
        [x0, xend],
        y0,
        args=(omega2_hat, l, bg_rho, bg_g, bg_c2, bg_N2),
        method='RK45',
        dense_output=True
    )
    return sol


def rayleigh_quotient(x, y1, y2, P_func, Q_func, W_func):
    """Estimate omega^2 for a mode via the variational Rayleigh quotient.

    Evaluates the standard variational (Rayleigh) estimate of the squared
    eigenfrequency,
        omega^2 = Integral[P*y2^2 + Q*y1^2] / Integral[W*y1^2]
    over the eigenfunction (y1, y2) using Simpson's rule, given the
    problem-specific weight functions P_func, Q_func, W_func (see P_newt,
    Q_newt, W_newt in lane-emden.ipynb). Used as an independent check on
    eigenfrequencies obtained from the shooting method.

    Parameters
    ----------
    x : array_like
        Dimensionless radial grid on which the eigenfunction is sampled.
    y1, y2 : array_like
        Eigenfunction components sampled on x (e.g. from `get_eigenfunction`).
    P_func, Q_func, W_func : callables
        Weight functions of x entering the variational integral.

    Returns
    -------
    float
        Rayleigh-quotient estimate of omega^2.
    """

    P = P_func(x)
    Q = Q_func(x)
    W = W_func(x)

    return (simpson(y=P*y2**2 - Q*y1**2, x=x))/(simpson(y=W*y1**2, x=x))
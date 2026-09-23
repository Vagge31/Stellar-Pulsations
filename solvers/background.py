import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Newtonian case: Lane-Emden equation for a polytropic star
# ---------------------------------------------------------------------------
def solve_lane_emden(n, xi_start=1e-5, xi_max=100):
    """Solve the dimensionless Lane-Emden equation for a polytrope of index n.

    The Lane-Emden equation (in the standard dimensionless variables
    theta(xi), xi) is:
        theta'' + (2/xi) theta' + theta^n = 0
    describing hydrostatic equilibrium of a Newtonian polytropic star with
    P = K * rho^((n+1)/n). Integration starts at a small xi_start (instead of
    xi=0, which is a regular singular point) using the analytic series
    expansion of theta near the origin, and proceeds outward until theta
    first reaches zero, which defines the dimensionless stellar surface xi_1.

    Parameters
    ----------
    n : float
        Polytropic index.
    xi_start : float
        Small starting value of xi used to avoid the coordinate singularity
        at the origin (theta and theta' are set from the series expansion).
    xi_max : float
        Upper bound on xi in case the surface (theta=0) is never reached.

    Returns
    -------
    xi_surface : float
        Dimensionless radius xi_1 where theta first vanishes (np.inf if the
        surface was not found before xi_max).
    solution : OdeResult
        Full solve_ivp result (with dense_output=True) so theta(xi) and
        theta'(xi) can be evaluated anywhere on [xi_start, xi_surface].
    """

    def lane_emden_odes(xi, y):
        th, dth_dxi = y

        # Clamp to avoid a small negative theta (from numerical overshoot)
        # being raised to a non-integer power n.
        th_safe = max(th, 1e-12)

        dy0_dxi = dth_dxi
        dy1_dxi = (-th_safe**n) - (2/xi)*dth_dxi

        return [dy0_dxi, dy1_dxi]

    def surface_event(xi, y):
        return y[0]
    surface_event.terminal = True
    surface_event.direction = -1

    # Leading-order series expansion of theta(xi) near xi=0:
    # theta(xi) ~= 1 - xi^2/6 + ...
    th_start = 1-(xi_start**2/6)
    dth_start = -xi_start/3

    y0 = [th_start, dth_start]

    solution = solve_ivp(
        fun=lane_emden_odes,
        t_span=(xi_start, xi_max),
        y0=y0,
        method='RK45',
        events=surface_event,
        dense_output=True,
        rtol=1e-9,
        atol=1e-12
    )

    if solution.t_events[0].size > 0:
        xi_surface = solution.t_events[0][0]
    else:
        xi_surface = np.inf
    return xi_surface, solution



def eos(P):
    """Polytropic equation of state rho(P) used for the GR (TOV) models.

    Density is split into a rest-mass part following P = K * rho_0^Gamma
    plus an internal-energy contribution P/(Gamma-1), i.e.
        rho(P) = (P/K)^(1/Gamma) + P/(Gamma-1)
    with K=100, Gamma=2 (in geometrized units, G=c=1) fixed here. This
    mirrors the "Gamma=2" polytrope models compared against the Newtonian
    Lane-Emden background elsewhere in the project.
    """
    K = 100
    G = 2
    return (P/K)**(1/G) + P/(G-1)

def eos_kokkotas(P):
        K = 100
        G = 2
        return (P/K)**(1/G)

def real_eos(t):
    """Parametric realistic (degenerate free-neutron gas) equation of state.

    Reproduces the classic Oppenheimer-Volkoff free-neutron-gas EOS given in
    parametric form via an auxiliary variable t (related to the neutron
    Fermi momentum), rather than as a closed-form rho(P):git s
        rho(t) = K * (sinh(t) - t)
        P(t)   = (K/3) * (sinh(t) - 8*sinh(t/2) + 3*t)
    with K=4.251e-4 in geometrized units. Used together with interp1d (see
    tov.ipynb) to build an effective rho(P) function for solve_tov.

    Returns
    -------
    rho, P : arrays or floats matching the shape of t.
    """
    K = 4.251e-4
    rho = K*(np.sinh(t)-t)
    P = (K/3)*(np.sinh(t) - 8*np.sinh(t/2) + 3*t)

    return rho, P

# ---------------------------------------------------------------------------
# General Relativistic case: Tolman-Oppenheimer-Volkoff (TOV) equations
# ---------------------------------------------------------------------------
def solve_tov(Pc, EoS, r0=1e-6, rb=np.inf):
    """Integrate the TOV equations for a static, spherically symmetric star.

    Solves (in geometrized units, G=c=1) the relativistic hydrostatic
    equilibrium equations for enclosed mass m(r) and pressure P(r):
        dm/dr = 4*pi*r^2*rho
        dP/dr = -(rho+P) * (m + 4*pi*r^3*P) / (r*(r-2m))
    given a supplied equation of state rho = EoS(P). Integration starts at a
    small radius r0 (using the small-r regular expansion m ~ (4/3)*pi*r0^3*rho_c)
    to avoid the r=0 coordinate singularity, and stops at the stellar surface,
    defined as the radius where the pressure drops to (numerically) zero.

    Parameters
    ----------
    Pc : float
        Central pressure (in the same geometrized units as EoS expects/returns).
    EoS : callable
        Equation of state rho(P), e.g. `eos` or an interpolant built from
        `real_eos`.
    r0 : float
        Small starting radius used to avoid the coordinate singularity at r=0.
    rb : float
        Upper bound on the integration domain (default: unbounded).

    Returns
    -------
    R : float
        Stellar radius (surface where P first drops to ~0); np.inf if not found.
    M : float
        Total gravitational mass enclosed at R; np.inf if the surface was not found.
    solution : OdeResult
        Full solve_ivp result (with dense_output=True) for m(r), P(r).
    """

    def tov_eqs(r, y):
        m, P, Phi = y

        rho = EoS(P)
        dm_dr = 4*np.pi*r**2 * rho
        dP_dr = -(rho + P)*((m+4*np.pi*r**3 * P)/(r*(r-2*m)))
        dPhi_dr = -dP_dr/(rho+P)
        return [dm_dr, dP_dr, dPhi_dr]

    def surface_event(r, y):
        # Trigger just above P=0 (1e-11) to avoid stalling the integrator
        # as dP/dr steepens near the true surface.
        return y[1]-(Pc*1e-8)
    surface_event.terminal = True
    surface_event.direction = -1

    rho_c = EoS(Pc)
    # Regular (non-singular) expansion of the enclosed mass near r=0,
    # assuming near-constant density rho_c inside the small core [0, r0].
    m_start = (4/3)*np.pi*r0**3 * rho_c
    P_start = Pc
    Phi_start = 0.0 #By hand
    y0 = [m_start, P_start, Phi_start]

    solution = solve_ivp(
        fun=tov_eqs,
        t_span=(r0, rb),
        y0=y0,
        method='RK45',
        events=surface_event,
        dense_output=True,
        rtol=1e-9,
        atol=Pc*1e-10
    )
    if solution.t_events[0].size > 0:
        R = solution.t_events[0][0]
        M = solution.y_events[0][0][0]
        Phi_surf = solution.y_events[0][0][2]
        Phi_true = (1/2)*np.log(1 - (2*M/R))
        Phi_offset = Phi_true - Phi_surf
    else:
        R = np.inf
        M = np.inf
        Phi_offset = 0.0
    return R, M, Phi_offset, solution


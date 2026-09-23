import numpy as np
from scipy.integrate import solve_ivp 
from scipy.optimize import brentq
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import numpy as np


def solve_system(r,P,W,Q, omega2): 
    #P = interp1d(r,P_fun,"linear",fill_value="extrapolate") 
    #W = interp1d(r,W_fun,"linear",fill_value="extrapolate")
    #Q = interp1d(r,Q_fun,"linear",fill_value="extrapolate")

    def pulsation_system(r_grid,y):
        zeta,eta = y 
        dzeta_dr = eta/P(r_grid) 
        deta_dr = -(omega2*W(r_grid) + Q(r_grid))*zeta

        return [dzeta_dr,deta_dr]

    r_start = r[0]
    r_end = r[-1] #Ακτίνα του άστρου, θα υπολογίζεται από το 
    eta_0 = 1
    zeta_0 = r_start/(3*P(r_start))

    y0 = [zeta_0,eta_0]
    sol = solve_ivp(
        pulsation_system,
        t_span=[r_start,r_end],
        y0=y0,
        method="BDF",
        dense_output=True,
        rtol=1e-8, 
        atol=1e-10
    )
    if not sol.success:
         print(f"Ο λύτης απέτυχε στο w2={omega2:.5f}. Αιτία: {sol.message}")

    residue = sol.y[1][-1] #Η 2η οριακή, αυτό θα τσεκάρω αν == 0, και αν ναι τοτε κρατάω την ιδιοσυχνότητα
    return residue, sol 

def get_mode(omega_min, omega_max,r,P_fun,W_fun,Q_fun):

    def bc_surf(omega2_guess):
        surf, _ = solve_system(r,P_fun,W_fun,Q_fun, omega2_guess)
        return surf

    try:
        eig_freq = brentq(
            bc_surf,
            omega_min,
            omega_max,
            xtol=1e-6
        )
        return eig_freq
    except ValueError as e:
            print(f"Αποτυχία εύρεσης ρίζας στο διάστημα [{omega_min}, {omega_max}].")
            print("Σφάλμα:", e)
            return None






import numpy as np
from scipy.integrate import solve_ivp 
from scipy.optimize import brentq
from scipy.integrate import simpson
import numpy as np

def pulsation_system(r,y,P_fun,W_fun,Q_fun, omega2):
    P = P_fun(r)
    W = W_fun(r)
    Q = Q_fun(r)

    zeta,eta = y 
    dzeta_dr = eta/P 
    deta_dr = -(omega2*W + Q)*zeta

    return [dzeta_dr,deta_dr]
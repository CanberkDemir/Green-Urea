"""Correlations transcribed from the A4 dossier; domain limits apply.

Solubility: NaCl proxy. Diffusion: water correlation, not hot-brine validation.
Iliuta: downflow trickle beds, extrapolative at 60 C and 1 m diameter.
"""
import math

def hydrogen_equilibrium(T, P_MPa, f_MPa, psat_MPa, salt_molality, water_kg_m3):
    theta = T/273.15 - 1
    H = math.exp(4.6449 + 3.3252*theta - 10.901*theta**2 + 8.0526*theta**3)
    v = 1e-6*(19.615 - 7.6400*theta + 33.425*theta**2)
    ks = 0.2898 - 1.4330*theta + 3.9584*theta**2 - 3.1666*theta**3
    b = f_MPa/H*math.exp(-ks*salt_molality-v*(P_MPa-psat_MPa)*1e6/(8.314462618*T))
    return b*water_kg_m3  # mol/m3 of solution

import numpy as np

def iliuta_kla(ReL, WeL, XG, MoL, ScL, Sb, dh, DL):
    x = np.array([ReL, WeL, XG, MoL, ScL, Sb])
    lo = np.array([1.994e-2, 1.851e-6, 2.636e-3, 2.545e-15, 44.33, 2.552])
    span = np.array([4.898,6.298,4.918,10.114,4.042,1.9011])
    U = np.r_[np.log10(x/lo)/span, 1.]
    W = np.array([
      [1.7352,-1.0859,4.4131,5.9993,5.5325,-0.8589,2.7195,1.8965,-0.4475,-1.7098,1.9772,-1.8640,6.8217],
      [3.7209,4.3527,-0.9842,2.2940,-1.2809,1.5063,-2.1694,5.1352,2.4182,-3.1443,-7.2618,6.3021,7.4821],
      [-3.1528,-5.9518,1.4652,-5.1562,-0.7164,-0.6652,-1.2966,4.2572,-1.2258,-0.5253,-4.1405,-2.4334,2.5210],
      [10.8069,0.3045,2.5158,4.6663,-3.2695,4.2675,-4.6230,-3.7013,0.6073,-0.7606,0.1795,2.5706,0.0143],
      [-5.7494,-0.8630,6.0466,3.2448,2.0916,-7.2309,0.4930,-1.9482,0.1645,4.6749,1.1673,2.0291,0.6759],
      [9.2444,1.2997,10.0231,8.7118,-5.8084,-5.6044,2.3693,11.4437,9.7706,-6.4817,3.1763,4.8084,8.9547],
      [-5.0819,5.1608,-6.9573,3.3028,3.9598,5.6439,2.6890,-5.2396,-1.6727,-2.3656,4.6354,-1.7145,1.2650]])
    v = np.array([5.2505,-3.9571,7.6703,4.3998,9.9828,-8.1581,6.3469,-4.3963,-7.0899,3.4881,-6.1047,-4.3143,5.5141,-4.4515])
    sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-700,700)))
    S = sigmoid(np.r_[sigmoid(U@W),1.]@v)
    Sh = 0.3359*10**(5.2428*S)
    return Sh*DL/dh**2, U[:-1] # s^-1, normalized inputs (flag any outside 0..1)


def hydrogen_diffusivity(T, P_MPa, salinity_factor=1.0):
    if not 293 <= T <= 393 or not 0 <= P_MPa <= 20:
        raise ValueError("Diffusivity outside temperature/pressure domain")
    if not 0 < salinity_factor <= 1:
        raise ValueError("Salinity factor must be in (0,1]")
    return 4.182e-11*math.exp(1.380649e-23*T/8.425e-22)*(1-0.009703*P_MPa)*salinity_factor

# Table S1: measurements, not a mechanistic product network.
BATCH_TIME_H = (1., 2., 3., 4.)
BATCH_NO3_MM = (0.6, 0.2, 0.1, 0.02)
BATCH_NH4_MM = (0.4, 0.5, 0.8, 1.5)

def batch_calibration():
    k = sum(t*math.log(1.6/c) for t,c in zip(BATCH_TIME_H,BATCH_NO3_MM))/sum(t*t for t in BATCH_TIME_H)
    return k, k/(3600*(0.010/0.030))

def batch_report():
    k, kw = batch_calibration()
    return {"k_app_h": k, "k_W_m3_kg_s": kw,
            "predicted_no3_mM": [1.6*math.exp(-k*t) for t in BATCH_TIME_H],
            "unassigned_N_mM": [1.6-n-a for n,a in zip(BATCH_NO3_MM,BATCH_NH4_MM)]}

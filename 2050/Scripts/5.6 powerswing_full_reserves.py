# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:22:21 2025

@author: navia
"""

#%%

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import itertools
import time

# ============================
# PARÁMETROS FIJOS (puedes ajustar)
# ============================
Contingency_val = 400.0     # MW
H_default = 25.0            # valor por defecto (se variará en la búsqueda)
D = 1.0                     # amortiguamiento simple MW/Hz


# PERFIL DE TIEMPO (igual que tu original)
t = np.arange(0, 2000, 0.01)


# ============================
# TIEMPOS DE ACTIVACIÓN (basados en TSO)
# ============================
t_ffr_start = 1
t_ffr_ramp_end = 2
t_ffr_hold_end = 10
t_ffr_ramp_down_end = 15

t_pfr_start = 6
t_pfr_ramp_end = 17
t_pfr_hold_end = 32
t_pfr_ramp_down_end = 150

t_affr_start = 30
t_affr_ramp_end = 150
t_affr_hold_end = 480
t_affr_ramp_down_end = 900

t_mffr_start = 450
t_mffr_ramp_end = 900

def compute_weights(tt):
    """Devuelve w_ffr, w_pfr, w_affr, w_mffr para vector tt (igual que tu implementación)."""
    w_ffr = np.piecewise(tt,
        [tt < t_ffr_start,
         (t_ffr_start <= tt) & (tt < t_ffr_ramp_end),
         (t_ffr_ramp_end <= tt) & (tt < t_ffr_hold_end),
         (t_ffr_hold_end <= tt) & (tt < t_ffr_ramp_down_end),
         tt >= t_ffr_ramp_down_end],
        [0,
         lambda t: (t - t_ffr_start) / (t_ffr_ramp_end - t_ffr_start),
         1,
         lambda t: 1 - (t - t_ffr_hold_end) / (t_ffr_ramp_down_end - t_ffr_hold_end),
         0]
    )

    w_pfr = np.piecewise(tt,
        [tt < t_pfr_start,
         (t_pfr_start <= tt) & (tt < t_pfr_ramp_end),
         (t_pfr_ramp_end <= tt) & (tt < t_pfr_hold_end),
         (t_pfr_hold_end <= tt) & (tt < t_pfr_ramp_down_end),
         tt >= t_pfr_ramp_down_end],
        [0,
         lambda t: (t - t_pfr_start) / (t_pfr_ramp_end - t_pfr_start),
         1,
         lambda t: 1 - (t - t_pfr_hold_end) / (t_pfr_ramp_down_end - t_pfr_hold_end),
         0]
    )

    w_affr = np.piecewise(tt,
        [tt < t_affr_start,
         (t_affr_start <= tt) & (tt < t_affr_ramp_end),
         (t_affr_ramp_end <= tt) & (tt < t_affr_hold_end),
         (t_affr_hold_end <= tt) & (tt < t_affr_ramp_down_end),
         tt >= t_affr_ramp_down_end],
        [0,
         lambda t: (t - t_affr_start) / (t_affr_ramp_end - t_affr_start),
         1,
         lambda t: 1 - (t - t_affr_hold_end) / (t_affr_ramp_down_end - t_affr_hold_end),
         0]
    )

    w_mffr = np.piecewise(tt,
        [tt < t_mffr_start,
         (t_mffr_start <= tt) & (tt < t_mffr_ramp_end),
         tt >= t_mffr_ramp_end],
        [0,
         lambda t: (t - t_mffr_start) / (t_mffr_ramp_end - t_mffr_start),
         1]
    )

    return w_ffr, w_pfr, w_affr, w_mffr

# Precompute weights vector for plotting and for fixed deployment logic
W_ffr_vec, W_pfr_vec, W_affr_vec, W_mffr_vec = compute_weights(t)

# ============================
# FUNCIÓN DE SIMULACIÓN
# ============================
def run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, H_val,
                   contingency=Contingency_val, D_local=D, verbose=False):
    """
    Ejecuta simulación y devuelve métricas:
    - max_freq_dev (Hz, valor absoluto máximo de desviación)
    - max_rocof (Hz/s, valor absoluto máximo)
    - results_df (DataFrame con series temporales)
    """
    def state(y, tt):
        f = y[0]
        contingency_local = contingency if tt >= 1.0 else 0.0

        # pesos en instante tt (scalar)
        w_ffr, w_pfr, w_affr, w_mffr = compute_weights(np.array([tt]))
        w_ffr = float(w_ffr[0]); w_pfr = float(w_pfr[0]); w_affr = float(w_affr[0]); w_mffr = float(w_mffr[0])

        # reservas desplegadas
        ffr = FFR_cap * w_ffr
        pfr = PFR_cap * w_pfr
        affr = aFFR_cap * w_affr
        mffr = mFFR_cap * w_mffr

        deltap = contingency_local - (ffr + pfr + affr + mffr) - D_local * f
        dfdt = deltap / (1000.0 * (2.0 * H_val / 50.0))
        return [dfdt, deltap]

    y0 = [0.0, 0.0]
    sol = odeint(state, y0, t)
    f = sol[:, 0]


    ffr = FFR_cap * W_ffr_vec
    pfr = PFR_cap * W_pfr_vec
    affr = aFFR_cap * W_affr_vec
    mffr= mFFR_cap * W_mffr_vec


    contingency_vec = np.where(t >= 1.0, contingency, 0.0)
    deltap = contingency_vec - (ffr + pfr + affr + mffr) - D_local * f

    max_freq_dev = np.max(np.abs(f))
    rocof = -np.gradient(f, t)
    max_rocof = np.max(np.abs(rocof))

    results_df = pd.DataFrame({
        'Time [s]': t,
        'Frequency Deviation [Hz]': -f,
        'RoCoF [Hz/s]': rocof,
        'FFR [MW]': ffr,
        'PFR [MW]': pfr,
        'aFFR [MW]': affr,
        'mFFR [MW]': mffr,

        'Contingency [MW]': contingency_vec,
        'deltap [MW]': -deltap
    })

    if verbose:
        print(f"Sim finished: FFR={FFR_cap},PFR={PFR_cap},aFFR={aFFR_cap},mFFR={mFFR_cap},H={H_val}")
        print(f" -> max_freq_dev={max_freq_dev:.4f} Hz, max_rocof={max_rocof:.4f} Hz/s")

    return max_freq_dev, max_rocof, results_df

# ============================
# OPTIMIZACIÓN CON PENALIZACIÓN FUERTE
# ============================
def optimize_search(limit_freq=0.8, limit_rocof=0.25, limit_freq_steady_state=0.2, maxiter=20, popsize=8):
    """
    Optimiza capacidad total (FFR+PFR+aFFR+mFFR+RR_max) minimizando
    y penalizando fuerte si se violan límites de frecuencia, ROCOF,
    y límite de desviación de frecuencia en estado estable.
    """

    bounds = [
        (0, 500),  # FFR
        (0, 400),   # PFR
        (0, 400),   # aFFR
        (0, 400),   # mFFR
        (1, 25)     # H (inercias)
    ]

    def objective(x):
        FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, H_val = x
        max_fd, max_r, results_df = run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, H_val)
        total_capacity = FFR_cap + PFR_cap + aFFR_cap + mFFR_cap 

        penalty = 0
        # Penalización por frecuencia máxima
        if max_fd > limit_freq:
            penalty += 1e5 * (max_fd - limit_freq)**2
        # Penalización por ROCOF máximo
        if max_r > limit_rocof:
            penalty += 1e5 * (max_r - limit_rocof)**2
        # Penalización por desviación frecuencia estado estable (último valor absoluto)
        freq_steady_state_vals = results_df['Frequency Deviation [Hz]'].iloc[90000:].abs()
        if (freq_steady_state_vals > limit_freq_steady_state).any():
            penalty += 1e5 * (freq_steady_state_vals.max() - limit_freq_steady_state)**2
        
        return total_capacity + penalty

    print("Iniciando optimización (differential_evolution)...")
    start_time = time.time()

    result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, disp=True)

    elapsed_time = time.time() - start_time
    print(f"Optimización terminada en {elapsed_time:.1f} segundos.")

    best_params = result.x
    best_FFR, best_PFR, best_aFFR, best_mFFR, best_H = best_params
    best_max_fd, best_max_r, best_results_df = run_simulation(best_FFR, best_PFR, best_aFFR, best_mFFR, best_H, verbose=True)

    freq_steady_state = abs(best_results_df['Frequency Deviation [Hz]'].iloc[-1])

    print("\n--- RESULTADOS FINALES ---")
    print(f"FFR={best_FFR:.2f} MW, PFR={best_PFR:.2f} MW, aFFR={best_aFFR:.2f} MW, mFFR={best_mFFR:.2f} MW, H={best_H:.2f} MWs")
    print(f"Max freq deviation: {best_max_fd:.4f} Hz (límite {limit_freq})")
    print(f"Max ROCOF: {best_max_r:.4f} Hz/s (límite {limit_rocof})")
    print(f"Freq steady state deviation: {freq_steady_state:.4f} Hz (límite {limit_freq_steady_state})")

    return best_params, best_results_df


# ============================
# EJECUTAR OPTIMIZACIÓN
# ============================
if __name__ == "__main__":
    best_params, best_df = optimize_search()
    
    # Graficar resultados de la mejor solución
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(best_df['Time [s]'], best_df['Frequency Deviation [Hz]'], 'k', lw=2, label='Frequency deviation [Hz]')
    ax1.plot(best_df['Time [s]'], best_df['RoCoF [Hz/s]'], label='RoCoF [Hz/s]')
    ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Frequency deviation [Hz]')
    
    ax2 = ax1.twinx()
    ax2.plot(best_df['Time [s]'], best_df['FFR [MW]'], label='FFR')
    ax2.fill_between(best_df['Time [s]'], best_df['FFR [MW]'], 0, where=best_df['FFR [MW]'] > 0, color='gold', alpha=0.4)
    ax2.plot(best_df['Time [s]'], best_df['PFR [MW]'], label='PFR')
    ax2.fill_between(best_df['Time [s]'], best_df['PFR [MW]'], 0, where=best_df['PFR [MW]'] > 0, color='blue', alpha=0.3)
    ax2.plot(best_df['Time [s]'], best_df['aFFR [MW]'], label='aFFR')
    ax2.fill_between(best_df['Time [s]'], best_df['aFFR [MW]'], 0, where=best_df['aFFR [MW]'] > 0, color='green', alpha=0.3)
    ax2.plot(best_df['Time [s]'], best_df['mFFR [MW]'], label='mFFR')
    ax2.fill_between(best_df['Time [s]'], best_df['mFFR [MW]'], 0, where=best_df['mFFR [MW]'] > 0, color='red', alpha=0.2)
    
    ax2.plot(best_df['Time [s]'], best_df['Contingency [MW]'], 'k--', label='Contingency')

    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    # ax1.set_ylim(-3, 3)
    # ax2.set_ylim(-610, 610)
    plt.xlabel('Time [s]')
    plt.ylabel('Valor')
    plt.title('Optimized solution time series')
    plt.legend()
    plt.grid(True)
    plt.show()




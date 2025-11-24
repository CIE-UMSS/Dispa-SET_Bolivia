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
D = 75.0                     # amortiguamiento simple MW/Hz. Normalmente es 1.5% de la demanda o carga


# PERFIL DE TIEMPO (igual que tu original)
t = np.arange(0, 2000, 0.01)


# ============================
# TIEMPOS DE ACTIVACIÓN (basados en TSO)
# ============================
t_ffr_start = 1
t_ffr_ramp_end = 2
t_ffr_hold_end = 10
t_ffr_ramp_down_end = 15

t_pfr_start = 5
t_pfr_ramp_end = 15
t_pfr_hold_end = 45
t_pfr_ramp_down_end = 150

t_affr_start = 30
t_affr_ramp_end = 150
t_affr_hold_end = 450
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
def run_simulation(H_val, FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, 
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
# OPTIMIZACIÓN CON BÚSQUEDA BINARIA/EXHAUSTIVA
# ============================
def optimize_search(limit_freq=0.8, limit_rocof=0.25, limit_freq_steady_state=0.2):
    """
    Búsqueda binaria secuencial de H, FFR, PFR, aFFR y mFFR,
    considerando las ventanas de tiempo específicas de cada variable.
    """
    # Rango y pasos
    H_range = (1, 51)
    H_step = 1
    reserve_range = (0, Contingency_val*1.3)
    reserve_step = Contingency_val*0.01

    best_solution = None
    best_df = None

    print("Iniciando búsqueda secuencial por ventanas de tiempo...")

    # --- Paso 1: buscar H ---
    H_candidates = []
    low, high = H_range
    while low <= high:
        mid = (low + high)/2
        max_fd, max_r, results_df = run_simulation(mid, 0, 0, 0, 0)
        # Ventana de tiempo: 1 → 2 s
        freq_window = results_df.loc[results_df['Time [s]'] < 2.01]
        # Calcular máximo absoluto de cada columna
        max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
        max_r = freq_window['RoCoF [Hz/s]'].abs().max()
        if (max_fd <= limit_freq) and (max_r <= limit_rocof):
            H_candidates.append(mid)
            high = mid - H_step  # buscamos mínimo
        else:
            low = mid + H_step
    
    if not H_candidates:
        print("No se encontró ningún H válido.")
        return None, None

    print(f"H válidos: {H_candidates}")

    # --- Paso 2: buscar FFR ---
    FFR_candidates = []
    for H_val in H_candidates:
        low, high = reserve_range
        while low <= high:
            mid = (low + high)/2
            max_fd, max_r, results_df = run_simulation(H_val, mid, 0, 0, 0)
            # Ventana de tiempo FFR
            freq_window = results_df.loc[results_df['Time [s]'] < t_ffr_hold_end + 0.01]
            # Calcular máximo absoluto de cada columna
            max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
            max_r = freq_window['RoCoF [Hz/s]'].abs().max()
            # # Filtrar por tiempos mayores a 60 s
            # mask = freq_window['Time [s]'] > 60
            # freq_after_60 = freq_window.loc[mask, 'Frequency Deviation [Hz]'].abs()

            # # Calcular el valor máximo absoluto después de 60 s
            # freq_steady_state = freq_after_60.max()
            if (max_fd <= limit_freq) and (max_r <= limit_rocof): #and (freq_steady_state <= limit_freq_steady_state)
                FFR_candidates.append((H_val, mid))
                high = mid - reserve_step  # buscamos mínimo
            else:
                low = mid + reserve_step

    if not FFR_candidates:
        print("No se encontró ninguna combinación H+FFR válida.")
        return None, None

    print(f"Combinaciones H+FFR válidas: {FFR_candidates}")

    # --- Paso 3: PFR ---
    PFR_candidates = []
    for H_val, FFR_val in FFR_candidates:
        low, high = reserve_range
        while low <= high:
            mid = (low + high)/2
            max_fd, max_r, results_df = run_simulation(H_val, FFR_val, mid, 0, 0)
            # Ventana de tiempo PFR
            freq_window = results_df.loc[results_df['Time [s]'] < t_pfr_hold_end + 0.01]
            # Calcular máximo absoluto de cada columna
            max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
            max_r = freq_window['RoCoF [Hz/s]'].abs().max()
            # # Filtrar por tiempos mayores a 60 s
            # mask = freq_window['Time [s]'] > 60
            # freq_after_60 = freq_window.loc[mask, 'Frequency Deviation [Hz]'].abs()

            # # Calcular el valor máximo absoluto después de 60 s
            # freq_steady_state = freq_after_60.max()
            if (max_fd <= limit_freq) and (max_r <= limit_rocof):  #and (freq_steady_state <= limit_freq_steady_state)
                PFR_candidates.append((H_val, FFR_val, mid))
                high = mid - reserve_step
            else:
                low = mid + reserve_step

    if not PFR_candidates:
        print("No se encontró ninguna combinación H+FFR+PFR válida.")
        return None, None
    
    print(f"Combinaciones H+FFR+PFR válidas: {PFR_candidates}")

    # --- Paso 4: aFFR ---
    aFFR_candidates = []
    for H_val, FFR_val, PFR_val in PFR_candidates:
        low, high = reserve_range
        while low <= high:
            mid = (low + high)/2
            max_fd, max_r, results_df = run_simulation(H_val, FFR_val, PFR_val, mid, 0)
            # Ventana de tiempo PFR
            freq_window = results_df.loc[results_df['Time [s]'] < t_affr_hold_end + 0.01]
            # Calcular máximo absoluto de cada columna
            max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
            max_r = freq_window['RoCoF [Hz/s]'].abs().max()
            # Filtrar por tiempos mayores a 60 s
            mask = freq_window['Time [s]'] > 300
            freq_after_60 = freq_window.loc[mask, 'Frequency Deviation [Hz]'].abs()

            # Calcular el valor máximo absoluto después de 60 s
            freq_steady_state = freq_after_60.max()
            if (max_fd <= limit_freq) and (max_r <= limit_rocof) and (freq_steady_state <= limit_freq_steady_state):
                aFFR_candidates.append((H_val, FFR_val, PFR_val, mid))
                high = mid - reserve_step
            else:
                low = mid + reserve_step

    if not aFFR_candidates:
        print("No se encontró ninguna combinación H+FFR+PFR+aFFR válida.")
        return None, None
        
    print(f"Combinaciones H+FFR+PFR+aFFR válidas: {aFFR_candidates}")

    # --- Paso 5: mFFR ---
    mFFR_candidates = []
    for H_val, FFR_val, PFR_val, aFFR_val in aFFR_candidates:
        low, high = reserve_range
        while low <= high:
            mid = (low + high)/2
            max_fd, max_r, results_df = run_simulation(H_val, FFR_val, PFR_val, aFFR_val, mid)
            # Ventana de tiempo PFR
            freq_window = results_df.loc[results_df['Time [s]'] < t_mffr_ramp_end]
            # Calcular máximo absoluto de cada columna
            max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
            max_r = freq_window['RoCoF [Hz/s]'].abs().max()
            # Filtrar por tiempos mayores a 60 s
            mask = freq_window['Time [s]'] > 300
            freq_after_60 = freq_window.loc[mask, 'Frequency Deviation [Hz]'].abs()

            # Calcular el valor máximo absoluto después de 60 s
            freq_steady_state = freq_after_60.max()
            if (max_fd <= limit_freq) and (max_r <= limit_rocof) and (freq_steady_state <= limit_freq_steady_state):
                mFFR_candidates.append((H_val, FFR_val, PFR_val, aFFR_val, mid))
                high = mid - reserve_step
            else:
                low = mid + reserve_step

    if not mFFR_candidates:
        print("No se encontró ninguna combinación válida final.")
        return None, None
    
    print(f"Combinaciones H+FFR+PFR+aFFR+mFFR válidas: {mFFR_candidates}")

    # Tomamos la combinación con los valores mínimos
    best_solution = min(mFFR_candidates, key=lambda x: sum(x[1:]))  # mínima suma de reservas
    H_val, FFR_val, PFR_val, aFFR_val, mFFR_val = best_solution
    _, _, best_df = run_simulation(H_val, FFR_val, PFR_val, aFFR_val, mFFR_val)

    print(f"Combinación óptima encontrada: H={H_val}, FFR={FFR_val}, PFR={PFR_val}, aFFR={aFFR_val}, mFFR={mFFR_val}")
    return best_solution, best_df


# ============================
# EJECUTAR OPTIMIZACIÓN
# ============================
if __name__ == "__main__":
    best_solution, best_df = optimize_search()
    
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
    ax1.set_ylim(-1, 1)
    ax2.set_ylim(-610, 610)
    plt.xlabel('Time [s]')
    plt.ylabel('Valor')
    plt.title('Optimized solution time series')
    plt.legend()
    plt.grid(True)
    plt.show()




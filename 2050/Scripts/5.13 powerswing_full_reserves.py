# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:22:21 2025

@author: navia
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# ============================
# PARÁMETROS FIJOS (puedes ajustar)
# ============================
Contingency_val = 400.0     # MW
D = 75.0                     # amortiguamiento simple MW/Hz. Normalmente es 1.5% de la demanda o carga

# ============================
# TIEMPOS DE ACTIVACIÓN (basados en TSO)
# ============================
t_ffr_preparation = 2
t_ffr_ramping = 3
t_ffr_delivery = 18
t_ffr_deactivation = 21

t_pfr_preparation = 6
t_pfr_ramping = 21
t_pfr_delivery = 51
t_pfr_deactivation = 181

t_afrr_preparation = 31
t_afrr_ramping = 181
t_afrr_delivery = 481
t_afrr_deactivation = 931

t_mfrr_preparation = 481
t_mfrr_ramping = 931

# ============================
# FUNCIÓN DE WEIGHTS 
# ============================
def compute_weights(tt):
    """Devuelve w_ffr, w_pfr, w_afrr, w_mfrr para vector tt (igual que tu implementación)."""
    w_ffr = np.piecewise(tt,
        [tt < t_ffr_preparation,
         (t_ffr_preparation <= tt) & (tt < t_ffr_ramping),
         (t_ffr_ramping <= tt) & (tt < t_ffr_delivery),
         (t_ffr_delivery <= tt) & (tt < t_ffr_deactivation),
         tt >= t_ffr_deactivation],
        [0,
         lambda t: (t - t_ffr_preparation) / (t_ffr_ramping - t_ffr_preparation),
         1,
         lambda t: 1 - (t - t_ffr_delivery) / (t_ffr_deactivation - t_ffr_delivery),
         0]
    )

    w_pfr = np.piecewise(tt,
        [tt < t_pfr_preparation,
         (t_pfr_preparation <= tt) & (tt < t_pfr_ramping),
         (t_pfr_ramping <= tt) & (tt < t_pfr_delivery),
         (t_pfr_delivery <= tt) & (tt < t_pfr_deactivation),
         tt >= t_pfr_deactivation],
        [0,
         lambda t: (t - t_pfr_preparation) / (t_pfr_ramping - t_pfr_preparation),
         1,
         lambda t: 1 - (t - t_pfr_delivery) / (t_pfr_deactivation - t_pfr_delivery),
         0]
    )

    w_afrr = np.piecewise(tt,
        [tt < t_afrr_preparation,
         (t_afrr_preparation <= tt) & (tt < t_afrr_ramping),
         (t_afrr_ramping <= tt) & (tt < t_afrr_delivery),
         (t_afrr_delivery <= tt) & (tt < t_afrr_deactivation),
         tt >= t_afrr_deactivation],
        [0,
         lambda t: (t - t_afrr_preparation) / (t_afrr_ramping - t_afrr_preparation),
         1,
         lambda t: 1 - (t - t_afrr_delivery) / (t_afrr_deactivation - t_afrr_delivery),
         0]
    )

    w_mfrr = np.piecewise(tt,
        [tt < t_mfrr_preparation,
         (t_mfrr_preparation <= tt) & (tt < t_mfrr_ramping),
         tt >= t_mfrr_ramping],
        [0,
         lambda t: (t - t_mfrr_preparation) / (t_mfrr_ramping - t_mfrr_preparation),
         1]
    )

    return w_ffr, w_pfr, w_afrr, w_mfrr


# ============================
# FUNCIÓN DE SIMULACIÓN
# ============================
def run_simulation(sim_time, H_val, FFR_cap, PFR_cap, aFRR_cap, mFRR_cap, 
                   contingency=Contingency_val, D_local=D, verbose=False):
    """
    Ejecuta simulación y devuelve métricas:
    - max_freq_dev (Hz, valor absoluto máximo de desviación)
    - max_rocof (Hz/s, valor absoluto máximo)
    - results_df (DataFrame con series temporales)
    """
    
    # PERFIL DE TIEMPO (igual que tu original)
    t = np.arange(0, sim_time, 0.1)

    # Precompute weights vector for plotting and for fixed deployment logic
    W_ffr_vec, W_pfr_vec, W_afrr_vec, W_mfrr_vec = compute_weights(t)

    def state(y, tt):
        f = y[0]
        contingency_local = contingency if tt >= 1.0 else 0.0

        # pesos en instante tt (scalar)
        w_ffr, w_pfr, w_afrr, w_mfrr = compute_weights(np.array([tt]))
        w_ffr = float(w_ffr[0]); w_pfr = float(w_pfr[0]); w_afrr = float(w_afrr[0]); w_mfrr = float(w_mfrr[0])

        # reservas desplegadas
        ffr = FFR_cap * w_ffr * f
        pfr = PFR_cap * w_pfr * f
        afrr = aFRR_cap * w_afrr
        mfrr = mFRR_cap * w_mfrr

        deltap = contingency_local - (ffr + pfr + afrr + mfrr) - D_local * f
        dfdt = deltap / (1000.0 * (2.0 * H_val / 50.0))
        return [dfdt, deltap]

    y0 = [0.0, 0.0]
    sol = odeint(state, y0, t)
    f = sol[:, 0]


    ffr = FFR_cap * W_ffr_vec
    pfr = PFR_cap * W_pfr_vec
    afrr = aFRR_cap * W_afrr_vec
    mfrr= mFRR_cap * W_mfrr_vec


    contingency_vec = np.where(t >= 1.0, contingency, 0.0)
    deltap = contingency_vec - (ffr + pfr + afrr + mfrr) - D_local * f

    max_freq_dev = np.max(np.abs(f))
    rocof = -np.gradient(f, t)
    max_rocof = np.max(np.abs(rocof))

    results_df = pd.DataFrame({
        'Time [s]': t,
        'Frequency Deviation [Hz]': -f,
        'RoCoF [Hz/s]': rocof,
        'FFR [MW]': ffr,
        'PFR [MW]': pfr,
        'aFRR [MW]': afrr,
        'mFRR [MW]': mfrr,
        'Damping [MW/Hz]': D_local,
        'Contingency [MW]': contingency_vec,
        'deltap [MW]': -deltap
    })

    if verbose:
        print(f"Sim finished: FFR={FFR_cap},PFR={PFR_cap},aFRR={aFRR_cap},mFRR={mFRR_cap},H={H_val}")
        print(f" -> max_freq_dev={max_freq_dev:.4f} Hz, max_rocof={max_rocof:.4f} Hz/s")

    return max_freq_dev, max_rocof, results_df

# ============================
# OPTIMIZACIÓN CON BÚSQUEDA BINARIA/EXHAUSTIVA
# ============================
def optimize_search(limit_freq=0.8, limit_rocof=0.25, limit_freq_steady_state=0.2):
    """
    Búsqueda binaria secuencial de H, FFR, PFR, aFRR y mFRR,
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
        t = 2.01
        mid = (low + high)/2
        max_fd, max_r, results_df = run_simulation(t, mid, 0, 0, 0, 0)
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
            t = t_ffr_deactivation + 0.01
            mid = (low + high)/2
            max_fd, max_r, results_df = run_simulation(t, H_val, mid, 0, 0, 0)
            # Ventana de tiempo FFR
            freq_window = results_df.loc[results_df['Time [s]'] < t_ffr_delivery + 0.01]
            # Calcular máximo absoluto de cada columna
            max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
            max_r = freq_window['RoCoF [Hz/s]'].abs().max()
            # # Filtrar por tiempos mayores a 60 s
            # mask = freq_window['Time [s]'] > 60
            # freq_after_60 = freq_window.loc[mask, 'Frequency Deviation [Hz]'].abs().max()

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
            t = t_pfr_deactivation + 0.01
            mid = (low + high)/2
            max_fd, max_r, results_df = run_simulation(t, H_val, FFR_val, mid, 0, 0)
            # Ventana de tiempo PFR
            freq_window = results_df.loc[results_df['Time [s]'] < t_pfr_delivery + 0.01]
            # Calcular máximo absoluto de cada columna
            max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
            max_r = freq_window['RoCoF [Hz/s]'].abs().max()
            # # Filtrar por tiempos mayores a 60 s
            # mask = freq_window['Time [s]'] > 60
            # freq_after_60 = freq_window.loc[mask, 'Frequency Deviation [Hz]'].abs()

            # # Calcular el valor máximo absoluto después de 60 s
            # freq_steady_state = freq_after_60.max()
            if (max_fd <= limit_freq) and (max_r <= limit_rocof): #and (freq_steady_state <= limit_freq_steady_state)
                PFR_candidates.append((H_val, FFR_val, mid))
                high = mid - reserve_step
            else:
                low = mid + reserve_step

    if not PFR_candidates:
        print("No se encontró ninguna combinación H+FFR+PFR válida.")
        return None, None
    
    print(f"Combinaciones H+FFR+PFR válidas: {PFR_candidates}")
    

    # --- Paso 4: PFR ---
    all_candidates = []
    for H_val, FFR_val, PFR_val in PFR_candidates:
        t = t_mfrr_ramping + 50
        max_fd, max_r, results_df = run_simulation(t, H_val, FFR_val, PFR_val, Contingency_val, Contingency_val)
        # Ventana de tiempo PFR
        freq_window = results_df.loc[results_df['Time [s]'] < t_pfr_delivery + 0.01]
        # Calcular máximo absoluto de cada columna
        max_fd = freq_window['Frequency Deviation [Hz]'].abs().max()
        max_r = freq_window['RoCoF [Hz/s]'].abs().max()
        # # Filtrar por tiempos mayores a 60 s
        # mask = freq_window['Time [s]'] > 60
        # freq_after_60 = freq_window.loc[mask, 'Frequency Deviation [Hz]'].abs()

        # # Calcular el valor máximo absoluto después de 60 s
        # freq_steady_state = freq_after_60.max()
        if (max_fd <= limit_freq) and (max_r <= limit_rocof): #and (freq_steady_state <= limit_freq_steady_state)
            all_candidates.append((H_val, FFR_val, PFR_val, Contingency_val, Contingency_val))

    if not all_candidates:
        print("No se encontró ninguna combinación H+FFR+PFR válida.")
        return None, None
    
    print(f"Combinaciones H+FFR+PFR+aFRR+mFRR válidas: {all_candidates}")

    # Tomamos la combinación con los valores mínimos
    best_solution = min(all_candidates, key=lambda x: sum(x[1:]))  # mínima suma de reservas
    H_val, FFR_val, PFR_val, aFRR_val, mFRR_val = best_solution
    _, _, best_df = run_simulation(t, H_val, FFR_val, PFR_val, aFRR_val, mFRR_val)

    print(f"Combinación óptima encontrada: H={H_val}, FFR={FFR_val}, PFR={PFR_val}, aFRR={aFRR_val}, mFRR={mFRR_val}")
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
    ax2.plot(best_df['Time [s]'], best_df['aFRR [MW]'], label='aFRR')
    ax2.fill_between(best_df['Time [s]'], best_df['aFRR [MW]'], 0, where=best_df['aFRR [MW]'] > 0, color='green', alpha=0.3)
    ax2.plot(best_df['Time [s]'], best_df['mFRR [MW]'], label='mFRR')
    ax2.fill_between(best_df['Time [s]'], best_df['mFRR [MW]'], 0, where=best_df['mFRR [MW]'] > 0, color='red', alpha=0.2)
    
    ax2.plot(best_df['Time [s]'], best_df['Contingency [MW]'], 'k--', label='Contingency')

    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax1.set_ylim(-1, 1)
    ax2.set_ylim(-500, 500)
    plt.xlabel('Time [s]')
    plt.ylabel('Valor')
    plt.title('Optimized solution time series')
    plt.legend()
    plt.grid(True)
    plt.show()




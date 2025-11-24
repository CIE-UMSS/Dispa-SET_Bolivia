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
Contingency_val = 194.63     # MW
D = 13.76                     # amortiguamiento simple MW/Hz. Normalmente es 1.5% de la demanda o carga

# ============================
# TIEMPOS DE ACTIVACIÓN (basados en TSO)
# ============================

activation_times = {
            "ffr": dict(prep=2, ramp=3, delivery=61, deact=301),
            "pfr": dict(prep=5, ramp=16, delivery=181, deact=301),
            "afrr": dict(prep=31, ramp=301, delivery=481, deact=901),
            "mfrr": dict(prep=481, ramp=901)  # mfrr leght is considered for the whole timestep
        }
# ============================
# FUNCIÓN DE WEIGHTS 
# ============================
def trapezoid_weight(tt, prep, ramp, delivery=None, deact=None):
    """
    Construye un perfil trapezoidal (0→1→0) o triangular (0→1).
    """
    if delivery is None or deact is None:
        # Caso triangular: solo subida (ej. mFRR)
        return np.piecewise(tt,
            [tt < prep,
             (prep <= tt) & (tt < ramp),
             tt >= ramp],
            [0,
             lambda t: (t - prep) / (ramp - prep),
             1]
        )
    else:
        # Caso trapezoidal: subida → meseta → bajada
        return np.piecewise(tt,
            [tt < prep,
             (prep <= tt) & (tt < ramp),
             (ramp <= tt) & (tt < delivery),
             (delivery <= tt) & (tt < deact),
             tt >= deact],
            [0,
             lambda t: (t - prep) / (ramp - prep),
             1,
             lambda t: 1 - (t - delivery) / (deact - delivery),
             0]
        )

def compute_weights(tt):
    """
    Devuelve w_ffr, w_pfr, w_afrr, w_mfrr para vector tt.
    """
    w_ffr = trapezoid_weight(tt, **activation_times["ffr"])
    w_pfr = trapezoid_weight(tt, **activation_times["pfr"])
    w_afrr = trapezoid_weight(tt, **activation_times["afrr"])
    w_mfrr = trapezoid_weight(tt, **activation_times["mfrr"])
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
    # Manejo de H=0
    if H_val == 0:
        max_freq_dev = np.inf
        max_rocof = np.inf
        results_df = pd.DataFrame({
            'Time [s]': [],
            'Frequency Deviation [Hz]': [],
            'RoCoF [Hz/s]': [],
            'Inertia [GWs]': [],
            'FFR [MW]': [],
            'PFR [MW]': [],
            'aFRR [MW]': [],
            'mFRR [MW]': [],
            'Damping [MW/Hz]': [],
            'Contingency [MW]': [],
            'deltap [MW]': []
        })
        if verbose:
            print("H_val=0: sistema inestable, retornando infinidades.")
        return max_freq_dev, max_rocof, results_df

    
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
        'Inertia [GWs]': H_val,
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
def optimize_search(limit_freq=0.8, limit_rocof=0.5, limit_freq_steady_state=0.2,
                    use_ffr=True, use_pfr=True, use_afrr=True, use_mfrr=True):
    """
    Búsqueda binaria secuencial de H, FFR, PFR, aFRR y mFRR,
    considerando las ventanas de tiempo específicas de cada variable.
    """
    # Rango y pasos
    H_range = (1, 70)
    H_step = 1
    reserve_range = (0, Contingency_val*3)
    reserve_step = Contingency_val*0.01

    best_solution = None
    best_df = None

    print("Iniciando búsqueda secuencial por ventanas de tiempo...")

    # --- Paso 1: buscar H ---
    H_candidates = []
    low, high = H_range
    while low <= high:
        if use_ffr:
            t = activation_times["ffr"]["prep"]
        elif use_pfr:
            t = activation_times["pfr"]["prep"]
        
        mid = (low + high)/2
        max_fd, max_r, results_df = run_simulation(t, mid, 0, 0, 0, 0)

        if (max_fd <= limit_freq) and (max_r <= limit_rocof):
            H_candidates.append(mid)
            high = mid - H_step  # buscamos mínimo
        else:
            low = mid + H_step
    
    if not H_candidates:
        print("No se encontró ningún H válido.")
        H_candidates = [(0)]
    else:
        print(f"H válidos: {H_candidates}")

    # --- Paso 2: buscar FFR ---
    FFR_candidates = []
    if use_ffr:
        for H_val in H_candidates:
            low, high = reserve_range
            while low <= high:
                t = activation_times["ffr"]["delivery"]
                mid = (low + high)/2
                max_fd, max_r, results_df = run_simulation(t, H_val, mid, 0, 0, 0)

                if (max_fd <= limit_freq) and (max_r <= limit_rocof): 
                    FFR_candidates.append((H_val, mid))
                    high = mid - reserve_step  # buscamos mínimo
                else:
                    low = mid + reserve_step
    
        if not FFR_candidates:
            print("No se encontró ninguna combinación H+FFR válida.")
            FFR_candidates = [(H_val, 0) for H_val in H_candidates]
        else:
            print(f"Combinaciones H+FFR válidas: {FFR_candidates}")
    else:
        print("Reservas FFR inactivas")
        FFR_candidates = [(H_val, 0) for H_val in H_candidates]

    # --- Paso 3: PFR ---
    PFR_candidates = []
    if use_pfr:
        for H_val, FFR_val in FFR_candidates:
            low, high = reserve_range
            while low <= high:
                t = activation_times["pfr"]["delivery"]
                mid = (low + high)/2
                max_fd, max_r, results_df = run_simulation(t, H_val, FFR_val, mid, 0, 0)
                
                if (max_fd <= limit_freq) and (max_r <= limit_rocof): 
                    PFR_candidates.append((H_val, FFR_val, mid))
                    high = mid - reserve_step
                else:
                    low = mid + reserve_step
    
        if not PFR_candidates:
            print("No se encontró ninguna combinación H+FFR+PFR válida.")
            PFR_candidates = [(H_val, FFR_val, 0) for H_val, FFR_val in FFR_candidates]  
        else:    
            print(f"Combinaciones H+FFR+PFR válidas: {PFR_candidates}")
    else:
        print("Reservas PFR inactivas")
        PFR_candidates = [(H_val, FFR_val, 0) for H_val, FFR_val in FFR_candidates]  
    
    # --- Paso 4: all ---
    all_candidates = []
    if use_afrr and use_mfrr:
        for H_val, FFR_val, PFR_val in PFR_candidates:
            t = activation_times["mfrr"]["ramp"] + 50
            max_fd, max_r, results_df = run_simulation(t, H_val, FFR_val, PFR_val, Contingency_val, Contingency_val)
            # Ventana de tiempo frequency estado estacionario
            freq_window = results_df.loc[results_df['Time [s]'] > 300]
            # Calcular máximo absoluto de cada columna
            freq_steady_state = freq_window['Frequency Deviation [Hz]'].abs().max()

            if (max_fd <= limit_freq) and (max_r <= limit_rocof) and (freq_steady_state <= limit_freq_steady_state):
                all_candidates.append((H_val*1000, FFR_val, PFR_val, Contingency_val, Contingency_val))
    
        if not all_candidates:
            print("No se encontró ninguna combinación H+FFR+PFR+aFRR+mFRR válida.")
            all_candidates = [(H_val*1000, FFR_val, PFR_val, 0, 0) for H_val, FFR_val, PFR_val in PFR_candidates]   
        else:    
            print(f"Combinaciones +FFR+PFR+aFRR+mFRR válidas: {all_candidates}")
    else:
        print("Reservas aFRR y mFRR inactivas")
        all_candidates = [(H_val*1000, FFR_val, PFR_val, 0, 0) for H_val, FFR_val, PFR_val in PFR_candidates]
        
    # Tomamos la combinación con los valores mínimos
    best_solution = min(all_candidates, key=lambda x: sum(x[:]))  # mínima suma de reservas
    H_val, FFR_val, PFR_val, aFRR_val, mFRR_val = best_solution
    _, _, best_df = run_simulation(t, H_val/1000, FFR_val, PFR_val, aFRR_val, mFRR_val)

    print(f"Combinación óptima encontrada: H={H_val/1000}, FFR={FFR_val}, PFR={PFR_val}, aFRR={aFRR_val}, mFRR={mFRR_val}")
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

#%%

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot de frecuencia y RoCoF en el eje izquierdo
    ax1.plot(best_df['Time [s]'], best_df['Frequency Deviation [Hz]'], 'k', lw=2, label='Frequency deviation [Hz]')
    ax1.plot(best_df['Time [s]'], best_df['RoCoF [Hz/s]'], label='RoCoF [Hz/s]')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency deviation [Hz]')
    ax1.set_ylim(-1, 1)
    
    # Segundo eje para reservas
    ax2 = ax1.twinx()
    
    # Datos de reservas
    time = best_df['Time [s]']
    ffr = best_df['FFR [MW]']
    pfr = best_df['PFR [MW]']
    afrr = best_df['aFRR [MW]']
    mfrr = best_df['mFRR [MW]']
    
    # Stackplot en vez de fill_between
    ax2.stackplot(time, ffr, pfr, afrr, mfrr, 
                  labels=['FFR', 'PFR', 'aFRR', 'mFRR'],
                  colors=['gold', 'blue', 'green', 'red'], alpha=0.3)
    
    # Contingencia como línea
    ax2.plot(time, best_df['Contingency [MW]'], 'k--', label='Contingency')
    
    ax2.set_ylabel('Reserves [MW]')
    ax2.set_ylim(-1000, 1000)
    
    # Leyendas
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Optimized solution time series')
    plt.grid(True)
    plt.show()



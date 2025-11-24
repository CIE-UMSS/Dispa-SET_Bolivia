# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:22:21 2025

@author: navia
"""
#%%

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
kffr = 300                # FFR MW/Hz
kpfr = 400                # PFR MW/Hz
kaffr = 500               # aFFR MW/Hz
kmffr = 900               # mFFR MW/Hz
H = 50                   # MW·s/MVA

# ============================
# TIEMPO CON RESOLUCIÓN VARIABLE
# ============================
t1 = np.arange(0, 60, 0.1)        # cada 1 segundo hasta 60 s
t2 = np.arange(60, 1200, 0.1)     # cada 10 segundos de 60 a 900 s
t = np.unique(np.concatenate((t1, t2)))

# ============================
# FUNCIONES DE ACTIVACIÓN
# ============================
def compute_weights(t):
    w_ffr = np.piecewise(t, 
        [t < 2, (2 <= t) & (t < 5), (5 <= t) & (t < 10), (10 <= t) & (t < 15), t >= 15],
        [0, lambda t: (t - 2)/3, 1, lambda t: 1 - (t - 10)/5, 0]
    )

    w_pfr = np.piecewise(t,
        [t < 5, (5 <= t) & (t < 15), (15 <= t) & (t < 35), (35 <= t) & (t < 905), t >= 905],
        [0, lambda t: (t - 5)/10, 1, lambda t: 1 - (t - 35)/870, 0]
    )

    w_affr = np.piecewise(t,
        [t < 35, (35 <= t) & (t < 255), (255 <= t) & (t < 450), (450 <= t) & (t < 905), t >= 905],
        [0, lambda t: (t - 35)/220, 1, lambda t: 1 - (t - 450)/455, 0]
    )

    w_mffr = np.piecewise(t,
        [t < 480, (480 <= t) & (t < 905), t >= 905],
        [0, lambda t: (t - 480)/425, 1]
    )

    return w_ffr, w_pfr, w_affr, w_mffr

# ============================
# ECUACIÓN DIFERENCIAL
# ============================
def power_swing(f, t, H, kffr, kpfr, kaffr, kmffr):
    contingency = Contingency_val if t >= 1 else 0

    # Pesos (activaciones)
    w_ffr, w_pfr, w_affr, w_mffr = compute_weights(np.array([t]))
    ffr = kffr * f * w_ffr[0]
    pfr = kpfr * f * w_pfr[0]
    affr = kaffr * f * w_affr[0]
    mffr = kmffr * f * w_mffr[0]

    deltap = contingency - ffr - pfr - affr - mffr
    delf = deltap / (1000 * (2 * H / 50))
    return delf

# ============================
# SIMULACIÓN
# ============================
f0 = 0
f = odeint(power_swing, f0, t, args=(H, kffr, kpfr, kaffr, kmffr)).flatten()
rocof = -np.gradient(f, t)

# ============================
# CÁLCULO DE RESERVAS Y DELTAP
# ============================
w_ffr, w_pfr, w_affr, w_mffr = compute_weights(t)
ffr = kffr * f * w_ffr
pfr = kpfr * f * w_pfr
affr = kaffr * f * w_affr
mffr = kmffr * f * w_mffr
contingency = np.where(t >= 1, Contingency_val, 0)

deltap = contingency - ffr - pfr - affr - mffr

# ============================
# RESULTADOS EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'RoCoF [Hz/s]': rocof,
    '∆P [MW]': -deltap,
    'FFR [MW]': ffr,
    'PFR [MW]': pfr,
    'aFFR [MW]': affr,
    'mFFR [MW]': mffr,
    'Contingency [MW]': contingency,
    'w_ffr': w_ffr,
    'w_pfr': w_pfr,
    'w_affr': w_affr,
    'w_mffr': w_mffr
})

# ============================
# PLOT
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.plot(t, -f, 'black', lw=2, label='Frequency deviation [Hz]')
ax1.plot(t, rocof, 'fuchsia', lw=2, ls='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=14)
ax1.set_ylabel('Frequency / RoCoF', fontsize=14)
ax1.grid(True)
ax1.tick_params(labelsize=12)

ax2 = ax1.twinx()
ax2.plot(t, ffr, color='gold', lw=2, label='FFR [MW]')
ax2.fill_between(t, ffr, 0, where=ffr > 0, color='gold', alpha=0.4)

ax2.plot(t, pfr, color='blue', lw=2, label='PFR [MW]')
ax2.fill_between(t, pfr, 0, where=pfr > 0, color='blue', alpha=0.3)

ax2.plot(t, affr, color='green', lw=2, label='aFFR [MW]')
ax2.fill_between(t, affr, 0, where=affr > 0, color='green', alpha=0.3)

ax2.plot(t, mffr, color='red', lw=2, label='mFFR [MW]')
ax2.fill_between(t, mffr, 0, where=mffr > 0, color='red', alpha=0.2)

ax2.plot(t, -deltap, color='grey', lw=2, ls='-.', label='∆P [MW]')
ax2.plot(t, contingency, 'black', ls=':', lw=2, label='Contingency [MW]')

ax2.set_ylabel('Power [MW]', fontsize=14)
ax2.tick_params(labelsize=12)

# Leyenda
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
ax1.set_ylim(-1.5, 1.5)
ax2.set_ylim(-610, 610)

plt.title('Dynamic Response to Contingency with Weighted Reserves', fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


#%%

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400.0     # MW
kffr = 300.0                # FFR MW/Hz (proporcional)
kpfr = 400.0                # PFR MW/Hz
kaffr = 500.0               # aFFR MW/Hz
kmffr = 900.0               # mFFR (proporcional) MW/Hz
H = 50.0                    # MW·s/MVA
D = 1.0                     # amortiguamiento simple MW/Hz (opcional)
Krr = 1.0                   # ganancia del integrador RR (MW por unidad de I)
Prr_max = Contingency_val   # saturación: RR no entrega más de la contingencia
Prr_min = 0.0               # mínimo (no negativo)

# ============================
# TIEMPO CON RESOLUCIÓN VARIABLE
# ============================
t1 = np.arange(0, 60, 0.1)
t2 = np.arange(60, 1200, 0.1)
t = np.unique(np.concatenate((t1, t2)))

# ============================
# PARÁMETROS DE ACTIVACIÓN
# ============================
# Fast Frequency Response
t_ffr_start = 2       # Inicio activación (s)
t_ffr_ramp_end = 5    # Fin rampa inicial (s)
t_ffr_hold_end = 10   # Fin periodo estable (s)
t_ffr_ramp_down_end = 15  # Fin rampa de desactivación (s)

# Primary Frequency Response
t_pfr_start = 5
t_pfr_ramp_end = 15
t_pfr_hold_end = 35
t_pfr_ramp_down_end = 905

# Automatic FRR
t_affr_start = 35
t_affr_ramp_end = 255
t_affr_hold_end = 450
t_affr_ramp_down_end = 905

# Manual FRR
t_mffr_start = 480
t_mffr_ramp_end = 905

# ============================
# FUNCIONES DE ACTIVACIÓN
# ============================
def compute_weights(t):
    w_ffr = np.piecewise(t, 
        [t < t_ffr_start, 
         (t_ffr_start <= t) & (t < t_ffr_ramp_end), 
         (t_ffr_ramp_end <= t) & (t < t_ffr_hold_end), 
         (t_ffr_hold_end <= t) & (t < t_ffr_ramp_down_end), 
         t >= t_ffr_ramp_down_end],
        [0, 
         lambda t: (t - t_ffr_start) / (t_ffr_ramp_end - t_ffr_start), 
         1, 
         lambda t: 1 - (t - t_ffr_hold_end) / (t_ffr_ramp_down_end - t_ffr_hold_end), 
         0]
    )

    w_pfr = np.piecewise(t,
        [t < t_pfr_start, 
         (t_pfr_start <= t) & (t < t_pfr_ramp_end), 
         (t_pfr_ramp_end <= t) & (t < t_pfr_hold_end), 
         (t_pfr_hold_end <= t) & (t < t_pfr_ramp_down_end), 
         t >= t_pfr_ramp_down_end],
        [0, 
         lambda t: (t - t_pfr_start) / (t_pfr_ramp_end - t_pfr_start), 
         1, 
         lambda t: 1 - (t - t_pfr_hold_end) / (t_pfr_ramp_down_end - t_pfr_hold_end), 
         0]
    )

    w_affr = np.piecewise(t,
        [t < t_affr_start, 
         (t_affr_start <= t) & (t < t_affr_ramp_end), 
         (t_affr_ramp_end <= t) & (t < t_affr_hold_end), 
         (t_affr_hold_end <= t) & (t < t_affr_ramp_down_end), 
         t >= t_affr_ramp_down_end],
        [0, 
         lambda t: (t - t_affr_start) / (t_affr_ramp_end - t_affr_start), 
         1, 
         lambda t: 1 - (t - t_affr_hold_end) / (t_affr_ramp_down_end - t_affr_hold_end), 
         0]
    )

    w_mffr = np.piecewise(t,
        [t < t_mffr_start, 
         (t_mffr_start <= t) & (t < t_mffr_ramp_end), 
         t >= t_mffr_ramp_end],
        [0, 
         lambda t: (t - t_mffr_start) / (t_mffr_ramp_end - t_mffr_start), 
         1]
    )

    return w_ffr, w_pfr, w_affr, w_mffr

# ============================
# SISTEMA DE EDO (estados: f, I_rr)
# ============================
def power_swing_state(y, t, H, kffr, kpfr, kaffr, kmffr, Krr, D):
    f = y[0]      # frecuencia (estado)
    I_rr = y[1]   # integrador de la reserva de restauración

    contingency = Contingency_val if t >= 1 else 0.0

    # Pesos (activaciones) en el instante t (se pasa scalar)
    w_ffr, w_pfr, w_affr, w_mffr = compute_weights(np.array([t]))
    w_ffr = w_ffr[0]; w_pfr = w_pfr[0]; w_affr = w_affr[0]; w_mffr = w_mffr[0]

    # Reservas proporcionales (dependen de f)
    ffr = kffr * f * w_ffr
    pfr = kpfr * f * w_pfr
    affr = kaffr * f * w_affr
    mffr = kmffr * f * w_mffr

    # Reserva de restauración (integral) - activada con w_mffr
    P_rr = Krr * I_rr * w_mffr
    # Saturación anti-windup (limitamos P_rr)
    if P_rr > Prr_max:
        P_rr = Prr_max
    if P_rr < Prr_min:
        P_rr = Prr_min

    # Balance de potencia (nótese el signo: contingencia positiva = pérdida que hay que reemplazar)
    deltap = contingency - (ffr + pfr + affr + mffr + P_rr) - D * f

    # Dinámica de frecuencia (factor 1000*(2H/50) as in your original code)
    dfdt = deltap / (1000.0 * (2.0 * H / 50.0))

    # Dinámica del integrador: integramos la frecuencia (signo positivo: I crece cuando f>0)
    # Con esta convención, si la contingencia produce f>0, I aumenta y P_rr aumenta.
    dIdt = f

    return [dfdt, dIdt]

# ============================
# SIMULACIÓN (estado inicial: f0=0, I0=0)
# ============================
y0 = [0.0, 0.0]
sol = odeint(power_swing_state, y0, t, args=(H, kffr, kpfr, kaffr, kmffr, Krr, D))
f = sol[:, 0]
I_rr = sol[:, 1]
rocof = -np.gradient(f, t)

# ============================
# Recalcular reservas para plotting
# ============================
w_ffr, w_pfr, w_affr, w_mffr = compute_weights(t)
ffr = kffr * f * w_ffr
pfr = kpfr * f * w_pfr
affr = kaffr * f * w_affr
mffr = kmffr * f * w_mffr
P_rr = Krr * I_rr * w_mffr
# saturación
P_rr = np.clip(P_rr, Prr_min, Prr_max)

contingency = np.where(t >= 1, Contingency_val, 0.0)
deltap = contingency - (ffr + pfr + affr + mffr + P_rr) - D * f

# ============================
# DataFrame y plots (igual que antes, con P_rr añadido)
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'I_rr': I_rr,
    'P_rr [MW]': P_rr,
    '∆P [MW]': -deltap,
    'FFR [MW]': ffr,
    'PFR [MW]': pfr,
    'aFFR [MW]': affr,
    'mFFR [MW]': mffr,
    'mFFR_total [MW]': mffr + P_rr,
    'Contingency [MW]': contingency,
    'w_ffr': w_ffr,
    'w_pfr': w_pfr,
    'w_affr': w_affr,
    'w_mffr': w_mffr
})

# # plot ejemplo (añadí P_rr y mFFR_total)
# fig, ax1 = plt.subplots(figsize=(14, 8))
# ax1.plot(t, -f, 'black', lw=2, label='Frequency deviation [Hz]')
# ax1.plot(t, -I_rr * 0.0, 'grey', lw=1)  # placeholder si quieres ver I_rr con escala distinta
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel('Frequency [Hz]')
# ax1.grid(True)

# ax2 = ax1.twinx()
# ax2.plot(t, P_rr, color='magenta', lw=2, label='RR (P_rr) [MW]')
# ax2.plot(t, mffr_prop, color='red', lw=1, ls='--', label='mFFR_prop [MW]')
# ax2.plot(t, mffr_prop + P_rr, color='red', lw=2, label='mFFR_total [MW]')
# ax2.plot(t, contingency, color='black', ls=':', lw=2, label='Contingency [MW]')
# ax2.set_ylabel('Power [MW]')
# ax2.set_ylim(-50, 450)

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
# plt.title('Respuesta dinámica con RR (integrador) + saturación')
# plt.tight_layout()
# plt.show()



# ============================
# PLOT
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.plot(t, -f, 'black', lw=2, label='Frequency deviation [Hz]')
ax1.plot(t, rocof, 'fuchsia', lw=2, ls='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=14)
ax1.set_ylabel('Frequency / RoCoF', fontsize=14)
ax1.grid(True)
ax1.tick_params(labelsize=12)

ax2 = ax1.twinx()
ax2.plot(t, ffr, color='gold', lw=2, label='FFR [MW]')
ax2.fill_between(t, ffr, 0, where=ffr > 0, color='gold', alpha=0.4)

ax2.plot(t, pfr, color='blue', lw=2, label='PFR [MW]')
ax2.fill_between(t, pfr, 0, where=pfr > 0, color='blue', alpha=0.3)

ax2.plot(t, affr, color='green', lw=2, label='aFFR [MW]')
ax2.fill_between(t, affr, 0, where=affr > 0, color='green', alpha=0.3)

ax2.plot(t, mffr, color='red', lw=2, label='mFFR [MW]')
ax2.fill_between(t, mffr, 0, where=mffr > 0, color='red', alpha=0.2)

ax2.plot(t, P_rr, color='magenta', lw=2, label='RR (P_rr) [MW]')
ax2.fill_between(t, P_rr, 0, where=P_rr > 0, color='magenta', alpha=0.2)

ax2.plot(t, mffr + P_rr, color='gray', lw=2, label='mFFR_total [MW]')
ax2.fill_between(t, mffr + P_rr, 0, where=P_rr > 0, color='gray', alpha=0.2)

ax2.plot(t, -deltap, color='grey', lw=2, ls='-.', label='∆P [MW]')
ax2.plot(t, contingency, 'black', ls=':', lw=2, label='Contingency [MW]')

ax2.set_ylabel('Power [MW]', fontsize=14)
ax2.tick_params(labelsize=12)

# Leyenda
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
ax1.set_ylim(-1.5, 1.5)
ax2.set_ylim(-610, 610)

plt.title('Dynamic Response to Contingency with Weighted Reserves', fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


#%%

# Reescribiendo el código original para:
# - tratar las reservas como capacidades fijas desplegadas según los pesos w(t)
# - añadir funciones para ejecutar simulación y obtener métricas
# - implementar: (A) Grid search (barrido) y (B) Optimización (differential_evolution)
# según los límites de frecuencia y ROCOF

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
H_default = 50.0            # valor por defecto (se variará en la búsqueda)
D = 1.0                     # amortiguamiento simple MW/Hz
Krr = 1.0                   # ganancia integrador RR (no cambia)
Prr_min = 0.0               # RR mínimo

# PERFIL DE TIEMPO (igual que tu original)
t1 = np.arange(0, 60, 0.1)
t2 = np.arange(60, 2000, 0.1)
t = np.unique(np.concatenate((t1, t2)))

# ============================
# PARÁMETROS DE ACTIVACIÓN (igual que tu original)
# ============================
# FFR (Fast Frequency Response)
t_ffr_start = 1
t_ffr_ramp_end = 5
t_ffr_hold_end = 20
t_ffr_ramp_down_end = 30

# PFR (Primary Frequency Response)
t_pfr_start = 6
t_pfr_ramp_end = 30
t_pfr_hold_end = 180    # 3 min
t_pfr_ramp_down_end = 900   # 15 min

# AFFR (Automatic Fast Frequency Response)
t_affr_start = 40
t_affr_ramp_end = 300   # 5 min
t_affr_hold_end = 900   # 15 min
t_affr_ramp_down_end = 1200  # 20 min

# MFFR (Manual Fast Frequency Response)
t_mffr_start = 600    # 10 min
t_mffr_ramp_end = 1800  # 30 min

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
# - RESERVAS se modelan como capacidades fijas desplegadas con w(t):
#   ffr = FFR_cap * w_ffr(t)  (MW)
#   pfr = PFR_cap * w_pfr(t)
#   affr = aFFR_cap * w_affr(t)
#   mffr_prop = mFFR_cap * w_mffr(t)   (parte proporcional/instantánea)
# - RR (P_rr) sigue siendo integrador limitado por RR_max.
# ============================
def run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val,
                   contingency=Contingency_val, D_local=D, verbose=False):
    """
    Ejecuta simulación y devuelve métricas:
    - max_freq_dev (Hz, valor absoluto máximo de desviación)
    - max_rocof (Hz/s, valor absoluto máximo)
    - results_df (DataFrame con series temporales)
    """
    # EDO de dos estados: f, I_rr
    def state(y, tt):
        f = y[0]
        I_rr = y[1]
        contingency_local = contingency if tt >= 1.0 else 0.0

        # pesos en instante tt (scalar)
        w_ffr, w_pfr, w_affr, w_mffr = compute_weights(np.array([tt]))
        w_ffr = float(w_ffr[0]); w_pfr = float(w_pfr[0]); w_affr = float(w_affr[0]); w_mffr = float(w_mffr[0])

        # reservas desplegadas (capacidades fijas multiplicadas por pesos)
        ffr = FFR_cap * w_ffr
        pfr = PFR_cap * w_pfr
        affr = aFFR_cap * w_affr
        mffr_prop = mFFR_cap * w_mffr

        # RR desde integrador
        P_rr = Krr * I_rr * w_mffr
        # saturación anti-windup
        if P_rr > RR_max:
            P_rr = RR_max
        if P_rr < Prr_min:
            P_rr = Prr_min

        deltap = contingency_local - (ffr + pfr + affr + mffr_prop + P_rr) - D_local * f
        dfdt = deltap / (1000.0 * (2.0 * H_val / 50.0))
        dIdt = f  # integrador simple (puedes escalar si quieres velocidad diferente)
        return [dfdt, dIdt]

    # integrar
    y0 = [0.0, 0.0]
    sol = odeint(state, y0, t)
    f = sol[:, 0]
    I_rr = sol[:, 1]

    # calcular series auxiliares (vectorizadas)
    ffr = FFR_cap * W_ffr_vec
    pfr = PFR_cap * W_pfr_vec
    affr = aFFR_cap * W_affr_vec
    mffr_prop = mFFR_cap * W_mffr_vec
    P_rr = Krr * I_rr * W_mffr_vec
    P_rr = np.clip(P_rr, Prr_min, RR_max)

    contingency_vec = np.where(t >= 1.0, contingency, 0.0)
    deltap = contingency_vec - (ffr + pfr + affr + mffr_prop + P_rr) - D_local * f

    # métricas
    max_freq_dev = np.max(np.abs(f))           # en Hz (tu f ya es desviación en Hz)
    rocof = -np.gradient(f, t)
    max_rocof = np.max(np.abs(rocof))

    # DataFrame con resultados
    results_df = pd.DataFrame({
        'Time [s]': t,
        'Frequency Deviation [Hz]': -f,
        'RoCoF [Hz/s]': rocof,
        'FFR [MW]': ffr,
        'PFR [MW]': pfr,
        'aFFR [MW]': affr,
        'mFFR_prop [MW]': mffr_prop,
        'P_rr [MW]': P_rr,
        'mFFR_total [MW]': mffr_prop + P_rr,
        'Contingency [MW]': contingency_vec,
        'deltap [MW]': -deltap
    })

    if verbose:
        print(f"Sim finished: FFR={FFR_cap},PFR={PFR_cap},aFFR={aFFR_cap},mFFR={mFFR_cap},RR_max={RR_max},H={H_val}")
        print(f" -> max_freq_dev={max_freq_dev:.4f} Hz, max_rocof={max_rocof:.4f} Hz/s")

    return max_freq_dev, max_rocof, results_df

# ============================
# RANGOS PARA GRID SEARCH (propuesta)
# ============================
H_range = [25.0, 30.0, 50.0]          # inercia (escala usada en denominador)
FFR_range = np.arange(0, 5001, 2500)            # MW
PFR_range = np.arange(0, 5001, 2500)           # MW
aFFR_range = np.arange(0, 5001, 2500)          # MW
mFFR_range = np.arange(0, 5001, 2500)         # MW
RR_range = np.arange(0, 5001, 2500)            # MW (saturación RR)

# límites aceptables (puedes ajustar)
MAX_FREQ_DEV = 0.8  # Hz
MAX_ROCOF = 1.0     # Hz/s

# ============================
# GRID SEARCH (barrido)
# ============================
def grid_search(limit_freq=MAX_FREQ_DEV, limit_rocof=MAX_ROCOF, save_top=50):
    combos_ok = []
    combos_all = []
    tic = time.time()
    total = len(H_range)*len(FFR_range)*len(PFR_range)*len(aFFR_range)*len(mFFR_range)*len(RR_range)
    i = 0
    for H_val, FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max in itertools.product(
            H_range, FFR_range, PFR_range, aFFR_range, mFFR_range, RR_range):
        i += 1
        max_fd, max_r, _ = run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val)
        total_capacity = FFR_cap + PFR_cap + aFFR_cap + mFFR_cap + RR_max
        combos_all.append((H_val, FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, max_fd, max_r, total_capacity))
        if (max_fd <= limit_freq) and (max_r <= limit_rocof):
            combos_ok.append((H_val, FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, max_fd, max_r, total_capacity))
        # progress every 200 iters
        if i % 200 == 0:
            elapsed = time.time() - tic
            print(f"Grid progress: {i}/{total} combos done, elapsed {elapsed:.1f}s, accepted {len(combos_ok)}")
    df_all = pd.DataFrame(combos_all, columns=['H','FFR','PFR','aFFR','mFFR','RR_max','max_freq_dev','max_rocof','total_capacity'])
    df_ok = pd.DataFrame(combos_ok, columns=df_all.columns) if combos_ok else pd.DataFrame(columns=df_all.columns)
    # ordenar soluciones aceptables por menor total_capacity
    df_ok_sorted = df_ok.sort_values('total_capacity').head(save_top)
    return df_all, df_ok_sorted

# ============================
# OPTIMIZACIÓN (differential_evolution)
# - Minimiza suma de capacidades sujeto a penalización cuando viola límites.
# ============================
def optimize_search(limit_freq=MAX_FREQ_DEV, limit_rocof=MAX_ROCOF, maxiter=50, popsize=15):
    # bounds: [FFR, PFR, aFFR, mFFR, RR_max, H]
    bounds = [(0, 500),    # FFR
              (0, 300),    # PFR
              (0, 200),   # aFFR
              (0, 300),   # mFFR
              (0, 100),    # RR_max (no excede contingency in práctica)
              (10, 50)]   # H

    PEN = 1e5  # penalización grande para violaciones

    def objective(x):
        FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val = x
        max_fd, max_r, _ = run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val)
        total_capacity = FFR_cap + PFR_cap + aFFR_cap + mFFR_cap + RR_max
        penalty = 0.0
        if max_fd > limit_freq:
            penalty += PEN * (max_fd - limit_freq)
        if max_r > limit_rocof:
            penalty += PEN * (max_r - limit_rocof)
        return total_capacity + penalty

    res = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, polish=True, disp=True)
    # extraer solución fina y métricas
    FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val = res.x
    max_fd, max_r, results_df = run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val, verbose=True)
    summary = {
        'FFR': FFR_cap, 'PFR': PFR_cap, 'aFFR': aFFR_cap, 'mFFR': mFFR_cap, 'RR_max': RR_max, 'H': H_val,
        'max_freq_dev': max_fd, 'max_rocof': max_r, 'total_capacity': FFR_cap + PFR_cap + aFFR_cap + mFFR_cap + RR_max
    }
    return res, summary, results_df

# ============================
# EJECUCIÓN: ejemplo
# ============================
if __name__ == "__main__":
    # # 1) Grid search (puede tardar según la malla)
    # print("Starting grid search (coarse)...")
    # all_df, ok_df = grid_search()
    # print(f"Grid search done. Total combos: {len(all_df)}, acceptable combos: {len(ok_df)}")
    # if not ok_df.empty:
    #     print("Top acceptable combos (menor capacidad total):")
    #     print(ok_df)

    # 2) Optimization (DE)
    print("\nStarting optimization (differential_evolution)...")
    res, summary, opt_results_df = optimize_search(maxiter=5, popsize=2)
    print("Optimization summary:")
    print(summary)

    # 3) Plot the optimized result time series
    _, _, df_best = run_simulation(summary['FFR'], summary['PFR'], summary['aFFR'], summary['mFFR'], summary['RR_max'], summary['H'], verbose=False)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(df_best['Time [s]'], df_best['Frequency Deviation [Hz]'], 'k', lw=2, label='Frequency deviation [Hz]')
    ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Frequency deviation [Hz]')
    ax2 = ax1.twinx()
    ax2.plot(df_best['Time [s]'], df_best['FFR [MW]'], label='FFR')
    ax2.plot(df_best['Time [s]'], df_best['PFR [MW]'], label='PFR')
    ax2.plot(df_best['Time [s]'], df_best['aFFR [MW]'], label='aFFR')
    ax2.plot(df_best['Time [s]'], df_best['mFFR_total [MW]'], label='mFFR_total')
    ax2.plot(df_best['Time [s]'], df_best['Contingency [MW]'], 'k--', label='Contingency')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.title('Optimized solution time series')
    plt.tight_layout()
    plt.show()


#%%

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
Contingency_val = 300.0     # MW
H_default = 50.0            # valor por defecto (se variará en la búsqueda)
D = 1.0                     # amortiguamiento simple MW/Hz
Krr = 1.0                   # ganancia integrador RR (no cambia)
Prr_min = 0.0               # RR mínimo

# PERFIL DE TIEMPO (igual que tu original)
t1 = np.arange(0, 60, 0.1)
t2 = np.arange(60, 2000, 0.1)
t = np.unique(np.concatenate((t1, t2)))

# ============================
# PARÁMETROS DE ACTIVACIÓN (igual que tu original)
# ============================
# FFR (Fast Frequency Response)
t_ffr_start = 1
t_ffr_ramp_end = 5
t_ffr_hold_end = 20
t_ffr_ramp_down_end = 30

# PFR (Primary Frequency Response)
t_pfr_start = 6
t_pfr_ramp_end = 30
t_pfr_hold_end = 180    # 3 min
t_pfr_ramp_down_end = 900   # 15 min

# AFFR (Automatic Fast Frequency Response)
t_affr_start = 40
t_affr_ramp_end = 300   # 5 min
t_affr_hold_end = 900   # 15 min
t_affr_ramp_down_end = 1200  # 20 min

# MFFR (Manual Fast Frequency Response)
t_mffr_start = 600    # 10 min
t_mffr_ramp_end = 1800  # 30 min

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
def run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val,
                   contingency=Contingency_val, D_local=D, verbose=False):
    """
    Ejecuta simulación y devuelve métricas:
    - max_freq_dev (Hz, valor absoluto máximo de desviación)
    - max_rocof (Hz/s, valor absoluto máximo)
    - results_df (DataFrame con series temporales)
    """
    def state(y, tt):
        f = y[0]
        I_rr = y[1]
        contingency_local = contingency if tt >= 1.0 else 0.0

        # pesos en instante tt (scalar)
        w_ffr, w_pfr, w_affr, w_mffr = compute_weights(np.array([tt]))
        w_ffr = float(w_ffr[0]); w_pfr = float(w_pfr[0]); w_affr = float(w_affr[0]); w_mffr = float(w_mffr[0])

        # reservas desplegadas
        ffr = FFR_cap * w_ffr
        pfr = PFR_cap * w_pfr
        affr = aFFR_cap * w_affr
        mffr_prop = mFFR_cap * w_mffr

        # RR desde integrador
        P_rr = Krr * I_rr * w_mffr
        # saturación anti-windup
        P_rr = np.clip(P_rr, Prr_min, RR_max)

        deltap = contingency_local - (ffr + pfr + affr + mffr_prop + P_rr) - D_local * f
        dfdt = deltap / (1000.0 * (2.0 * H_val / 50.0))
        dIdt = f
        return [dfdt, dIdt]

    y0 = [0.0, 0.0]
    sol = odeint(state, y0, t)
    f = sol[:, 0]
    I_rr = sol[:, 1]

    ffr = FFR_cap * W_ffr_vec
    pfr = PFR_cap * W_pfr_vec
    affr = aFFR_cap * W_affr_vec
    mffr_prop = mFFR_cap * W_mffr_vec
    P_rr = Krr * I_rr * W_mffr_vec
    P_rr = np.clip(P_rr, Prr_min, RR_max)

    contingency_vec = np.where(t >= 1.0, contingency, 0.0)
    deltap = contingency_vec - (ffr + pfr + affr + mffr_prop + P_rr) - D_local * f

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
        'mFFR_prop [MW]': mffr_prop,
        'P_rr [MW]': P_rr,
        'mFFR_total [MW]': mffr_prop + P_rr,
        'Contingency [MW]': contingency_vec,
        'deltap [MW]': -deltap
    })

    if verbose:
        print(f"Sim finished: FFR={FFR_cap},PFR={PFR_cap},aFFR={aFFR_cap},mFFR={mFFR_cap},RR_max={RR_max},H={H_val}")
        print(f" -> max_freq_dev={max_freq_dev:.4f} Hz, max_rocof={max_rocof:.4f} Hz/s")

    return max_freq_dev, max_rocof, results_df

# ============================
# OPTIMIZACIÓN CON PENALIZACIÓN FUERTE
# ============================
def optimize_search(limit_freq=0.8, limit_rocof=0.5, maxiter=50, popsize=15):
    """
    Optimiza capacidad total (FFR+PFR+aFFR+mFFR+RR_max) minimizando
    y penalizando fuerte si se violan límites de frecuencia y ROCOF.
    """

    bounds = [
        (0, 10000),  # FFR
        (0, 10000),  # PFR
        (0, 10000),  # aFFR
        (0, 10000),  # mFFR
        (0, 10000),  # RR_max
        (25, 50)    # H (inercias)
    ]

    def objective(x):
        FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val = x
        max_fd, max_r, _ = run_simulation(FFR_cap, PFR_cap, aFFR_cap, mFFR_cap, RR_max, H_val)
        total_capacity = FFR_cap + PFR_cap + aFFR_cap + mFFR_cap + RR_max

        penalty = 0
        if max_fd > limit_freq:
            penalty += 1e7 * (max_fd - limit_freq)**2
        if max_r > limit_rocof:
            penalty += 1e7 * (max_r - limit_rocof)**2

        return total_capacity + penalty

    print("Iniciando optimización (differential_evolution)...")
    start_time = time.time()

    result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, disp=True)

    elapsed_time = time.time() - start_time
    print(f"Optimización terminada en {elapsed_time:.1f} segundos.")

    best_params = result.x
    best_FFR, best_PFR, best_aFFR, best_mFFR, best_RR_max, best_H = best_params
    best_max_fd, best_max_r, best_results_df = run_simulation(best_FFR, best_PFR, best_aFFR, best_mFFR, best_RR_max, best_H, verbose=True)

    print("\n--- RESULTADOS FINALES ---")
    print(f"FFR={best_FFR:.2f} MW, PFR={best_PFR:.2f} MW, aFFR={best_aFFR:.2f} MW, mFFR={best_mFFR:.2f} MW, RR_max={best_RR_max:.2f} MW, H={best_H:.2f} MWs")
    print(f"Max freq deviation: {best_max_fd:.4f} Hz (límite {limit_freq})")
    print(f"Max ROCOF: {best_max_r:.4f} Hz/s (límite {limit_rocof})")

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
    ax2.plot(df_best['Time [s]'], df_best['FFR [MW]'], label='FFR')
    ax2.fill_between(df_best['Time [s]'], df_best['FFR [MW]'], 0, where=df_best['FFR [MW]'] > 0, color='gold', alpha=0.4)
    ax2.plot(df_best['Time [s]'], df_best['PFR [MW]'], label='PFR')
    ax2.fill_between(df_best['Time [s]'], df_best['PFR [MW]'], 0, where=df_best['PFR [MW]'] > 0, color='blue', alpha=0.3)
    ax2.plot(df_best['Time [s]'], df_best['aFFR [MW]'], label='aFFR')
    ax2.fill_between(df_best['Time [s]'], df_best['aFFR [MW]'], 0, where=df_best['aFFR [MW]'] > 0, color='green', alpha=0.3)
    ax2.plot(df_best['Time [s]'], df_best['mFFR_total [MW]'], label='mFFR_total')
    ax2.fill_between(df_best['Time [s]'], df_best['mFFR_total [MW]'], 0, where=df_best['mFFR_total [MW]'] > 0, color='red', alpha=0.2)
    
    ax2.plot(df_best['Time [s]'], df_best['Contingency [MW]'], 'k--', label='Contingency')

    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax1.set_ylim(-3, 3)
    ax2.set_ylim(-610, 610)
    plt.xlabel('Time [s]')
    plt.ylabel('Valor')
    plt.title('Optimized solution time series')
    plt.legend()
    plt.grid(True)
    plt.show()




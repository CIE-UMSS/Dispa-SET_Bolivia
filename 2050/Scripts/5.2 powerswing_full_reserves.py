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

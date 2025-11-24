# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:51:39 2025

@author: navia
"""
#%% 1. PFR SIN AREA PINTADA
# CONTINGENCY 3
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
# k1 = 60               # MW/Hz/s
k = 500                # MW/Hz/s
H = 30                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k, H):  #k1,
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        # ffr = 0
        deltap = contingency
    # elif t < time_delay:
    #     contingency_val = contingency
    #     primaryreserve = 0
    #     ffr = 0
    #     deltap = contingency_val
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        # ffr = (k1 * f) * (t - time_delay)
        deltap = contingency
    else:
        contingency = Contingency_val
        # ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        # deltap = contingency_val - ffr - primaryreserve
        deltap = contingency - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k, H)) #k1
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
# ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    # 'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color='k', fontsize=font_size)
ax1.tick_params(axis='y', labelcolor='k', labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
# color_ffr = '#2ca02c'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'

# ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-2, 2)     # Frecuencia y RoCoF
ax2.set_ylim(-1200, 1200)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k={k}, H={H}', fontsize=26, weight='bold') #k1={k1},
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())
#%% 2. PFR SIN AREA PINTADA
# CONTINGENCY 5
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
# k1 = 60               # MW/Hz/s
k = 500                # MW/Hz/s
H = 50                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k, H):  #k1,
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        # ffr = 0
        deltap = contingency
    # elif t < time_delay:
    #     contingency_val = contingency
    #     primaryreserve = 0
    #     ffr = 0
    #     deltap = contingency_val
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        # ffr = (k1 * f) * (t - time_delay)
        deltap = contingency
    else:
        contingency = Contingency_val
        # ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        # deltap = contingency_val - ffr - primaryreserve
        deltap = contingency - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k, H)) #k1
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
# ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    # 'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color='k', fontsize=font_size)
ax1.tick_params(axis='y', labelcolor='k', labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
# color_ffr = '#2ca02c'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'

# ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-2, 2)     # Frecuencia y RoCoF
ax2.set_ylim(-1200, 1200)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k={k}, H={H}', fontsize=26, weight='bold') #k1={k1},
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())
#%% 3. PFR CON AREA PINTADA
# CONTINGENCY 3
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
# k1 = 60               # MW/Hz/s
k = 500                # MW/Hz/s
H = 30                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k, H):  #k1,
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        # ffr = 0
        deltap = contingency
    # elif t < time_delay:
    #     contingency_val = contingency
    #     primaryreserve = 0
    #     ffr = 0
    #     deltap = contingency_val
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        # ffr = (k1 * f) * (t - time_delay)
        deltap = contingency
    else:
        contingency = Contingency_val
        # ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        # deltap = contingency_val - ffr - primaryreserve
        deltap = contingency - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k, H)) #k1
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
# ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    # 'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color='k', fontsize=font_size)
ax1.tick_params(axis='y', labelcolor='k', labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
# color_ffr = '#2ca02c'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'

# # FFR
# ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
# ax2.fill_between(t, ffr, 0, where=ffr >= 0, interpolate=True, color=color_ffr, alpha=0.5)

# Primary Reserve
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.fill_between(t, primaryreserve, 0, where=primaryreserve >= 0, interpolate=True, color=color_pr, alpha=0.5)

ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-2, 2)     # Frecuencia y RoCoF
ax2.set_ylim(-1200, 1200)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k={k}, H={H}', fontsize=26, weight='bold') #k1={k1},
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())
#%% 4. PFR CON AREA PINTADA
# CONTINGENCY 5
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
# k1 = 60               # MW/Hz/s
k = 500                # MW/Hz/s
H = 50                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k, H):  #k1,
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        # ffr = 0
        deltap = contingency
    # elif t < time_delay:
    #     contingency_val = contingency
    #     primaryreserve = 0
    #     ffr = 0
    #     deltap = contingency_val
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        # ffr = (k1 * f) * (t - time_delay)
        deltap = contingency
    else:
        contingency = Contingency_val
        # ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        # deltap = contingency_val - ffr - primaryreserve
        deltap = contingency - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k, H)) #k1
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
# ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    # 'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color='k', fontsize=font_size)
ax1.tick_params(axis='y', labelcolor='k', labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
# color_ffr = '#2ca02c'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'

# # FFR
# ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
# ax2.fill_between(t, ffr, 0, where=ffr >= 0, interpolate=True, color=color_ffr, alpha=0.5)

# Primary Reserve
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.fill_between(t, primaryreserve, 0, where=primaryreserve >= 0, interpolate=True, color=color_pr, alpha=0.5)

ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-2, 2)     # Frecuencia y RoCoF
ax2.set_ylim(-1200, 1200)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k={k}, H={H}', fontsize=26, weight='bold') #k1={k1},
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())


#%% 5. FFR Y PFR SIN AREA PINTADA
# CONTINGENCY 6
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
k1 = 500               # MW/Hz/s
k = 500                # MW/Hz/s
H = 25                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k1, k, H):
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_delay:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = (k1 * f) * (t - time_delay)
        deltap = contingency - ffr
    else:
        contingency = Contingency_val
        ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        deltap = contingency - ffr - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k1, k, H))
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - ffr - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color=color_freq, fontsize=font_size)
ax1.tick_params(axis='y', labelcolor=color_freq, labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
color_ffr = 'goldenrod'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'

ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-1.5, 1.5)     # Frecuencia y RoCoF
ax2.set_ylim(-610, 610)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k1={k1}, k={k}, H={H}', fontsize=26, weight='bold')
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())

#%% 6. FFR Y PFR SIN AREA PINTADA
# CONTINGENCY 10
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
k1 = 500               # MW/Hz/s
k = 500                # MW/Hz/s
H = 25                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k1, k, H):
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_delay:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = (k1 * f) * (t - time_delay)
        deltap = contingency - ffr
    else:
        contingency = Contingency_val
        ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        deltap = contingency - ffr - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k1, k, H))
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - ffr - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color=color_freq, fontsize=font_size)
ax1.tick_params(axis='y', labelcolor=color_freq, labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
color_ffr = 'goldenrod'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'

ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-1.5, 1.5)     # Frecuencia y RoCoF
ax2.set_ylim(-610, 610)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k1={k1}, k={k}, H={H}', fontsize=26, weight='bold')
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())

#%% 7. FFR Y PFR CON AREA PINTADA
# CONTINGENCY 6
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
k1 = 100               # MW/Hz/s
k = 500                # MW/Hz/s
H = 25                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k1, k, H):
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_delay:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = (k1 * f) * (t - time_delay)
        deltap = contingency - ffr
    else:
        contingency = Contingency_val
        ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        deltap = contingency - ffr - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k1, k, H))
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - ffr - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color=color_freq, fontsize=font_size)
ax1.tick_params(axis='y', labelcolor=color_freq, labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
color_ffr = 'goldenrod'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'


# FFR
ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
ax2.fill_between(t, ffr, 0, where=ffr >= 0, interpolate=True, color=color_ffr, alpha=0.5)

# Primary Reserve
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.fill_between(t, primaryreserve, 0, where=primaryreserve >= 0, interpolate=True, color=color_pr, alpha=0.5)

ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-1.5, 1.5)     # Frecuencia y RoCoF
ax2.set_ylim(-610, 610)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k1={k1}, k={k}, H={H}', fontsize=26, weight='bold')
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())

#%% 8. FFR Y PFR CON AREA PINTADA
# CONTINGENCY 10
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ============================
# PARÁMETROS DE ENTRADA
# ============================
Contingency_val = 400     # MW
k1 = 500               # MW/Hz/s
k = 500                # MW/Hz/s
H = 25                # MW·s/MVA
time_delay = 2        # segundos
time_preparation = 5  # segundos

# ============================
# FUNCIÓN DIFERENCIAL
# ============================
def power_swing(y, t, k1, k, H):
    deltap, f = y
    if t < 1:
        contingency = 0
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_delay:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = 0
        deltap = contingency
    elif t < time_preparation:
        contingency = Contingency_val
        primaryreserve = 0
        ffr = (k1 * f) * (t - time_delay)
        deltap = contingency - ffr
    else:
        contingency = Contingency_val
        ffr = (k1 * f) * (t - time_delay)
        primaryreserve = (k * f) * (t - time_preparation)
        deltap = contingency - ffr - primaryreserve

    delf = deltap * 50 / (2 * H * 1000)
    return [deltap, delf]

# ============================
# SIMULACIÓN
# ============================
f0 = 0
y0 = [0, f0]
t = np.arange(0, 61, 0.1)

sol = odeint(power_swing, y0, t, args=(k1, k, H))
f = sol[:, 1]
der_f = np.gradient(f, t)

# ============================
# CÁLCULO DE COMPONENTES
# ============================
ffr = np.where(t < time_delay, 0, ((k1 * f) * (t - time_delay)))
primaryreserve = np.where(t < time_preparation, 0, ((k * f) * (t - time_preparation)))
contingency_series = np.where(t < 1, 0, Contingency_val)
deltap = np.where(t < 1, 0, Contingency_val  - ffr - primaryreserve)
# ============================
# GUARDAR EN DATAFRAME
# ============================
results_df = pd.DataFrame({
    'Time [s]': t,
    'Frequency Deviation [Hz]': -f,
    'dF/dt [Hz/s]': der_f,
    'FFR [MW]': ffr,
    'Primary Reserve [MW]': primaryreserve,
    'Deltap [MW]': deltap,
    'Contingency [MW]': -contingency_series
})
# Nuevas columnas solicitadas
results_df['RoCoF [Hz/s]'] = -der_f
results_df['DeltaP [MW]'] = -deltap
results_df['Contingency [MW]'] = contingency_series

# ============================
# PLOT PROFESIONAL
# ============================
fig, ax1 = plt.subplots(figsize=(14, 8))

# ===== EJE Y IZQUIERDO =====
color_freq = 'black'
color_roc = 'fuchsia'
line_width = 1.5
font_size = 20

ax1.plot(t, -f, color=color_freq, linewidth=line_width, label='Frequency deviation [Hz]')
ax1.plot(t, -der_f, color=color_roc, linewidth=line_width, linestyle='--', label='RoCoF [Hz/s]')
ax1.set_xlabel('Time [s]', fontsize=font_size)
ax1.set_ylabel('Frequency / RoCoF [Hz, Hz/s]', color=color_freq, fontsize=font_size)
ax1.tick_params(axis='y', labelcolor=color_freq, labelsize=20)
ax1.tick_params(axis='x', labelsize=20) 
ax1.grid(True, linestyle='--', alpha=0.6)

# ===== EJE Y DERECHO =====
ax2 = ax1.twinx()
color_ffr = 'goldenrod'
color_pr = 'blue'
color_dp = 'green'
color_cont = 'red'


# FFR
ax2.plot(t, ffr, color=color_ffr, linewidth=line_width, linestyle='-', label='FFR [MW]')
ax2.fill_between(t, ffr, 0, where=ffr >= 0, interpolate=True, color=color_ffr, alpha=0.5)

# Primary Reserve
ax2.plot(t, primaryreserve, color=color_pr, linewidth=line_width, linestyle='-', label='PFR [MW]')
ax2.fill_between(t, primaryreserve, 0, where=primaryreserve >= 0, interpolate=True, color=color_pr, alpha=0.5)

ax2.plot(t, -deltap, color=color_dp, linewidth=line_width, linestyle='-.', label='∆P [MW]')
ax2.plot(t, -contingency_series, color=color_cont, linewidth=line_width, linestyle='--', label='Contingency [MW]')
ax2.set_ylabel('Power [MW]', color='k', fontsize=font_size)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)

# ===== SINCRONIZAR CERO DE AMBOS EJES =====
ax1.set_ylim(auto=True)
y1min, y1max = ax1.get_ylim()
y2min, y2max = ax2.get_ylim()
offset = max(abs(y2min), abs(y2max))
ax2.set_ylim(-offset, offset)

# ===== LEYENDAS SEPARADAS =====
lines_left, labels_left = ax1.get_legend_handles_labels()
lines_right, labels_right = ax2.get_legend_handles_labels()
ax1.legend(lines_left, labels_left, loc='upper left', fontsize=font_size)
ax2.legend(lines_right, labels_right, loc='upper right', fontsize=font_size)
ax1.set_ylim(-1.5, 1.5)     # Frecuencia y RoCoF
ax2.set_ylim(-610, 610)     # Potencias

# ===== LIMITE DEL EJE X =====
ax1.set_xlim(0, 60)

# ===== TITULO Y DISEÑO FINAL =====
plt.title(f'Contingency={Contingency_val} MW, k1={k1}, k={k}, H={H}', fontsize=26, weight='bold')
plt.tight_layout()
plt.show()

# ============================
# MOSTRAR RESULTADOS
# ============================
print(results_df.head())



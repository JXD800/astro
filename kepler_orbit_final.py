
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
import requests
from io import StringIO

def load_kepler_data():
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+pl_name,hostname,pl_orbper,pl_orbsmax,pl_radj,"
        "pl_bmassj,disc_year,pl_orbeccen,st_mass,st_rad,st_age,"
        "st_teff,st_spectype,sy_dist+from+ps&format=csv"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data")
    csv_data = response.content.decode("utf-8")
    kepler_data = pd.read_csv(StringIO(csv_data))
    kepler_data.rename(columns={
        'pl_name': 'name', 'hostname': 'star_name', 'pl_orbper': 'orbital_period',
        'pl_orbsmax': 'semi_major_axis', 'pl_radj': 'radius', 'pl_bmassj': 'mass',
        'disc_year': 'discovery_year', 'pl_orbeccen': 'eccentricity',
        'st_mass': 'star_mass', 'st_rad': 'star_radius', 'st_age': 'star_age',
        'st_teff': 'star_temp', 'st_spectype': 'star_type', 'sy_dist': 'star_distance_ly'
    }, inplace=True)
    return kepler_data

def get_hr_position(temp):
    try:
        T = float(temp)
        if T > 30000: return "O (blue)"
        elif T > 10000: return "B (blue-white)"
        elif T > 7500: return "A (white)"
        elif T > 6000: return "F (yellow-white)"
        elif T > 5200: return "G (yellow)"
        elif T > 3700: return "K (orange)"
        else: return "M (red)"
    except:
        return "Unknown"

def plot_star_spectrum():
    selected = star_var.get()
    filtered = kepler_data[kepler_data['star_name'].str.lower() == selected.lower()]
    if filtered.empty:
        messagebox.showerror("Error", "Star not found")
        return
    temp = filtered.iloc[0]['star_temp']
    if pd.isna(temp) or temp <= 0:
        messagebox.showinfo("No Data", "Temperature not available.")
        return
    T = float(temp)
    wavelengths = np.linspace(100e-9, 3000e-9, 500)
    h, c, k = 6.626e-34, 3e8, 1.381e-23
    spectral_radiance = (2*h*c**2 / wavelengths**5) / (np.exp(h*c / (wavelengths * k * T)) - 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wavelengths * 1e9, spectral_radiance)
    ax.set_title(f"Brightness Spectrum of {selected} (T={T}K)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.grid(True)
    for w in canvas_frame.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def calc_distance_to_earth():
    selected = star_var.get()
    filtered = kepler_data[kepler_data['star_name'].str.lower() == selected.lower()]
    if filtered.empty: return
    dist = filtered.iloc[0].get('star_distance_ly')
    msg = f"Distance to Earth: {dist:.2f} light-years" if pd.notna(dist) else "Distance to Earth: unknown"
    output.insert(tk.END, f"\n[INFO] {msg}\n")
    messagebox.showinfo("Distance to Earth", msg)

def calc_position():
    selected_star = star_var.get()
    selected_planet = planet_var.get()
    filt = kepler_data[(kepler_data['star_name'].str.lower() == selected_star.lower()) &
                       (kepler_data['name'].str.lower() == selected_planet.lower())]
    if filt.empty: messagebox.showerror("Dados não encontrados", f"A combinação para a estrela '{selected_star}' e o planeta '{selected_planet}' não foi encontrada.\n\n" "verifique se um planeta foi selecionado corretamente para a estrela escolhida."
)
return

p = filt.iloc[0]
output.delete('1.0', tk.END)
output.insert(tk.END, f"[EXOPLANET]\nName: {p['name']}\nSemi-Major Axis: {p['semi_major_axis']} AU\n")
    output.insert(tk.END, f"Period: {p['orbital_period']} days\nRadius: {p['radius']} Rj\nMass: {p['mass']} Mj\n")
    output.insert(tk.END, f"[STAR]\nName: {p['star_name']}\nTemp: {p['star_temp']} K\nSpectral Type: {p['star_type']}\n")
    output.insert(tk.END, f"HR Class: {get_hr_position(p['star_temp'])}\n")
    try:
        a = float(p['semi_major_axis'])
        ecc = p['eccentricity'] if pd.notna(p['eccentricity']) else 0
        period = float(p['orbital_period'])
        b = a * np.sqrt(1 - ecc**2)
        theta = np.linspace(0, 2*np.pi, 360)
        x_orb = a*np.cos(theta) - a*ecc
        y_orb = b*np.sin(theta)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.set_facecolor("black")
        ax.plot(x_orb, y_orb, 'w--', alpha=0.6)
        ax.plot(0, 0, 'yo')
        ax.set_title(f"Orbit of {p['name']}")
        ax.set_xlim(-a*1.5, a*1.5)
        ax.set_ylim(-a*1.5, a*1.5)
        planet_dot, = ax.plot([], [], 'ro')
        dist_line, = ax.plot([], [], 'r--')
        mean_motion = 2*np.pi / (period * 86400)

        def update(frame):
            M = mean_motion * frame
            x = a*np.cos(M) - a*ecc
            y = b*np.sin(M)
            planet_dot.set_data([x], [y])
            dist_line.set_data([0, x], [0, y])
            info_var.set(f"Distance to star: {np.sqrt(x**2 + y**2):.3f} AU")
            return planet_dot, dist_line

        for w in canvas_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack()
        ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, period * 86400, 180),
                                      interval=100, blit=True, repeat=True)
        canvas.draw()
    except Exception as e:
        output.insert(tk.END, f"\n[ERROR] Orbit: {e}\n")

# GUI
kepler_data = load_kepler_data()
host_stars = sorted(kepler_data['star_name'].dropna().unique())
root = tk.Tk()
root.title("Kepler Viewer")
root.geometry("1200x700")
frame = ttk.Frame(root, padding=10)
frame.pack(fill='both', expand=True)
left = ttk.Frame(frame)
left.pack(side='left', fill='y')
right = ttk.Frame(frame)
right.pack(side='right', expand=True, fill='both')

ttk.Label(left, text="Select Star:").pack()
star_var = tk.StringVar()
star_combo = ttk.Combobox(left, textvariable=star_var, values=host_stars, width=30)
star_combo.pack()

planet_var = tk.StringVar()
planet_combo = ttk.Combobox(left, textvariable=planet_var, width=30)
planet_combo.pack(pady=5)

output = tk.Text(right, height=12, width=100)
output.pack(pady=5)

canvas_frame = ttk.Frame(right)
canvas_frame.pack(fill='both', expand=True)

info_var = tk.StringVar()
ttk.Label(right, textvariable=info_var, foreground="blue").pack(pady=2)

def update_planets(event=None):
    selected = star_var.get()
    planets = kepler_data[kepler_data['star_name'].str.lower() == selected.lower()]['name'].tolist()
    planet_combo['values'] = planets
    if planets:
        planet_var.set(planets[0])

star_combo.bind("<<ComboboxSelected>>", update_planets)

ttk.Button(left, text="Calculate Orbit", command=calc_position).pack(pady=5)
ttk.Button(left, text="Distance to Earth", command=calc_distance_to_earth).pack(pady=5)
ttk.Button(left, text="Plot Star Spectrum", command=plot_star_spectrum).pack(pady=5)
ttk.Button(left, text="Exit", command=root.destroy).pack(pady=5)

update_planets()
root.mainloop()

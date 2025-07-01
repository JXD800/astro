Understood. I will remove the buttons and make the spectrum and H-R diagram plots update automatically when you select a star or planet.

This change streamlines the interface, making it more responsive and intuitive. Here is the final code with these modifications implemented.

### Final Code with Automatic Plot Updates

```python
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
from io import StringIO
from typing import Optional

# Matplotlib and SciPy imports for plotting and calculations
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import newton

# --- Physical and Solar Constants ---
H_PLANCK = 6.626e-34      # Planck's constant
C_LIGHT = 3e8             # Speed of light
K_BOLTZMANN = 1.381e-23   # Boltzmann constant
SOLAR_TEMP = 5778         # Sun's effective temperature in Kelvin

def load_kepler_data() -> pd.DataFrame:
    """Fetches exoplanet data from the NASA Exoplanet Archive."""
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+pl_name,hostname,pl_orbper,pl_orbsmax,pl_radj,"
        "pl_bmassj,disc_year,pl_orbeccen,st_mass,st_rad,st_age,"
        "st_teff,st_spectype,sy_dist+from+ps&format=csv"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Network Error", f"Failed to fetch data: {e}")
        return pd.DataFrame()

    csv_data = response.content.decode("utf-8")
    kepler_data = pd.read_csv(StringIO(csv_data))
    kepler_data = kepler_data.rename(columns={
        'pl_name': 'name', 'hostname': 'star_name', 'pl_orbper': 'orbital_period',
        'pl_orbsmax': 'semi_major_axis', 'pl_radj': 'radius', 'pl_bmassj': 'mass',
        'disc_year': 'discovery_year', 'pl_orbeccen': 'eccentricity',
        'st_mass': 'star_mass', 'st_rad': 'star_radius', 'st_age': 'star_age',
        'st_teff': 'star_temp', 'st_spectype': 'star_type', 'sy_dist': 'star_distance'
    })
    return kepler_data

def get_hr_position(temp: Optional[float]) -> str:
    """Determines a star's spectral class from its temperature."""
    if pd.isna(temp): return "Unknown"
    try:
        T = float(temp)
        if T > 30000: return "O (blue)"
        elif T > 10000: return "B (blue-white)"
        elif T > 7500: return "A (white)"
        elif T > 6000: return "F (yellow-white)"
        elif T > 5200: return "G (yellow)"
        elif T > 3700: return "K (orange)"
        else: return "M (red)"
    except (ValueError, TypeError): return "Unknown"

def plot_hr_diagram():
    """Plots a Hertzsprung-Russell diagram in its dedicated tab."""
    ax_hr.clear()
    hr_data = kepler_data.dropna(subset=['star_temp', 'star_radius']).copy()
    if hr_data.empty:
        ax_hr.text(0.5, 0.5, "Not Enough Data for H-R Diagram", ha='center', va='center')
        canvas_hr.draw()
        return

    hr_data['luminosity'] = (hr_data['star_radius']**2) * (hr_data['star_temp'] / SOLAR_TEMP)**4
    ax_hr.scatter(hr_data['star_temp'], hr_data['luminosity'], alpha=0.3, s=5, label='Exoplanet Host Stars')

    selected_star_data = hr_data[hr_data['star_name'] == star_var.get()]
    if not selected_star_data.empty:
        star = selected_star_data.iloc[0]
        ax_hr.scatter(star['star_temp'], star['luminosity'], color='red', s=50, edgecolor='black', zorder=5, label=f'Selected: {star_var.get()}')

    ax_hr.set_yscale('log')
    ax_hr.set_xscale('log')
    ax_hr.set_xlim(ax_hr.get_xlim()[::-1])
    ax_hr.set_title("Hertzsprung-Russell Diagram")
    ax_hr.set_xlabel("Effective Temperature (K)")
    ax_hr.set_ylabel("Luminosity (Solar Units)")
    ax_hr.grid(True, which="both", ls="--", alpha=0.5)
    ax_hr.legend()
    canvas_hr.draw()

def plot_star_spectrum():
    """Plots the star's black-body radiation spectrum in its dedicated tab."""
    ax_spec.clear()
    selected_star = star_var.get()
    filtered = kepler_data[kepler_data['star_name'].str.lower() == selected_star.lower()]
    if filtered.empty or pd.isna(filtered.iloc[0]['star_temp']):
        ax_spec.text(0.5, 0.5, "Temperature Data Not Available", ha='center', va='center')
        canvas_spec.draw()
        return

    T = filtered.iloc[0]['star_temp']
    wavelengths = np.linspace(100e-9, 3000e-9, 500)
    spectral_radiance = (2*H_PLANCK*C_LIGHT**2/wavelengths**5) / (np.exp(H_PLANCK*C_LIGHT/(wavelengths*K_BOLTZMANN*T)) - 1)
    
    max_radiance = np.max(spectral_radiance)
    normalized_radiance = spectral_radiance / max_radiance if max_radiance > 0 else spectral_radiance

    ax_spec.plot(wavelengths * 1e9, normalized_radiance, color='orange', linewidth=2)
    ax_spec.set_title(f"Black-Body Spectrum of {selected_star} (T≈{T:,.0f} K)")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Normalized Intensity")
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    ax_spec.set_ylim(0, 1.1)
    canvas_spec.draw()

def calc_distance_to_earth():
    """Calculates and displays the distance to the selected star."""
    selected_star = star_var.get()
    filtered = kepler_data[kepler_data['star_name'].str.lower() == selected_star.lower()]
    if filtered.empty: return
        
    dist = filtered.iloc[0]['star_distance']
    if pd.notna(dist):
        dist_ly = dist * 3.26156
        msg = f"Distance to {selected_star}: {dist:.2f} parsecs ({dist_ly:.2f} light-years)"
    else:
        msg = f"Distance to {selected_star}: Unknown"
    messagebox.showinfo("Distance to Earth", msg)

def calc_position():
    """Calculates and animates the planet's orbit."""
    selected_star, selected_planet = star_var.get(), planet_var.get()
    filt = kepler_data[(kepler_data['star_name'] == selected_star) & (kepler_data['name'] == selected_planet)]
                       
    if filt.empty:
        messagebox.showerror("Data Not Found", f"Could not find data for planet '{selected_planet}' orbiting '{selected_star}'.")
        return

    p = filt.iloc[0]
    output.delete('1.0', tk.END)
    output.insert(tk.END, f"[EXOPLANET]\nName: {p['name']}\nSemi-Major Axis: {p.get('semi_major_axis', 'N/A')} AU\n")
    output.insert(tk.END, f"Period: {p.get('orbital_period', 'N/A')} days\nRadius: {p.get('radius', 'N/A')} Rj\nMass: {p.get('mass', 'N/A')} Mj\n")
    output.insert(tk.END, f"Eccentricity: {p.get('eccentricity', 'N/A')}\n\n[STAR]\nName: {p['star_name']}\nTemp: {p['star_temp']} K\n")

    if hasattr(canvas_orbit, 'ani') and canvas_orbit.ani:
        canvas_orbit.ani.event_source.stop()

    ax_orbit.clear()
    
    try:
        a = float(p['semi_major_axis'])
        period_days = float(p['orbital_period'])
        ecc = float(p['eccentricity']) if pd.notna(p['eccentricity']) else 0.0

        b = a * np.sqrt(1 - ecc**2)
        theta = np.linspace(0, 2 * np.pi, 360)
        x_orb, y_orb = a * (np.cos(theta) - ecc), b * np.sin(theta)

        ax_orbit.plot(x_orb, y_orb, 'w--', alpha=0.5)
        ax_orbit.plot(0, 0, 'o', color='gold', markersize=12)
        ax_orbit.set_title(f"Orbit of {p['name']}", fontsize=10)
        ax_orbit.set_facecolor("black")
        ax_orbit.set_xlim(-a * 1.5, a * 1.5); ax_orbit.set_ylim(-a * 1.5, a * 1.5)
        ax_orbit.set_aspect('equal', adjustable='box'); ax_orbit.grid(True, linestyle='--', color='gray', alpha=0.4)
        planet_dot, = ax_orbit.plot([], [], 'o', color='cyan', markersize=6)
        dist_line, = ax_orbit.plot([], [], '--', color='red', alpha=0.7)
        
        period_seconds = period_days * 86400
        mean_motion = 2 * np.pi / period_seconds

        def update(frame_time: float) -> tuple:
            M = mean_motion * frame_time
            try: E = newton(lambda E: E - ecc * np.sin(E) - M, M, tol=1e-6)
            except RuntimeError: E = M
            
            x, y = a * (np.cos(E) - ecc), b * np.sin(E)
            planet_dot.set_data([x], [y])
            dist_line.set_data([0, x], [0, y])
            return planet_dot, dist_line

        canvas_orbit.ani = animation.FuncAnimation(fig_orbit, update, frames=np.linspace(0, period_seconds, 360), interval=50, blit=True, repeat=True)
        canvas_orbit.draw()

    except (ValueError, TypeError, KeyError) as e:
        messagebox.showerror("Animation Error", f"Could not plot orbit due to invalid or missing data.\nDetails: {e}")

def search_stars(event=None):
    """Searches the dataframe and populates the research treeview."""
    for item in research_tree.get_children(): research_tree.delete(item)
    query = search_var.get().strip().lower()
    if not query: return

    mask = kepler_data['star_name'].str.lower().str.contains(query, na=False)
    results = kepler_data[mask].sort_values(by=['star_name', 'name'])

    if results.empty:
        research_tree.insert('', 'end', values=("No results found.", "", "", "", ""))
        return

    for _, row in results.iterrows():
        research_tree.insert('', 'end', values=(
            row.get('star_name', 'N/A'), row.get('name', 'N/A'),
            f"{row.get('orbital_period', 0):.2f}" if pd.notna(row.get('orbital_period')) else 'N/A',
            f"{row.get('semi_major_axis', 0):.3f}" if pd.notna(row.get('semi_major_axis')) else 'N/A',
            row.get('discovery_year', 'N/A')))

def update_plots_and_planets(event=None):
    """Master update function called on new star selection."""
    # First, update the available planets
    planets = sorted(kepler_data[kepler_data['star_name'] == star_var.get()]['name'].tolist())
    planet_combo['values'] = planets
    if planets:
        planet_var.set(planets[0])
    else:
        planet_var.set('')
    
    # Now, automatically update the visualization tabs
    plot_star_spectrum()
    plot_hr_diagram()

# --- GUI Setup ---
kepler_data = load_kepler_data()
host_stars = sorted(kepler_data['star_name'].dropna().unique())

root = tk.Tk()
root.title("Exoplanet Data Viewer")
root.geometry("1280x800")

frame = ttk.Frame(root, padding=10)
frame.pack(fill='both', expand=True)

left_panel = ttk.Frame(frame, width=250)
left_panel.pack(side='left', fill='y', padx=(0, 10))
left_panel.pack_propagate(False)

right_panel = ttk.Frame(frame)
right_panel.pack(side='right', expand=True, fill='both')
notebook = ttk.Notebook(right_panel)
notebook.pack(expand=True, fill='both')

# -- Tab 1: Orbit Viewer --
orbit_tab = ttk.Frame(notebook, padding=5)
notebook.add(orbit_tab, text="Orbit Viewer")
output = tk.Text(orbit_tab, height=10)
output.pack(fill='x', pady=(0, 5))
fig_orbit = Figure(figsize=(5, 5), dpi=100)
ax_orbit = fig_orbit.add_subplot(111)
canvas_orbit = FigureCanvasTkAgg(fig_orbit, master=orbit_tab)
canvas_orbit.get_tk_widget().pack(fill='both', expand=True)

# -- Tab 2: Black-Body Spectrum --
spec_tab = ttk.Frame(notebook, padding=5)
notebook.add(spec_tab, text="Black-Body Spectrum")
fig_spec = Figure(figsize=(5, 5), dpi=100)
ax_spec = fig_spec.add_subplot(111)
canvas_spec = FigureCanvasTkAgg(fig_spec, master=spec_tab)
canvas_spec.get_tk_widget().pack(fill='both', expand=True)

# -- Tab 3: H-R Diagram --
hr_tab = ttk.Frame(notebook, padding=5)
notebook.add(hr_tab, text="H-R Diagram")
fig_hr = Figure(figsize=(5, 5), dpi=100)
ax_hr = fig_hr.add_subplot(111)
canvas_hr = FigureCanvasTkAgg(fig_hr, master=hr_tab)
canvas_hr.get_tk_widget().pack(fill='both', expand=True)

# -- Tab 4: Research --
research_tab = ttk.Frame(notebook, padding=10)
notebook.add(research_tab, text="Research")
search_frame = ttk.Frame(research_tab)
search_frame.pack(fill='x', pady=5)
ttk.Label(search_frame, text="Search Star Name:").pack(side='left')
search_var = tk.StringVar()
search_entry = ttk.Entry(search_frame, textvariable=search_var)
search_entry.pack(side='left', expand=True, fill='x', padx=5)
search_entry.bind("<Return>", search_stars)
ttk.Button(search_frame, text="Search", command=search_stars).pack(side='left')
tree_frame = ttk.Frame(research_tab)
tree_frame.pack(expand=True, fill='both')
cols = ('Star', 'Planet', 'Period (days)', 'Axis (AU)', 'Discovered')
research_tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
for col in cols: research_tree.heading(col, text=col); research_tree.column(col, width=120)
tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=research_tree.yview)
research_tree.configure(yscrollcommand=tree_scroll.set)
tree_scroll.pack(side='right', fill='y')
research_tree.pack(side='left', expand=True, fill='both')

# --- Add controls to the left panel ---
ttk.Label(left_panel, text="Star & Planet Selection", font="-weight bold").pack(pady=(0, 5))
ttk.Label(left_panel, text="Select Star:").pack(anchor='w')
star_var = tk.StringVar()
star_combo = ttk.Combobox(left_panel, textvariable=star_var, values=host_stars, state='readonly')
star_combo.pack(fill='x', pady=(0,5))
# *** BIND STAR SELECTION TO MASTER UPDATE FUNCTION ***
star_combo.bind("<<ComboboxSelected>>", update_plots_and_planets)

ttk.Label(left_panel, text="Select Planet:").pack(anchor='w')
planet_var = tk.StringVar()
planet_combo = ttk.Combobox(left_panel, textvariable=planet_var, state='readonly')
planet_combo.pack(fill='x')

ttk.Separator(left_panel).pack(fill='x', pady=10)
ttk.Label(left_panel, text="Actions", font="-weight bold").pack(pady=5)
# *** REMOVED THE PLOT BUTTONS ***
ttk.Button(left_panel, text="Animate Selected Orbit", command=calc_position).pack(pady=2, fill='x')
ttk.Button(left_panel, text="Show Distance to Earth", command=calc_distance_to_earth).pack(pady=2, fill='x')

ttk.Separator(left_panel).pack(fill='x', pady=10)
ttk.Button(left_panel, text="Exit", command=root.destroy).pack(side='bottom', pady=5, fill='x')

# Initialize the application state
if host_stars:
    star_var.set(host_stars[0])
    # Initial call to populate everything on startup
    update_plots_and_planets()

root.mainloop()

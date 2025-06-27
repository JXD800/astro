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
STEFAN_BOLTZMANN = 5.67e-8 # Stefan-Boltzmann constant W⋅m−2⋅K−4
SOLAR_TEMP = 5778         # Sun's effective temperature in Kelvin
SOLAR_RADIUS_M = 6.9634e8  # Sun's radius in meters

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
        'st_mass': 'star_mass', 'st_rad': 'star_radius', 'st_teff': 'star_temp',
        'st_spectype': 'star_type', 'sy_dist': 'star_distance'
    })
    return kepler_data

def stop_animation():
    """Stops any running Matplotlib animation."""
    if hasattr(canvas, 'ani') and canvas.ani and canvas.ani.event_source:
        canvas.ani.event_source.stop()
    canvas.ani = None

def clear_plot():
    """Clears the plot and stops any animation."""
    stop_animation()
    ax.clear()
    ax.set_facecolor('white')
    ax.set_title("Select a Plotting Option")
    ax.text(0.5, 0.5, "No Data Plotted", ha='center', va='center', fontsize=16, color='gray')
    canvas.draw()
    info_var.set("")

def plot_hr_diagram():
    """Plots a Hertzsprung-Russell diagram of the dataset and highlights the selected star."""
    stop_animation()
    ax.clear()

    # Filter data for stars with the necessary temperature and radius data
    hr_data = kepler_data.dropna(subset=['star_temp', 'star_radius']).copy()
    if hr_data.empty:
        messagebox.showinfo("No Data", "Not enough stellar data (temperature, radius) to plot an H-R Diagram.")
        clear_plot()
        return

    # Calculate luminosity relative to the Sun (L/L_sun = (R/R_sun)^2 * (T/T_sun)^4)
    hr_data['luminosity'] = (hr_data['star_radius']**2) * (hr_data['star_temp'] / SOLAR_TEMP)**4

    # Plot all stars in the dataset
    ax.scatter(hr_data['star_temp'], hr_data['luminosity'], alpha=0.3, s=5, label='Exoplanet Host Stars')

    # Highlight the currently selected star
    selected_star_name = star_var.get()
    selected_star_data = hr_data[hr_data['star_name'] == selected_star_name]
    if not selected_star_data.empty:
        star = selected_star_data.iloc[0]
        ax.scatter(star['star_temp'], star['luminosity'], color='red', s=50,
                    edgecolor='black', zorder=5, label=f'Selected: {selected_star_name}')

    # H-R Diagram styling
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # Reverse temperature axis (hot on the left)
    ax.set_title("Hertzsprung-Russell Diagram")
    ax.set_xlabel("Effective Temperature (K)")
    ax.set_ylabel("Luminosity (Solar Units)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    canvas.draw()
    info_var.set("H-R diagram shows star luminosity vs. temperature.")

def plot_star_spectrum():
    """Plots the star's black-body radiation spectrum."""
    selected_star = star_var.get()
    filtered = kepler_data[kepler_data['star_name'].str.lower() == selected_star.lower()]
    if filtered.empty or pd.isna(filtered.iloc[0]['star_temp']):
        messagebox.showinfo("No Data", f"Stellar temperature for {selected_star} is not available.")
        clear_plot()
        return

    stop_animation()
    ax.clear()
    
    T = filtered.iloc[0]['star_temp']
    wavelengths = np.linspace(100e-9, 3000e-9, 500)
    spectral_radiance = (2 * H_PLANCK * C_LIGHT**2 / wavelengths**5) / \
                        (np.exp(H_PLANCK * C_LIGHT / (wavelengths * K_BOLTZMANN * T)) - 1)

    max_radiance = np.max(spectral_radiance)
    normalized_radiance = spectral_radiance / max_radiance if max_radiance > 0 else spectral_radiance

    ax.plot(wavelengths * 1e9, normalized_radiance, color='orange', linewidth=2)
    ax.set_title(f"Black-Body Spectrum of {selected_star} (T≈{T:,.0f} K)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.1)
    canvas.draw()
    info_var.set("Theoretical black-body radiation curve for the star.")

def calc_position():
    """Calculates and animates the planet's orbit using Kepler's Equation."""
    selected_star = star_var.get()
    selected_planet = planet_var.get()
    
    filt = kepler_data[(kepler_data['star_name'] == selected_star) & (kepler_data['name'] == selected_planet)]
                       
    if filt.empty:
        messagebox.showerror("Data Not Found", f"Could not find data for planet '{selected_planet}' orbiting '{selected_star}'.")
        clear_plot()
        return

    p = filt.iloc[0]
    
    # --- ROBUST ERROR HANDLING for required animation data ---
    required_cols = ['semi_major_axis', 'orbital_period', 'eccentricity']
    missing_data = [col for col in required_cols if pd.isna(p[col])]
    if missing_data:
        messagebox.showerror("Animation Error", 
                             f"Cannot plot orbit for {p['name']}.\n"
                             f"Missing essential data: {', '.join(missing_data)}")
        clear_plot()
        return

    try:
        a = float(p['semi_major_axis'])
        period_days = float(p['orbital_period'])
        ecc = float(p['eccentricity'])

        stop_animation()
        ax.clear()
        
        b = a * np.sqrt(1 - ecc**2)
        theta = np.linspace(0, 2 * np.pi, 360)
        x_orb, y_orb = a * (np.cos(theta) - ecc), b * np.sin(theta)

        ax.plot(x_orb, y_orb, 'w--', alpha=0.5)
        ax.plot(0, 0, 'o', color='gold', markersize=12, label=f'Star ({p["star_name"]})')
        ax.set_title(f"Orbit of {p['name']}", fontsize=10)
        ax.set_facecolor("black")
        ax.set_xlim(-a * 1.5, a * 1.5)
        ax.set_ylim(-a * 1.5, a * 1.5)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', color='gray', alpha=0.4)
        planet_dot, = ax.plot([], [], 'o', color='cyan', markersize=6)
        
        period_seconds = period_days * 86400
        mean_motion = 2 * np.pi / period_seconds

        def update(frame_time: float) -> tuple:
            M = mean_motion * frame_time
            try:
                E = newton(lambda E: E - ecc * np.sin(E) - M, M, tol=1e-6)
            except RuntimeError: E = M
            
            x, y = a * (np.cos(E) - ecc), b * np.sin(E)
            planet_dot.set_data([x], [y])
            info_var.set(f"Time: {frame_time/86400:.1f} days | Distance: {np.sqrt(x**2 + y**2):.3f} AU")
            return planet_dot,

        canvas.ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, period_seconds, 360),
                                              interval=50, blit=True, repeat=True)
        canvas.draw()
    except (ValueError, TypeError) as e:
        messagebox.showerror("Animation Error", f"Could not plot orbit due to invalid data.\nDetails: {e}")
        clear_plot()

def search_stars(event=None):
    """Searches the dataframe and populates the research treeview."""
    for item in research_tree.get_children(): research_tree.delete(item)
    query = search_var.get().strip().lower()
    if not query: return

    mask = kepler_data['star_name'].str.lower().str.contains(query, na=False)
    results = kepler_data[mask].sort_values(by=['star_name', 'name'])

    if results.empty:
        research_tree.insert('', 'end', values=("No results found.", "", ""))
        return

    for _, row in results.iterrows():
        research_tree.insert('', 'end', values=(row.get('star_name', 'N/A'), row.get('name', 'N/A'), row.get('discovery_year', 'N/A')))

def on_tree_select(event=None):
    """Handles selection in the research tree to plot the orbit."""
    if not research_tree.selection(): return
    selected_item = research_tree.selection()[0]
    star_name, planet_name, _ = research_tree.item(selected_item, 'values')

    if star_name not in star_combo['values']: return

    # Update controls
    star_var.set(star_name)
    update_planets() # Manually update planet list
    planet_var.set(planet_name)

    # Switch to the orbit viewer and plot
    main_notebook.select(orbit_viewer_tab)
    calc_position()

def update_planets(event=None):
    """Updates the planet dropdown based on the selected star."""
    planets = sorted(kepler_data[kepler_data['star_name'] == star_var.get()]['name'].tolist())
    planet_combo['values'] = planets
    planet_var.set(planets[0] if planets else '')

# --- GUI Setup ---
kepler_data = load_kepler_data()
host_stars = sorted(kepler_data['star_name'].dropna().unique())

root = tk.Tk()
root.title("Exoplanet Data Viewer")
root.geometry("1280x800")

# Main frame
frame = ttk.Frame(root, padding=10)
frame.pack(fill='both', expand=True)

# Left control panel
left_panel = ttk.Frame(frame, width=250)
left_panel.pack(side='left', fill='y', padx=(0, 10))
left_panel.pack_propagate(False)

# Right panel with main notebook
right_panel = ttk.Frame(frame)
right_panel.pack(side='right', expand=True, fill='both')
main_notebook = ttk.Notebook(right_panel)
main_notebook.pack(expand=True, fill='both')

# --- Setup the single, reusable Matplotlib canvas and info bar ---
plot_frame = ttk.Frame(main_notebook)
canvas_frame = ttk.Frame(plot_frame)
canvas_frame.pack(fill='both', expand=True, pady=5)
info_var = tk.StringVar()
ttk.Label(plot_frame, textvariable=info_var, foreground="blue").pack(pady=5)
fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill='both', expand=True)
canvas.ani = None

# -- Tabs for visualization --
orbit_viewer_tab = ttk.Frame(main_notebook)
hr_diagram_tab = ttk.Frame(main_notebook)
spectrum_tab = ttk.Frame(main_notebook)

main_notebook.add(plot_frame, text="Plot Viewer")
main_notebook.add(hr_diagram_tab, text="H-R Diagram")
main_notebook.add(spectrum_tab, text="Black-Body Spectrum")

# -- Tab 3: Research --
research_tab = ttk.Frame(main_notebook, padding=10)
main_notebook.add(research_tab, text="Research")

search_frame = ttk.Frame(research_tab)
search_frame.pack(fill='x', pady=5)
ttk.Label(search_frame, text="Search Star:").pack(side='left')
search_var = tk.StringVar()
search_entry = ttk.Entry(search_frame, textvariable=search_var)
search_entry.pack(side='left', expand=True, fill='x', padx=5)
search_entry.bind("<Return>", search_stars)
ttk.Button(search_frame, text="Search", command=search_stars).pack(side='left')

tree_frame = ttk.Frame(research_tab)
tree_frame.pack(expand=True, fill='both')
cols = ('Star', 'Planet', 'Discovered')
research_tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
for col in cols:
    research_tree.heading(col, text=col)
    research_tree.column(col, width=150)
research_tree.bind('<<TreeviewSelect>>', on_tree_select)
tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=research_tree.yview)
research_tree.configure(yscrollcommand=tree_scroll.set)
tree_scroll.pack(side='right', fill='y')
research_tree.pack(side='left', expand=True, fill='both')

# --- Add controls to the left panel ---
ttk.Label(left_panel, text="Star & Planet Selection", font="-weight bold").pack(pady=(0, 5))
ttk.Label(left_panel, text="Select Star:").pack(anchor='w')
star_var = tk.StringVar()
star_combo = ttk.Combobox(left_panel, textvariable=star_var, values=host_stars)
star_combo.pack(fill='x', pady=(0,5))
star_combo.bind("<<ComboboxSelected>>", update_planets)

ttk.Label(left_panel, text="Select Planet:").pack(anchor='w')
planet_var = tk.StringVar()
planet_combo = ttk.Combobox(left_panel, textvariable=planet_var)
planet_combo.pack(fill='x')

ttk.Separator(left_panel).pack(fill='x', pady=10)
ttk.Label(left_panel, text="Plotting Actions", font="-weight bold").pack(pady=5)
ttk.Button(left_panel, text="Animate Planet Orbit", command=calc_position).pack(pady=2, fill='x')
ttk.Button(left_panel, text="Plot H-R Diagram", command=plot_hr_diagram).pack(pady=2, fill='x')
ttk.Button(left_panel, text="Plot Star Spectrum", command=plot_star_spectrum).pack(pady=2, fill='x')

ttk.Separator(left_panel).pack(fill='x', pady=10)
ttk.Button(left_panel, text="Exit", command=root.destroy).pack(side='bottom', pady=5, fill='x')

# Initialize the application state
if host_stars:
    star_var.set(host_stars[0])
    update_planets()
clear_plot() # Start with a clean slate

root.mainloop()

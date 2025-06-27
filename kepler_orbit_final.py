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

# --- Physical Constants (Global Scope) ---
# Using standard uppercase notation for constants.
H_PLANCK = 6.626e-34  # Planck's constant
C_LIGHT = 3e8         # Speed of light
K_BOLTZMANN = 1.381e-23 # Boltzmann constant

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
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Network Error", f"Failed to fetch data: {e}")
        return pd.DataFrame() # Return empty dataframe on error

    csv_data = response.content.decode("utf-8")
    
    # Use assignment instead of `inplace=True` for modern Pandas best practices.
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
    if pd.isna(temp):
        return "Unknown"
    try:
        T = float(temp)
        if T > 30000: return "O (blue)"
        elif T > 10000: return "B (blue-white)"
        elif T > 7500: return "A (white)"
        elif T > 6000: return "F (yellow-white)"
        elif T > 5200: return "G (yellow)"
        elif T > 3700: return "K (orange)"
        else: return "M (red)"
    except (ValueError, TypeError):
        return "Unknown"

def stop_animation():
    """Stops any running Matplotlib animation."""
    if canvas.ani and canvas.ani.event_source:
        canvas.ani.event_source.stop()
        canvas.ani = None

def plot_star_spectrum():
    """Plots the star's black-body radiation spectrum on the shared canvas."""
    selected_star = star_var.get()
    filtered = kepler_data[kepler_data['star_name'].str.lower() == selected_star.lower()]
    if filtered.empty:
        messagebox.showerror("Error", "Star not found.")
        return

    temp = filtered.iloc[0]['star_temp']
    if pd.isna(temp) or temp <= 0:
        messagebox.showinfo("No Data", f"Stellar temperature for {selected_star} is not available.")
        return

    stop_animation()
    ax.clear()

    T = float(temp)
    wavelengths = np.linspace(100e-9, 3000e-9, 500) # Wavelengths in meters
    
    # Calculate Planck's law for black-body radiation
    spectral_radiance = (2 * H_PLANCK * C_LIGHT**2 / wavelengths**5) / \
                        (np.exp(H_PLANCK * C_LIGHT / (wavelengths * K_BOLTZMANN * T)) - 1)

    # --- FIX: NORMALIZE THE DATA ---
    # This scales the plot so the peak is always at 1.0 for clear visualization.
    # It prevents the y-axis from becoming excessively large.
    max_radiance = np.max(spectral_radiance)
    if max_radiance > 0:
        normalized_radiance = spectral_radiance / max_radiance
    else:
        normalized_radiance = spectral_radiance # Avoid division by zero if all values are zero

    # Plot the FIXED normalized data
    ax.plot(wavelengths * 1e9, normalized_radiance, color='orange', linewidth=2)
    ax.set_title(f"Black-Body Spectrum of {selected_star} (T≈{T:,.0f} K)", fontsize=10)
    ax.set_xlabel("Wavelength (nm)")
    
    # --- FIX: UPDATE THE Y-AXIS LABEL ---
    ax.set_ylabel("Normalized Intensity")
    ax.set_facecolor("white")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- FIX: Set clean Y-axis limits ---
    ax.set_ylim(0, 1.1)

    canvas.draw()
    output.delete('1.0', tk.END)
    output.insert(tk.END, f"[INFO] Displaying stellar spectrum for {selected_star}.\n"
                          f"This plot shows the theoretical black-body radiation curve based on the star's effective temperature. "
                          f"The peak of this curve determines the star's apparent color.")

def calc_distance_to_earth():
    """Calculates and displays the distance to the selected star."""
    selected_star = star_var.get()
    filtered = kepler_data[kepler_data['star_name'].str.lower() == selected_star.lower()]
    if filtered.empty:
        messagebox.showerror("Error", "Star not found.")
        return
        
    dist = filtered.iloc[0]['star_distance'] # Distance is in parsecs
    if pd.notna(dist):
        dist_ly = dist * 3.26156 # Convert parsecs to light-years
        msg = f"Distance to {selected_star}: {dist:.2f} parsecs ({dist_ly:.2f} light-years)"
    else:
        msg = f"Distance to {selected_star}: Unknown"
        
    messagebox.showinfo("Distance to Earth", msg)

def calc_position():
    """Calculates and animates the planet's orbit using Kepler's Equation."""
    selected_star = star_var.get()
    selected_planet = planet_var.get()
    
    filt = kepler_data[(kepler_data['star_name'].str.lower() == selected_star.lower()) &
                       (kepler_data['name'].str.lower() == selected_planet.lower())]
                       
    if filt.empty:
        messagebox.showerror("Data Not Found", f"Could not find data for planet '{selected_planet}' orbiting '{selected_star}'.")
        return

    p = filt.iloc[0]
    output.delete('1.0', tk.END)
    output.insert(tk.END, f"[EXOPLANET]\nName: {p['name']}\nSemi-Major Axis: {p.get('semi_major_axis', 'N/A')} AU\n")
    output.insert(tk.END, f"Period: {p.get('orbital_period', 'N/A')} days\nRadius: {p.get('radius', 'N/A')} Jupiter Radii\nMass: {p.get('mass', 'N/A')} Jupiter Masses\n")
    output.insert(tk.END, f"Eccentricity: {p.get('eccentricity', 'N/A')}\n\n")
    output.insert(tk.END, f"[STAR]\nName: {p['star_name']}\nTemp: {p['star_temp']} K\nSpectral Type: {p.get('star_type', 'N/A')}\n")
    output.insert(tk.END, f"HR Class: {get_hr_position(p['star_temp'])}\n")

    try:
        a = float(p['semi_major_axis'])
        period_days = float(p['orbital_period'])
        ecc = float(p['eccentricity']) if pd.notna(p['eccentricity']) else 0.0

        stop_animation()
        ax.clear()
        
        b = a * np.sqrt(1 - ecc**2)  # Semi-minor axis
        theta = np.linspace(0, 2 * np.pi, 360)
        x_orb = a * np.cos(theta) - a * ecc
        y_orb = b * np.sin(theta)

        ax.plot(x_orb, y_orb, 'w--', alpha=0.5, label='Orbit Path')
        ax.plot(0, 0, 'o', color='gold', markersize=12, label=f'Star ({p["star_name"]})')
        ax.set_title(f"Orbit of {p['name']}", fontsize=10)
        ax.set_facecolor("black")
        ax.set_xlim(-a * 1.5, a * 1.5)
        ax.set_ylim(-a * 1.5, a * 1.5)
        ax.set_xlabel("Distance (AU)")
        ax.set_ylabel("Distance (AU)")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', color='gray', alpha=0.4)
        planet_dot, = ax.plot([], [], 'o', color='cyan', markersize=6, label=f'Planet ({p["name"]})')
        dist_line, = ax.plot([], [], '--', color='red', alpha=0.7)
        ax.legend(fontsize='small')
        
        period_seconds = period_days * 86400
        mean_motion = 2 * np.pi / period_seconds

        def update(frame_time: float) -> tuple:
            """SCIENTIFICALLY ACCURATE UPDATE using Kepler's Equation."""
            M = mean_motion * frame_time # Mean Anomaly
            # Solve Kepler's Equation E - e*sin(E) = M for E (Eccentric Anomaly)
            try:
                E = newton(lambda E: E - ecc * np.sin(E) - M, M, tol=1e-6)
            except RuntimeError: # Solver might fail for extreme values
                E = M
            
            # Calculate planet coordinates from Eccentric Anomaly
            x = a * (np.cos(E) - ecc)
            y = b * np.sin(E)
            
            planet_dot.set_data([x], [y])
            dist_line.set_data([0, x], [0, y])
            info_var.set(f"Time: {frame_time/86400:.1f} days | Distance to star: {np.sqrt(x**2 + y**2):.3f} AU")
            return planet_dot, dist_line

        # Create and store the animation object to prevent garbage collection
        canvas.ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, period_seconds, 360),
                                              interval=50, blit=True, repeat=True)
        canvas.draw()

    except (ValueError, TypeError, KeyError) as e:
        output.insert(tk.END, f"\n[ERROR] Could not plot orbit: {e}.\nMissing or invalid data (e.g., semi-major axis, period).")

def search_stars():
    """Searches the dataframe and populates the research treeview."""
    query = search_var.get().strip().lower()
    
    # Clear previous results from the treeview
    for item in research_tree.get_children():
        research_tree.delete(item)

    if not query:
        return

    # Filter data based on star name
    mask = kepler_data['star_name'].str.lower().str.contains(query, na=False)
    results = kepler_data[mask].sort_values(by=['star_name', 'name'])

    if results.empty:
        research_tree.insert('', 'end', values=("No results found.", "", "", "", ""))
        return

    # Populate the treeview with search results
    for _, row in results.iterrows():
        period = f"{row.get('orbital_period', 0):.2f}" if pd.notna(row.get('orbital_period')) else 'N/A'
        axis = f"{row.get('semi_major_axis', 0):.3f}" if pd.notna(row.get('semi_major_axis')) else 'N/A'
        
        research_tree.insert('', 'end', values=(
            row.get('star_name', 'N/A'),
            row.get('name', 'N/A'),
            period,
            axis,
            row.get('discovery_year', 'N/A')
        ))

def update_planets(event=None):
    """Updates the planet dropdown based on the selected star."""
    selected = star_var.get()
    planets = sorted(kepler_data[kepler_data['star_name'].str.lower() == selected.lower()]['name'].tolist())
    planet_combo['values'] = planets
    if planets:
        planet_var.set(planets[0])
    else:
        planet_var.set('')

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

# Right panel with tabs
right_panel = ttk.Frame(frame)
right_panel.pack(side='right', expand=True, fill='both')
notebook = ttk.Notebook(right_panel)
notebook.pack(expand=True, fill='both')

# -- Tab 1: Orbit Viewer --
viewer_tab = ttk.Frame(notebook, padding=5)
notebook.add(viewer_tab, text="Orbit Viewer")

output = tk.Text(viewer_tab, height=10)
output.pack(fill='x', pady=(0, 5))
canvas_frame = ttk.Frame(viewer_tab)
canvas_frame.pack(fill='both', expand=True)
info_var = tk.StringVar()
ttk.Label(viewer_tab, textvariable=info_var, foreground="blue").pack(pady=(5, 0))

# --- Setup the single, reusable Matplotlib canvas ---
fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill='both', expand=True)
canvas.ani = None  # Initialize animation reference

# -- Tab 2: Research --
research_tab = ttk.Frame(notebook, padding=10)
notebook.add(research_tab, text="Research")

search_frame = ttk.Frame(research_tab)
search_frame.pack(fill='x', pady=5)
ttk.Label(search_frame, text="Search Star Name:").pack(side='left', padx=(0, 5))
search_var = tk.StringVar()
search_entry = ttk.Entry(search_frame, textvariable=search_var)
search_entry.pack(side='left', expand=True, fill='x')
search_entry.bind("<Return>", lambda event: search_stars())
search_button = ttk.Button(search_frame, text="Search", command=search_stars)
search_button.pack(side='left', padx=(5, 0))

tree_frame = ttk.Frame(research_tab)
tree_frame.pack(expand=True, fill='both')

cols = ('Star', 'Planet', 'Period (days)', 'Axis (AU)', 'Discovered')
research_tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
for col in cols:
    research_tree.heading(col, text=col)
    research_tree.column(col, width=120, anchor='center')

tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=research_tree.yview)
research_tree.configure(yscrollcommand=tree_scroll.set)
tree_scroll.pack(side='right', fill='y')
research_tree.pack(side='left', expand=True, fill='both')

# --- Add controls to the left panel ---
ttk.Label(left_panel, text="Select Star:", font="-weight bold").pack(pady=(0, 2))
star_var = tk.StringVar()
star_combo = ttk.Combobox(left_panel, textvariable=star_var, values=host_stars)
star_combo.pack(fill='x')

ttk.Label(left_panel, text="Select Planet:", font="-weight bold").pack(pady=(10, 2))
planet_var = tk.StringVar()
planet_combo = ttk.Combobox(left_panel, textvariable=planet_var)
planet_combo.pack(fill='x', pady=(0, 10))

star_combo.bind("<<ComboboxSelected>>", update_planets)

ttk.Button(left_panel, text="Animate Orbit", command=calc_position).pack(pady=5, fill='x')
ttk.Button(left_panel, text="Plot Star Spectrum", command=plot_star_spectrum).pack(pady=5, fill='x')
ttk.Button(left_panel, text="Show Distance to Earth", command=calc_distance_to_earth).pack(pady=5, fill='x')
ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)
ttk.Button(left_panel, text="Exit", command=root.destroy).pack(pady=5, fill='x')

if host_stars:
    star_var.set(host_stars[0])
    update_planets()

root.mainloop()

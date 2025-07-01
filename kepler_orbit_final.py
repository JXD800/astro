import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
from io import StringIO
from typing import Optional, List
import json

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
        messagebox.showerror("Network Error", f"Failed to fetch PS data: {e}")
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

# --- UPDATED ATMOSPHERE DATA FETCHING ---
def fetch_atmosphere_spectroscopy(planet_name: str) -> dict:
    """
    Fetches atmospheric spectroscopy data from NASA's Firefly service.
    Returns a dictionary with spectroscopy data and analysis.
    """
    # The correct URL for atmospheric spectroscopy data
    firefly_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly?atmospheres"
    
    print(f"🔗 Fetching atmospheric data from: {firefly_url}")
    print(f"🪐 Target planet: {planet_name}")
    
    try:
        # Try to get atmospheric data from the Firefly service
        response = requests.get(firefly_url, timeout=30)
        response.raise_for_status()
        
        # Parse the response (assuming it's JSON or text format)
        content = response.text
        
        # Also try the TAP service for molecular data as backup
        encoded_planet_name = requests.utils.quote(planet_name)
        tap_url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
            f"query=select+*+from+molecules+where+pl_name='{encoded_planet_name}'&format=csv"
        )
        
        print(f"🔗 Backup molecular query: {tap_url}")
        
        tap_response = requests.get(tap_url, timeout=20)
        molecules_data = None
        
        if tap_response.status_code == 200:
            molecules_df = pd.read_csv(StringIO(tap_response.content.decode("utf-8")))
            if not molecules_df.empty:
                molecules_data = molecules_df.to_dict('records')
        
        return {
            'firefly_url': firefly_url,
            'firefly_content': content,
            'molecules_data': molecules_data,
            'status': 'success'
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching atmospheric data: {e}")
        return {
            'firefly_url': firefly_url,
            'error': str(e),
            'status': 'error'
        }

def create_atmosphere_popup(planet_name: str, atmosphere_data: dict):
    """
    Creates a large popup window displaying atmospheric spectroscopy data.
    """
    popup = tk.Toplevel(root)
    popup.title(f"Atmospheric Analysis - {planet_name}")
    popup.geometry("900x700")
    popup.configure(bg='#1a1a2e')
    
    # Make it modal
    popup.transient(root)
    popup.grab_set()
    
    # Center the popup
    popup.update_idletasks()
    x = (popup.winfo_screenwidth() // 2) - (900 // 2)
    y = (popup.winfo_screenheight() // 2) - (700 // 2)
    popup.geometry(f"900x700+{x}+{y}")
    
    # Main frame with padding
    main_frame = tk.Frame(popup, bg='#1a1a2e', padx=20, pady=20)
    main_frame.pack(fill='both', expand=True)
    
    # Title
    title_label = tk.Label(
        main_frame, 
        text=f"🌌 ATMOSPHERIC SPECTROSCOPY ANALYSIS",
        font=('Arial', 16, 'bold'),
        fg='#00d4ff',
        bg='#1a1a2e'
    )
    title_label.pack(pady=(0, 10))
    
    planet_label = tk.Label(
        main_frame,
        text=f"Planet: {planet_name}",
        font=('Arial', 14, 'bold'),
        fg='#ffffff',
        bg='#1a1a2e'
    )
    planet_label.pack(pady=(0, 20))
    
    # Create notebook for different data sections
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill='both', expand=True, pady=(0, 20))
    
    # --- TAB 1: DATA SOURCES ---
    sources_frame = tk.Frame(notebook, bg='#2d2d44')
    notebook.add(sources_frame, text="🔗 Data Sources")
    
    sources_text = tk.Text(
        sources_frame,
        wrap=tk.WORD,
        font=('Consolas', 10),
        bg='#1e1e1e',
        fg='#00ff00',
        insertbackground='white',
        height=15
    )
    sources_scrollbar = tk.Scrollbar(sources_frame, command=sources_text.yview)
    sources_text.configure(yscrollcommand=sources_scrollbar.set)
    
    sources_info = f"""
🌐 PRIMARY DATA SOURCE:
NASA Exoplanet Archive - Atmospheric Spectroscopy Service
URL: {atmosphere_data.get('firefly_url', 'N/A')}

📡 FIREFLY SERVICE STATUS: {atmosphere_data.get('status', 'Unknown').upper()}

🔬 MOLECULAR DATABASE QUERY:
TAP Service for molecular detections
Query: SELECT * FROM molecules WHERE pl_name='{planet_name}'

📊 DATA RETRIEVAL TIMESTAMP: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

🎯 TARGET: {planet_name}
    """
    
    sources_text.insert('1.0', sources_info)
    sources_text.configure(state='disabled')
    
    sources_scrollbar.pack(side='right', fill='y')
    sources_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
    
    # --- TAB 2: SPECTROSCOPY DATA ---
    spectro_frame = tk.Frame(notebook, bg='#2d2d44')
    notebook.add(spectro_frame, text="📊 Spectroscopy Data")
    
    spectro_text = tk.Text(
        spectro_frame,
        wrap=tk.WORD,
        font=('Consolas', 9),
        bg='#1e1e1e',
        fg='#ffffff',
        insertbackground='white'
    )
    spectro_scrollbar = tk.Scrollbar(spectro_frame, command=spectro_text.yview)
    spectro_text.configure(yscrollcommand=spectro_scrollbar.set)
    
    if atmosphere_data.get('status') == 'success':
        firefly_content = atmosphere_data.get('firefly_content', '')
        spectro_info = f"""
🔬 FIREFLY SERVICE RESPONSE:
{'='*60}

{firefly_content[:2000]}{"..." if len(firefly_content) > 2000 else ""}

{'='*60}

📝 ANALYSIS NOTES:
• This data comes from NASA's atmospheric spectroscopy database
• Firefly service provides transmission/emission spectra when available
• Data includes wavelength-dependent opacity measurements
• Molecular absorption features can be identified from spectral lines
        """
    else:
        spectro_info = f"""
❌ SPECTROSCOPY DATA UNAVAILABLE

Error: {atmosphere_data.get('error', 'Unknown error')}

🔍 POSSIBLE REASONS:
• Planet not in atmospheric database
• No spectroscopic observations available
• Service temporarily unavailable
• Planet name formatting mismatch

💡 ALTERNATIVE APPROACH:
Try searching for this planet in published literature or 
check if it has been observed by space telescopes like:
• Hubble Space Telescope
• James Webb Space Telescope  
• Spitzer Space Telescope
        """
    
    spectro_text.insert('1.0', spectro_info)
    spectro_text.configure(state='disabled')
    
    spectro_scrollbar.pack(side='right', fill='y')
    spectro_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
    
    # --- TAB 3: MOLECULAR ANALYSIS ---
    molecules_frame = tk.Frame(notebook, bg='#2d2d44')
    notebook.add(molecules_frame, text="🧬 Molecular Detections")
    
    molecules_text = tk.Text(
        molecules_frame,
        wrap=tk.WORD,
        font=('Consolas', 10),
        bg='#1e1e1e',
        fg='#ffaa00',
        insertbackground='white'
    )
    molecules_scrollbar = tk.Scrollbar(molecules_frame, command=molecules_text.yview)
    molecules_text.configure(yscrollcommand=molecules_scrollbar.set)
    
    molecules_data = atmosphere_data.get('molecules_data')
    if molecules_data:
        molecules_info = f"""
🧪 DETECTED MOLECULES FOR {planet_name}:
{'='*50}

"""
        for i, mol_record in enumerate(molecules_data, 1):
            molecules_info += f"#{i}. {mol_record}\n\n"
        
        # Check for key molecules
        all_molecules = str(molecules_data).upper()
        has_h2o = 'H2O' in all_molecules or 'WATER' in all_molecules
        has_co2 = 'CO2' in all_molecules or 'CARBON DIOXIDE' in all_molecules
        has_ch4 = 'CH4' in all_molecules or 'METHANE' in all_molecules
        has_o2 = 'O2' in all_molecules or 'OXYGEN' in all_molecules
        
        molecules_info += f"""
{'='*50}
🔬 KEY ATMOSPHERIC COMPONENTS ANALYSIS:

💧 Water Vapor (H₂O):     {'✅ DETECTED' if has_h2o else '❌ NOT DETECTED'}
🌫️  Carbon Dioxide (CO₂): {'✅ DETECTED' if has_co2 else '❌ NOT DETECTED'}  
🔥 Methane (CH₄):         {'✅ DETECTED' if has_ch4 else '❌ NOT DETECTED'}
💨 Oxygen (O₂):           {'✅ DETECTED' if has_o2 else '❌ NOT DETECTED'}

🌡️ HABITABILITY INDICATORS:
• Presence of H₂O suggests potential for liquid water
• CO₂ indicates atmospheric greenhouse effects
• CH₄ could suggest biological or geological activity
• O₂ might indicate photosynthesis (very rare!)
        """
    else:
        molecules_info = f"""
🔍 MOLECULAR DETECTION STATUS: NO DATA AVAILABLE

❌ No molecular detections found in NASA database for {planet_name}

🌟 IMPORTANT NOTES:
• Most exoplanets lack detailed atmospheric characterization
• Only ~200 planets have atmospheric data in NASA archives
• Atmospheric analysis requires transit observations
• Molecular detection is cutting-edge science

🔬 DETECTION METHODS:
• Transit spectroscopy (planet passes in front of star)
• Emission spectroscopy (thermal emission from planet)
• Direct imaging (extremely rare)

🚀 FUTURE PROSPECTS:
• James Webb Space Telescope is revolutionizing this field
• Next-generation ground telescopes will detect more molecules
• New space missions planned for atmospheric characterization
        """
    
    molecules_text.insert('1.0', molecules_info)
    molecules_text.configure(state='disabled')
    
    molecules_scrollbar.pack(side='right', fill='y')
    molecules_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
    
    # Close button
    close_button = tk.Button(
        main_frame,
        text="🚀 CLOSE ANALYSIS",
        command=popup.destroy,
        font=('Arial', 12, 'bold'),
        bg='#ff4757',
        fg='white',
        padx=20,
        pady=10,
        relief='raised',
        borderwidth=2
    )
    close_button.pack(pady=10)
    
    # Focus the popup
    popup.focus_set()

# --- UPDATED ATMOSPHERE CHECK FUNCTION ---
def check_atmosphere():
    """Updated controller function called by the atmosphere button."""
    planet_name = planet_var.get()
    if not planet_name:
        messagebox.showwarning("No Planet Selected", "Please select a star and a planet first.")
        return

    # Show loading message
    loading_msg = messagebox.showinfo(
        "Fetching Atmospheric Data", 
        f"🔍 Querying NASA atmospheric databases for {planet_name}...\n\n"
        f"📡 Accessing Firefly spectroscopy service\n"
        f"🧬 Checking molecular detection database\n\n"
        f"This may take a moment...", 
        icon='info'
    )
    
    # Update the GUI to show the loading message
    root.update()
    
    # Fetch the atmospheric data
    atmosphere_data = fetch_atmosphere_spectroscopy(planet_name)
    
    # Create and show the detailed popup
    create_atmosphere_popup(planet_name, atmosphere_data)

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
    planets = sorted(kepler_data[kepler_data['star_name'] == star_var.get()]['name'].tolist())
    planet_combo['values'] = planets
    if planets:
        planet_var.set(planets[0])
    else:
        planet_var.set('')
    
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
star_combo.bind("<<ComboboxSelected>>", update_plots_and_planets)

ttk.Label(left_panel, text="Select Planet:").pack(anchor='w')
planet_var = tk.StringVar()
planet_combo = ttk.Combobox(left_panel, textvariable=planet_var, state='readonly')
planet_combo.pack(fill='x')

ttk.Separator(left_panel).pack(fill='x', pady=10)
ttk.Label(left_panel, text="Actions", font="-weight bold").pack(pady=5)
ttk.Button(left_panel, text="Animate Selected Orbit", command=calc_position).pack(pady=2, fill='x')
ttk.Button(left_panel, text="Show Distance to Earth", command=calc_distance_to_earth).pack(pady=2, fill='x')
# *** UPDATED BUTTON FOR ATMOSPHERE CHECK ***
ttk.Button(left_panel, text="🧬 Check Atmosphere Composition", command=check_atmosphere).pack(pady=2, fill='x')

ttk.Separator(left_panel).pack(fill='x', pady=10)
ttk.Button(left_panel, text="Exit", command=root.destroy).pack(side='bottom', pady=5, fill='x')

# Initialize the application state
if host_stars:
    star_var.set(host_stars[0])
    update_plots_and_planets()

root.mainloop()

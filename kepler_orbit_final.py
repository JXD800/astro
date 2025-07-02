import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
from io import StringIO
from typing import Optional, List, Dict
import logging
from functools import lru_cache

# Matplotlib and SciPy imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import newton

# Configure logging
logging.basicConfig(filename='exoplanet_tool.log', level=logging.INFO)

# --- Constants ---
H_PLANCK = 6.626e-34
C_LIGHT = 3e8
K_BOLTZMANN = 1.381e-23
SOLAR_TEMP = 5778
CACHE_SIZE = 100  # Number of API responses to cache

# --- Enhanced Data Loading with Caching ---
@lru_cache(maxsize=CACHE_SIZE)
def fetch_complete_atmospheric_data() -> pd.DataFrame:
    """Fetches and combines planetary, atmospheric, and molecular data"""
    try:
        logging.info("Fetching exoplanet data from NASA API")
        
        # Main planetary data
        base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        params = {
            'query': "select pl_name,hostname,pl_orbper,pl_orbsmax,pl_radj,pl_bmassj,"
                    "disc_year,pl_orbeccen,st_mass,st_rad,st_age,st_teff,"
                    "st_spectype,sy_dist,disc_facility,disposition from ps",
            'format': 'csv'
        }
        kepler_df = pd.read_csv(f"{base_url}?{requests.compat.urlencode(params)}")
        
        # Rename columns consistently
        kepler_df = kepler_df.rename(columns={
            'pl_name': 'name', 'hostname': 'star_name', 'pl_orbper': 'orbital_period',
            'pl_orbsmax': 'semi_major_axis', 'pl_radj': 'radius', 'pl_bmassj': 'mass',
            'disc_year': 'discovery_year', 'pl_orbeccen': 'eccentricity',
            'st_mass': 'star_mass', 'st_rad': 'star_radius', 'st_age': 'star_age',
            'st_teff': 'star_temp', 'st_spectype': 'star_type', 'sy_dist': 'star_distance'
        })

        # Atmospheric properties
        atm_df = pd.read_csv(f"{base_url}?query=select+*+from+atmospheres&format=csv")
        
        # Molecular composition data
        mol_df = pd.read_csv(f"{base_url}?query=select+*+from+molecules&format=csv")
        
        # Combine all datasets
        merged_df = pd.merge(kepler_df, atm_df, how='left', left_on='name', right_on='pl_name')
        merged_df = pd.merge(merged_df, mol_df, how='left', left_on='name', right_on='pl_name')
        
        return merged_df.drop_duplicates()
    
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        messagebox.showerror("Data Error", f"Failed to load data: {str(e)}")
        return pd.DataFrame()

def fetch_real_spectroscopy_data(planet_name: str) -> Optional[pd.DataFrame]:
    """Fetches transmission spectroscopy data with error handling"""
    try:
        encoded_name = requests.utils.quote(planet_name)
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
            f"query=select+wavelength,flux,flux_err,flux_unit,wavelength_unit,"
            f"instrument,facility,reference+from+transmissionspec+"
            f"where+pl_name='{encoded_name}'&format=csv"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        csv_data = response.content.decode("utf-8")
        if not csv_data.strip() or "empty" in csv_data.lower():
            return None
            
        df = pd.read_csv(StringIO(csv_data))
        return df[df['flux'].notna()]  # Filter out null flux values
        
    except Exception as e:
        logging.warning(f"Spectroscopy data fetch failed for {planet_name}: {str(e)}")
        return None

# --- Enhanced Atmospheric Analysis ---
def plot_complete_atmosphere(planet_name: str):
    """Creates comprehensive atmospheric analysis window"""
    try:
        popup = tk.Toplevel(root)
        popup.title(f"Atmospheric Analysis - {planet_name}")
        popup.geometry("1200x800")
        
        # Notebook for multiple analysis tabs
        atm_notebook = ttk.Notebook(popup)
        atm_notebook.pack(expand=True, fill='both')
        
        # Tab 1: Transmission Spectrum
        spec_tab = ttk.Frame(atm_notebook)
        fig_spec = Figure(figsize=(10, 6))
        ax_spec = fig_spec.add_subplot(111)
        
        # Fetch and plot spectral data
        spec_data = fetch_real_spectroscopy_data(planet_name)
        if spec_data is not None and not spec_data.empty:
            for instrument, group in spec_data.groupby('instrument'):
                if 'flux_err' in group.columns:
                    ax_spec.errorbar(
                        group['wavelength'], group['flux'], 
                        yerr=group['flux_err'],
                        fmt='o-', label=f"{instrument} ({group['facility'].iloc[0]})",
                        capsize=3
                    )
                else:
                    ax_spec.plot(
                        group['wavelength'], group['flux'], 
                        'o-', label=f"{instrument} ({group['facility'].iloc[0]})"
                    )
            
            ax_spec.set_title(f"Transmission Spectrum - {planet_name}")
            ax_spec.set_xlabel(f"Wavelength ({group['wavelength_unit'].iloc[0]})")
            ax_spec.set_ylabel(f"Transit Depth ({group['flux_unit'].iloc[0]})")
            ax_spec.legend()
            ax_spec.grid(True)
        else:
            ax_spec.text(0.5, 0.5, "No spectral data available", ha='center')
        
        canvas_spec = FigureCanvasTkAgg(fig_spec, spec_tab)
        canvas_spec.get_tk_widget().pack(expand=True, fill='both')
        atm_notebook.add(spec_tab, text="Transmission Spectrum")
        
        # Tab 2: Molecular Abundances
        mol_tab = ttk.Frame(atm_notebook)
        fig_mol = Figure(figsize=(10, 6))
        ax_mol = fig_mol.add_subplot(111)
        
        # Get molecular data for this planet
        planet_data = kepler_data[kepler_data['name'] == planet_name]
        if planet_data is not None and not planet_data.empty and 'molecule' in planet_data.columns:
            molecules = planet_data['molecule'].dropna().unique()
            if len(molecules) > 0:
                abundances = [
                    planet_data[planet_data['molecule'] == m]['abundance'].mean() 
                    for m in molecules
                ]
                ax_mol.bar(molecules, abundances)
                ax_mol.set_title("Molecular Abundances")
                ax_mol.set_ylabel("Abundance (ppm)")
                ax_mol.tick_params(axis='x', rotation=45)
            else:
                ax_mol.text(0.5, 0.5, "No molecular data available", ha='center')
        else:
            ax_mol.text(0.5, 0.5, "No molecular data available", ha='center')
        
        canvas_mol = FigureCanvasTkAgg(fig_mol, mol_tab)
        canvas_mol.get_tk_widget().pack(expand=True, fill='both')
        atm_notebook.add(mol_tab, text="Molecular Composition")
        
        # Tab 3: Atmospheric Properties
        table_tab = ttk.Frame(atm_notebook)
        columns = ['atm_mass', 'atm_comp', 'atm_measurement', 'atm_altitude']
        
        # Create treeview with scrollbar
        tree_frame = ttk.Frame(table_tab)
        tree_frame.pack(expand=True, fill='both')
        
        atm_table = ttk.Treeview(tree_frame, columns=columns, show='headings')
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=atm_table.yview)
        atm_table.configure(yscrollcommand=scrollbar.set)
        
        for col in columns:
            atm_table.heading(col, text=col.replace('_', ' ').title())
            atm_table.column(col, width=150, anchor='center')
        
        if not planet_data.empty:
            for _, row in planet_data.iterrows():
                if pd.notna(row.get('atm_mass')):
                    atm_table.insert('', 'end', values=[row.get(col, 'N/A') for col in columns])
        
        scrollbar.pack(side="right", fill="y")
        atm_table.pack(side="left", expand=True, fill="both")
        
        # Add tooltip functionality
        def create_tooltip(widget, text):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()
            
            def motion(event):
                tooltip.geometry(f"+{event.x_root+20}+{event.y_root+10}")
            
            widget.bind("<Motion>", motion)
            widget.bind("<Leave>", lambda e: tooltip.destroy())
        
        # Example tooltip
        create_tooltip(atm_table, "Atmospheric properties from NASA Exoplanet Archive")
        
        atm_notebook.add(table_tab, text="Atmospheric Properties")
        
    except Exception as e:
        logging.error(f"Atmosphere plot failed for {planet_name}: {str(e)}")
        messagebox.showerror("Error", f"Failed to create atmospheric analysis: {str(e)}")

# --- Core Analysis Functions ---
def cleanup_animation():
    """Properly cleans up existing animation"""
    if hasattr(canvas_orbit, 'ani'):
        canvas_orbit.ani.event_source.stop()
        del canvas_orbit.ani

def plot_hr_diagram():
    """Plots Hertzsprung-Russell diagram with filtered data"""
    try:
        ax_hr.clear()
        filtered_data = apply_filters()
        hr_data = filtered_data.dropna(subset=['star_temp', 'star_radius']).copy()
        
        if hr_data.empty:
            ax_hr.text(0.5, 0.5, "Not Enough Data for H-R Diagram", ha='center', va='center')
            canvas_hr.draw()
            return

        hr_data['luminosity'] = (hr_data['star_radius']**2) * (hr_data['star_temp'] / SOLAR_TEMP)**4
        ax_hr.scatter(hr_data['star_temp'], hr_data['luminosity'], alpha=0.3, s=5, label='Exoplanet Host Stars')

        selected_star_data = hr_data[hr_data['star_name'] == star_var.get()]
        if not selected_star_data.empty:
            star = selected_star_data.iloc[0]
            ax_hr.scatter(star['star_temp'], star['luminosity'], color='red', s=50, 
                          edgecolor='black', zorder=5, label=f'Selected: {star_var.get()}')

        ax_hr.set_yscale('log')
        ax_hr.set_xscale('log')
        ax_hr.set_xlim(ax_hr.get_xlim()[::-1])
        ax_hr.set_title("Hertzsprung-Russell Diagram")
        ax_hr.set_xlabel("Effective Temperature (K)")
        ax_hr.set_ylabel("Luminosity (Solar Units)")
        ax_hr.grid(True, which="both", ls="--", alpha=0.5)
        ax_hr.legend()
        canvas_hr.draw()
        
    except Exception as e:
        logging.error(f"HR Diagram failed: {str(e)}")
        messagebox.showerror("Error", f"Failed to plot HR Diagram: {str(e)}")

def apply_filters() -> pd.DataFrame:
    """Applies user-selected filters to the dataset"""
    filtered = kepler_data.copy()
    try:
        if show_atmosphere.get():
            filtered = filtered[filtered['atm_mass'].notna()]
        if show_confirmed.get() and 'disposition' in filtered.columns:
            filtered = filtered[filtered['disposition'] == 'Confirmed']
        return filtered
    except Exception as e:
        logging.warning(f"Filter application failed: {str(e)}")
        return kepler_data

# [Rest of the functions (plot_star_spectrum, calc_position, etc.) remain the same as previous version...]

# --- Enhanced GUI Setup ---
root = tk.Tk()
root.title("NASA Exoplanet Archive - Atmospheric Analysis Tool")
root.geometry("1280x900")
root.minsize(1000, 700)  # Set minimum window size

# Configure grid weights for responsive layout
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Load data with atmospheric properties
kepler_data = fetch_complete_atmospheric_data()
host_stars = sorted(kepler_data['star_name'].dropna().unique())

# [Rest of the GUI setup remains the same as previous version...]

# Initialize with first star
if host_stars:
    star_var.set(host_stars[0])
    update_plots_and_planets()

root.mainloop()

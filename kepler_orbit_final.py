import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import newton
import logging

# Scientific Constants
H_PLANCK = 6.626e-34
C_LIGHT = 3e8
K_BOLTZMANN = 1.381e-23
SOLAR_TEMP = 5778

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('exoplanet_analyzer.log')]
)

class ExoplanetAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("NASA Exoplanet Archive Analyzer")
        self.root.geometry("1280x900")
        
        # Load data directly from NASA
        self.planetary_data = self.fetch_planetary_data()
        self.host_stars = sorted(self.planetary_data['hostname'].dropna().unique())
        
        self.setup_ui()
        self.setup_plots()
        
        # Initialize with first star
        if self.host_stars:
            self.star_var.set(self.host_stars[0])
            self.update_planets()

    def fetch_planetary_data(self) -> pd.DataFrame:
        """Fetches confirmed exoplanet data from NASA Archive"""
        try:
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            params = {
                'query': "select pl_name,hostname,pl_orbper,pl_orbsmax,pl_radj,"
                         "pl_bmassj,disc_year,pl_orbeccen,st_mass,st_rad,st_age,"
                         "st_teff,st_spectype,sy_dist from pscomppars",
                'format': 'csv'
            }
            df = pd.read_csv(url, params=params)
            logging.info(f"Fetched {len(df)} planetary records")
            return df
        except Exception as e:
            logging.error(f"Planetary data fetch failed: {str(e)}")
            messagebox.showerror("Data Error", f"Failed to load planetary data: {str(e)}")
            return pd.DataFrame()

    def fetch_atmospheric_data(self, planet_name: str) -> pd.DataFrame:
        """Gets molecular composition data from NASA Archive"""
        try:
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            params = {
                'query': f"select molecule,abundance,abundance_err from molecules "
                         f"where pl_name = '{planet_name}'",
                'format': 'csv'
            }
            df = pd.read_csv(url, params=params)
            logging.info(f"Fetched atmospheric data for {planet_name}")
            return df
        except Exception as e:
            logging.warning(f"Atmospheric data fetch failed for {planet_name}: {str(e)}")
            return pd.DataFrame()

    def fetch_spectroscopy_data(self, planet_name: str) -> pd.DataFrame:
        """Gets transmission spectroscopy data from NASA Archive"""
        try:
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            params = {
                'query': f"select wavelength,flux,flux_err,wavelength_unit,instrument "
                         f"from transmissionspec where pl_name = '{planet_name}'",
                'format': 'csv'
            }
            df = pd.read_csv(url, params=params)
            logging.info(f"Fetched spectroscopy data for {planet_name}")
            return df
        except Exception as e:
            logging.warning(f"Spectroscopy data fetch failed for {planet_name}: {str(e)}")
            return pd.DataFrame()

    def setup_ui(self):
        """Initialize main UI components"""
        # Create notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both')
        
        # Create tabs
        self.create_orbit_tab()
        self.create_spectrum_tab()
        self.create_hr_tab()
        self.create_research_tab()
        self.create_controls()
        
    def create_controls(self):
        """Create control panel"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Star selection
        ttk.Label(control_frame, text="Host Star:").grid(row=0, column=0, padx=5)
        self.star_var = tk.StringVar()
        star_combo = ttk.Combobox(control_frame, textvariable=self.star_var, 
                                 values=self.host_stars, state='readonly')
        star_combo.grid(row=0, column=1, padx=5)
        star_combo.bind("<<ComboboxSelected>>", self.update_planets)
        
        # Planet selection
        ttk.Label(control_frame, text="Planet:").grid(row=0, column=2, padx=5)
        self.planet_var = tk.StringVar()
        self.planet_combo = ttk.Combobox(control_frame, textvariable=self.planet_var, 
                                        state='readonly')
        self.planet_combo.grid(row=0, column=3, padx=5)
        
        # Action buttons
        ttk.Button(control_frame, text="Animate Orbit", 
                  command=self.animate_orbit).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Show Atmosphere", 
                  command=self.show_atmosphere).grid(row=0, column=5, padx=5)
        
    def create_orbit_tab(self):
        """Configure orbit animation tab"""
        self.orbit_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.orbit_tab, text="Orbit Viewer")
        
        self.fig_orbit = Figure(figsize=(8, 6))
        self.ax_orbit = self.fig_orbit.add_subplot(111)
        self.canvas_orbit = FigureCanvasTkAgg(self.fig_orbit, master=self.orbit_tab)
        self.canvas_orbit.get_tk_widget().pack(expand=True, fill='both')
        
        # Animation components
        self.planet_dot, = self.ax_orbit.plot([], [], 'o', color='cyan', markersize=6)
        self.dist_line, = self.ax_orbit.plot([], [], '--', color='red', alpha=0.7)
        self.ani = None

    def create_spectrum_tab(self):
        """Configure black body spectrum tab"""
        self.spectrum_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.spectrum_tab, text="Stellar Spectrum")
        
        self.fig_spectrum = Figure(figsize=(8, 6))
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.canvas_spectrum = FigureCanvasTkAgg(self.fig_spectrum, master=self.spectrum_tab)
        self.canvas_spectrum.get_tk_widget().pack(expand=True, fill='both')

    def create_hr_tab(self):
        """Configure HR diagram tab"""
        self.hr_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hr_tab, text="HR Diagram")
        
        self.fig_hr = Figure(figsize=(8, 6))
        self.ax_hr = self.fig_hr.add_subplot(111)
        self.canvas_hr = FigureCanvasTkAgg(self.fig_hr, master=self.hr_tab)
        self.canvas_hr.get_tk_widget().pack(expand=True, fill='both')

    def create_research_tab(self):
        """Configure research tab with data table"""
        self.research_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.research_tab, text="Research")
        
        columns = ("Planet", "Star", "Period (days)", "Distance (pc)", "Radius (Jup)")
        self.research_tree = ttk.Treeview(self.research_tab, columns=columns, show='headings')
        
        for col in columns:
            self.research_tree.heading(col, text=col)
            self.research_tree.column(col, width=120)
        
        vsb = ttk.Scrollbar(self.research_tab, orient="vertical", 
                           command=self.research_tree.yview)
        self.research_tree.configure(yscrollcommand=vsb.set)
        
        vsb.pack(side="right", fill="y")
        self.research_tree.pack(expand=True, fill='both')
        
        # Populate with real data
        self.populate_research_table()

    def populate_research_table(self):
        """Fill research table with NASA data"""
        for item in self.research_tree.get_children():
            self.research_tree.delete(item)
        
        for _, row in self.planetary_data.iterrows():
            self.research_tree.insert("", "end", values=(
                row['pl_name'],
                row['hostname'],
                f"{row['pl_orbper']:.2f}" if not pd.isna(row['pl_orbper']) else "N/A",
                f"{row['sy_dist']:.1f}" if not pd.isna(row['sy_dist']) else "N/A",
                f"{row['pl_radj']:.2f}" if not pd.isna(row['pl_radj']) else "N/A"
            ))

    def update_planets(self, event=None):
        """Update planet list when star changes"""
        star = self.star_var.get()
        planets = self.planetary_data[self.planetary_data['hostname'] == star]['pl_name'].tolist()
        self.planet_combo['values'] = planets
        if planets:
            self.planet_var.set(planets[0])
        self.plot_black_body_spectrum()
        self.plot_hr_diagram()

    def plot_black_body_spectrum(self):
        """Plot star's black body spectrum using NASA temperature data"""
        self.ax_spectrum.clear()
        star = self.star_var.get()
        
        if not star:
            return
            
        star_data = self.planetary_data[self.planetary_data['hostname'] == star]
        if star_data.empty or pd.isna(star_data.iloc[0]['st_teff']):
            self.ax_spectrum.text(0.5, 0.5, "Temperature data unavailable", ha='center')
            self.canvas_spectrum.draw_idle()
            return
        
        T = star_data.iloc[0]['st_teff']
        wavelengths = np.linspace(100e-9, 3000e-9, 500)
        
        # Planck's law
        spectral_radiance = (2 * H_PLANCK * C_LIGHT**2 / wavelengths**5) / (
            np.exp(H_PLANCK * C_LIGHT / (wavelengths * K_BOLTZMANN * T)) - 1)
        
        # Normalize
        normalized = spectral_radiance / np.max(spectral_radiance)
        
        self.ax_spectrum.plot(wavelengths * 1e9, normalized, color='orange')
        self.ax_spectrum.set_title(f"Black Body Spectrum: {star} (T={T:.0f} K)")
        self.ax_spectrum.set_xlabel("Wavelength (nm)")
        self.ax_spectrum.set_ylabel("Normalized Intensity")
        self.ax_spectrum.grid(True, linestyle='--', alpha=0.6)
        self.canvas_spectrum.draw_idle()

    def plot_hr_diagram(self):
        """Plot HR diagram using NASA stellar data"""
        self.ax_hr.clear()
        
        if self.planetary_data.empty:
            return
            
        # Calculate luminosity (L ∝ R²T⁴)
        valid_data = self.planetary_data.dropna(subset=['st_rad', 'st_teff'])
        valid_data['luminosity'] = (valid_data['st_rad']**2) * (valid_data['st_teff']/SOLAR_TEMP)**4
        
        self.ax_hr.scatter(
            valid_data['st_teff'],
            valid_data['luminosity'],
            alpha=0.3,
            s=5,
            label='Exoplanet Host Stars'
        )
        
        # Highlight selected star
        if self.star_var.get():
            star_data = valid_data[valid_data['hostname'] == self.star_var.get()]
            if not star_data.empty:
                star = star_data.iloc[0]
                self.ax_hr.scatter(
                    star['st_teff'],
                    star['luminosity'],
                    color='red',
                    s=50,
                    edgecolor='black',
                    label=f'Selected: {self.star_var.get()}'
                )
        
        self.ax_hr.set_yscale('log')
        self.ax_hr.set_xscale('log')
        self.ax_hr.set_xlim(self.ax_hr.get_xlim()[::-1])  # Reverse for HR diagram
        self.ax_hr.set_title("Hertzsprung-Russell Diagram")
        self.ax_hr.set_xlabel("Effective Temperature (K)")
        self.ax_hr.set_ylabel("Luminosity (Solar Units)")
        self.ax_hr.legend()
        self.ax_hr.grid(True, which="both", ls="--", alpha=0.5)
        self.canvas_hr.draw_idle()

    def animate_orbit(self):
        """Animate planet orbit using NASA orbital parameters"""
        planet_name = self.planet_var.get()
        if not planet_name:
            return
            
        if self.ani:
            self.ani.event_source.stop()
            
        planet_data = self.planetary_data[self.planetary_data['pl_name'] == planet_name]
        if planet_data.empty:
            return
            
        p = planet_data.iloc[0]
        
        try:
            a = float(p['pl_orbsmax'])  # Semi-major axis (AU)
            period_days = float(p['pl_orbper'])  # Orbital period (days)
            ecc = float(p['pl_orbeccen']) if pd.notna(p['pl_orbeccen']) else 0.0
            
            # Calculate orbit path
            b = a * np.sqrt(1 - ecc**2)
            theta = np.linspace(0, 2*np.pi, 360)
            x_orb = a * (np.cos(theta) - ecc)
            y_orb = b * np.sin(theta)
            
            # Setup plot
            self.ax_orbit.clear()
            self.ax_orbit.plot(x_orb, y_orb, 'w--', alpha=0.5)
            self.ax_orbit.plot(0, 0, 'o', color='gold', markersize=12)
            self.ax_orbit.set_title(f"Orbit of {planet_name}")
            self.ax_orbit.set_facecolor("black")
            self.ax_orbit.set_xlim(-a*1.5, a*1.5)
            self.ax_orbit.set_ylim(-a*1.5, a*1.5)
            self.ax_orbit.set_aspect('equal')
            self.ax_orbit.grid(True, linestyle='--', color='gray', alpha=0.4)
            
            # Animation function
            period_seconds = period_days * 86400
            mean_motion = 2 * np.pi / period_seconds
            
            def update(frame):
                M = mean_motion * frame
                try:
                    E = newton(lambda E: E - ecc*np.sin(E) - M, M, tol=1e-6)
                except RuntimeError:
                    E = M
                
                x = a * (np.cos(E) - ecc)
                y = b * np.sin(E)
                
                self.planet_dot.set_data([x], [y])
                self.dist_line.set_data([0, x], [0, y])
                return self.planet_dot, self.dist_line
            
            self.ani = animation.FuncAnimation(
                self.fig_orbit,
                update,
                frames=np.linspace(0, period_seconds, 360),
                interval=50,
                blit=True,
                repeat=True
            )
            
            self.canvas_orbit.draw_idle()
            
        except (ValueError, TypeError) as e:
            messagebox.showerror("Animation Error", f"Invalid data for orbit: {str(e)}")

    def show_atmosphere(self):
        """Display atmospheric composition using NASA data"""
        planet_name = self.planet_var.get()
        if not planet_name:
            return
            
        popup = tk.Toplevel(self.root)
        popup.title(f"Atmospheric Composition - {planet_name}")
        popup.geometry("1000x700")
        
        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill='both')
        
        # Transmission Spectrum Tab
        spec_frame = ttk.Frame(notebook)
        self.plot_transmission_spectrum(spec_frame, planet_name)
        notebook.add(spec_frame, text="Transmission Spectrum")
        
        # Molecular Abundance Tab
        mol_frame = ttk.Frame(notebook)
        self.plot_molecular_abundance(mol_frame, planet_name)
        notebook.add(mol_frame, text="Molecular Abundance")

    def plot_transmission_spectrum(self, parent, planet_name):
        """Plot actual transmission spectrum from NASA"""
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        
        data = self.fetch_spectroscopy_data(planet_name)
        
        if data.empty:
            ax.text(0.5, 0.5, "No spectroscopy data available", ha='center', va='center')
        else:
            # Group by instrument
            for instrument, group in data.groupby('instrument'):
                if 'flux_err' in group.columns:
                    ax.errorbar(
                        group['wavelength'],
                        group['flux'],
                        yerr=group['flux_err'],
                        fmt='o',
                        label=f"{instrument}",
                        capsize=3
                    )
                else:
                    ax.plot(
                        group['wavelength'],
                        group['flux'],
                        'o-',
                        label=f"{instrument}"
                    )
            
            ax.set_title(f"Transmission Spectrum: {planet_name}")
            ax.set_xlabel(f"Wavelength ({data['wavelength_unit'].iloc[0]})")
            ax.set_ylabel("Transit Depth")
            ax.legend()
            ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

    def plot_molecular_abundance(self, parent, planet_name):
        """Plot molecular abundance from NASA data"""
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        
        data = self.fetch_atmospheric_data(planet_name)
        
        if data.empty:
            ax.text(0.5, 0.5, "No molecular data available", ha='center', va='center')
        else:
            molecules = data['molecule'].tolist()
            abundances = data['abundance'].tolist()
            
            ax.bar(molecules, abundances)
            ax.set_title(f"Molecular Abundance: {planet_name}")
            ax.set_ylabel("Abundance")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y')
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

if __name__ == "__main__":
    root = tk.Tk()
    app = ExoplanetAnalyzer(root)
    root.mainloop()

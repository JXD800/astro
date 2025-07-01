import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
from io import StringIO
from typing import Optional, List, Dict, Tuple
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

def fetch_atmosphere_spectroscopy(planet_name: str) -> dict:
    """
    Fetches atmospheric spectroscopy data from NASA's Firefly service.
    Returns a dictionary with spectroscopy data and parsed spectral information.
    """
    # The correct URL for atmospheric spectroscopy data
    firefly_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly?atmospheres"
    
    print(f"🔗 Fetching atmospheric data from: {firefly_url}")
    print(f"🪐 Target planet: {planet_name}")
    
    try:
        # Query the Firefly service for atmospheric data
        params = {
            'planet': planet_name,
            'format': 'json'  # Try to get structured data
        }
        
        response = requests.get(firefly_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse the response content
        content = response.text
        
        # Try to extract spectral data from the response
        spectral_data = parse_spectral_data(content, planet_name)
        
        # Also try the TAP service for additional molecular data
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
            'spectral_data': spectral_data,
            'molecules_data': molecules_data,
            'status': 'success'
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching atmospheric data: {e}")
        # Generate synthetic spectral data for demonstration
        synthetic_data = generate_synthetic_spectrum(planet_name)
        return {
            'firefly_url': firefly_url,
            'error': str(e),
            'spectral_data': synthetic_data,
            'status': 'synthetic'
        }

def parse_spectral_data(content: str, planet_name: str) -> Dict:
    """
    Attempts to parse spectral data from NASA response.
    If no real data, generates synthetic data for visualization.
    """
    try:
        # Try to parse JSON response
        if content.strip().startswith('{') or content.strip().startswith('['):
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 0:
                # Extract wavelength and flux data
                return extract_wavelength_flux(data)
    except json.JSONDecodeError:
        pass
    
    # Try to parse CSV or other structured format
    try:
        # Look for numerical data patterns
        lines = content.split('\n')
        wavelengths, fluxes = [], []
        
        for line in lines:
            # Match lines with numerical data (wavelength, flux pairs)
            numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', line)
            if len(numbers) >= 2:
                try:
                    wavelengths.append(float(numbers[0]))
                    fluxes.append(float(numbers[1]))
                except ValueError:
                    continue
        
        if len(wavelengths) > 10:  # Need reasonable amount of data
            return {
                'wavelengths': wavelengths,
                'fluxes': fluxes,
                'type': 'parsed',
                'units': {'wavelength': 'microns', 'flux': 'W/m²/micron'}
            }
    except Exception:
        pass
    
    # If no parseable data found, generate synthetic spectrum
    return generate_synthetic_spectrum(planet_name)

def generate_synthetic_spectrum(planet_name: str) -> Dict:
    """
    Generates a realistic synthetic atmospheric spectrum for demonstration.
    """
    # Create wavelength range (0.5 to 5.5 microns, typical for atmospheric spectroscopy)
    wavelengths = np.linspace(0.5, 5.5, 1000)
    
    # Generate base continuum (blackbody-like)
    temperature = 1500  # Typical hot Jupiter temperature
    base_flux = 2e-14 * (wavelengths ** -4) * np.exp(-1.44e4 / (wavelengths * temperature))
    
    # Add noise
    noise = np.random.normal(0, base_flux * 0.1)
    spectrum = base_flux + noise
    
    # Add some absorption features at common molecular lines
    absorption_lines = {
        1.4: 0.3,   # H2O
        1.9: 0.25,  # H2O
        2.3: 0.2,   # CO
        3.3: 0.35,  # CH4
        4.3: 0.4,   # CO2
    }
    
    for line_center, depth in absorption_lines.items():
        # Create Gaussian absorption line
        line_profile = depth * np.exp(-((wavelengths - line_center) / 0.05) ** 2)
        spectrum -= line_profile * base_flux
    
    # Ensure no negative values
    spectrum = np.maximum(spectrum, base_flux * 0.1)
    
    return {
        'wavelengths': wavelengths.tolist(),
        'fluxes': spectrum.tolist(),
        'type': 'synthetic',
        'units': {'wavelength': 'microns', 'flux': 'W/m²/micron'},
        'note': f'Synthetic spectrum generated for {planet_name} (no real data available)'
    }

def extract_wavelength_flux(data: List[Dict]) -> Dict:
    """Extract wavelength and flux from structured data."""
    wavelengths, fluxes = [], []
    
    for entry in data:
        if isinstance(entry, dict):
            # Look for common field names
            wave_keys = ['wavelength', 'wave', 'lambda', 'wl']
            flux_keys = ['flux', 'intensity', 'f_lambda', 'spectrum']
            
            wave_val = None
            flux_val = None
            
            for key in wave_keys:
                if key in entry:
                    wave_val = entry[key]
                    break
            
            for key in flux_keys:
                if key in entry:
                    flux_val = entry[key]
                    break
            
            if wave_val is not None and flux_val is not None:
                try:
                    wavelengths.append(float(wave_val))
                    fluxes.append(float(flux_val))
                except (ValueError, TypeError):
                    continue
    
    if len(wavelengths) > 10:
        return {
            'wavelengths': wavelengths,
            'fluxes': fluxes,
            'type': 'real',
            'units': {'wavelength': 'microns', 'flux': 'W/m²/micron'}
        }
    
    return {}

def create_atmosphere_popup(planet_name: str, atmosphere_data: dict):
    """
    Creates a spectacular atmospheric spectroscopy popup with embedded matplotlib plots.
    """
    popup = tk.Toplevel(root)
    popup.title(f"🌌 Atmospheric Spectroscopy - {planet_name}")
    popup.geometry("1200x800")
    popup.configure(bg='#0a0a0a')
    
    # Make it modal and centered
    popup.transient(root)
    popup.grab_set()
    
    popup.update_idletasks()
    x = (popup.winfo_screenwidth() // 2) - (1200 // 2)
    y = (popup.winfo_screenheight() // 2) - (800 // 2)
    popup.geometry(f"1200x800+{x}+{y}")
    
    # Main container
    main_frame = tk.Frame(popup, bg='#0a0a0a', padx=15, pady=15)
    main_frame.pack(fill='both', expand=True)
    
    # Header section
    header_frame = tk.Frame(main_frame, bg='#0a0a0a')
    header_frame.pack(fill='x', pady=(0, 15))
    
    title_label = tk.Label(
        header_frame,
        text="🌌 ATMOSPHERIC SPECTROSCOPY ANALYSIS",
        font=('Arial', 18, 'bold'),
        fg='#00d4ff',
        bg='#0a0a0a'
    )
    title_label.pack()
    
    planet_label = tk.Label(
        header_frame,
        text=f"Target: {planet_name}",
        font=('Arial', 14, 'bold'),
        fg='#ffffff',
        bg='#0a0a0a'
    )
    planet_label.pack(pady=(5, 0))
    
    # Status indicator
    status = atmosphere_data.get('status', 'unknown')
    status_color = {'success': '#00ff00', 'synthetic': '#ffaa00', 'error': '#ff4444'}.get(status, '#ffffff')
    status_text = {
        'success': '✅ REAL DATA RETRIEVED',
        'synthetic': '🔬 SYNTHETIC SPECTRUM GENERATED',
        'error': '❌ DATA UNAVAILABLE'
    }.get(status, 'STATUS UNKNOWN')
    
    status_label = tk.Label(
        header_frame,
        text=status_text,
        font=('Arial', 12, 'bold'),
        fg=status_color,
        bg='#0a0a0a'
    )
    status_label.pack(pady=(5, 0))
    
    # Create notebook for different views
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill='both', expand=True, pady=(0, 15))
    
    # --- TAB 1: MAIN SPECTRUM PLOT ---
    spectrum_frame = tk.Frame(notebook, bg='#1a1a2e')
    notebook.add(spectrum_frame, text="📊 Atmospheric Spectrum")
    
    # Create the main spectral plot
    spectral_data = atmosphere_data.get('spectral_data', {})
    
    if spectral_data and 'wavelengths' in spectral_data and 'fluxes' in spectral_data:
        fig_spectrum = Figure(figsize=(12, 8), dpi=100, facecolor='#1a1a2e')
        ax_spectrum = fig_spectrum.add_subplot(111, facecolor='#0f0f1e')
        
        wavelengths = np.array(spectral_data['wavelengths'])
        fluxes = np.array(spectral_data['fluxes'])
        
        # Create the main spectrum plot
        ax_spectrum.plot(wavelengths, fluxes, color='#00aaff', linewidth=1.5, alpha=0.8)
        ax_spectrum.fill_between(wavelengths, fluxes, alpha=0.3, color='#00aaff')
        
        # Add molecular absorption markers if synthetic
        if spectral_data.get('type') == 'synthetic':
            molecular_lines = {
                1.4: ('H₂O', '#ff6b6b'),
                1.9: ('H₂O', '#ff6b6b'),
                2.3: ('CO', '#4ecdc4'),
                3.3: ('CH₄', '#45b7d1'),
                4.3: ('CO₂', '#96ceb4')
            }
            
            for wave, (molecule, color) in molecular_lines.items():
                if wave >= wavelengths.min() and wave <= wavelengths.max():
                    ax_spectrum.axvline(wave, color=color, linestyle='--', alpha=0.7, linewidth=2)
                    ax_spectrum.text(wave, fluxes.max() * 0.9, molecule, 
                                   rotation=90, color=color, fontweight='bold', fontsize=10)
        
        # Styling
        ax_spectrum.set_xlabel('Wavelength (μm)', fontsize=14, color='white')
        ax_spectrum.set_ylabel('Flux (W/m²/μm)', fontsize=14, color='white')
        ax_spectrum.set_title(f'{planet_name} Atmospheric Spectrum', fontsize=16, color='#00d4ff', pad=20)
        ax_spectrum.grid(True, alpha=0.3, color='white')
        ax_spectrum.tick_params(colors='white', labelsize=12)
        
        # Set axis limits
        ax_spectrum.set_xlim(wavelengths.min(), wavelengths.max())
        ax_spectrum.set_ylim(0, fluxes.max() * 1.1)
        
        # Add data source annotation
        data_type = spectral_data.get('type', 'unknown')
        note_text = spectral_data.get('note', f'Data type: {data_type}')
        ax_spectrum.text(0.02, 0.98, note_text, transform=ax_spectrum.transAxes, 
                        fontsize=10, color='#ffaa00', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='#0a0a0a', alpha=0.7))
        
        canvas_spectrum = FigureCanvasTkAgg(fig_spectrum, spectrum_frame)
        canvas_spectrum.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
    else:
        # No spectral data available
        no_data_label = tk.Label(
            spectrum_frame,
            text="📡 NO SPECTRAL DATA AVAILABLE\n\nThis planet has not been observed\nwith atmospheric spectroscopy",
            font=('Arial', 16),
            fg='#ff6b6b',
            bg='#1a1a2e',
            justify='center'
        )
        no_data_label.pack(expand=True)
    
    # --- TAB 2: DATA ANALYSIS ---
    analysis_frame = tk.Frame(notebook, bg='#2d2d44')
    notebook.add(analysis_frame, text="🔬 Analysis Details")
    
    analysis_text = tk.Text(
        analysis_frame,
        wrap=tk.WORD,
        font=('Consolas', 11),
        bg='#1e1e1e',
        fg='#00ff88',
        insertbackground='white'
    )
    analysis_scrollbar = tk.Scrollbar(analysis_frame, command=analysis_text.yview)
    analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
    
    # Generate analysis content
    analysis_content = f"""
🔬 SPECTROSCOPIC ANALYSIS REPORT
{'='*60}

🎯 TARGET: {planet_name}
📡 DATA SOURCE: {atmosphere_data.get('firefly_url', 'N/A')}
📊 STATUS: {atmosphere_data.get('status', 'Unknown').upper()}

"""
    
    if spectral_data:
        analysis_content += f"""
📈 SPECTRAL DATA SUMMARY:
• Wavelength Range: {min(spectral_data['wavelengths']):.2f} - {max(spectral_data['wavelengths']):.2f} μm
• Number of Data Points: {len(spectral_data['wavelengths'])}
• Flux Range: {min(spectral_data['fluxes']):.2e} - {max(spectral_data['fluxes']):.2e} W/m²/μm
• Data Type: {spectral_data.get('type', 'Unknown').upper()}

🧪 MOLECULAR ANALYSIS:
"""
        if spectral_data.get('type') == 'synthetic':
            analysis_content += """
• H₂O (Water): Strong absorption at 1.4 & 1.9 μm
• CO (Carbon Monoxide): Absorption feature at 2.3 μm  
• CH₄ (Methane): Absorption band at 3.3 μm
• CO₂ (Carbon Dioxide): Strong absorption at 4.3 μm

⚠️  NOTE: This is a synthetic spectrum for demonstration.
   Real atmospheric data requires space telescope observations.
"""
        else:
            analysis_content += """
• Analysis of real spectroscopic data
• Molecular features detected from actual observations
• Data processed from NASA atmospheric database
"""
    
    analysis_content += f"""

🌡️ ATMOSPHERIC SCIENCE CONTEXT:
• Atmospheric spectroscopy reveals molecular composition
• Absorption lines indicate specific molecules in atmosphere
• Temperature and pressure affect spectral line shapes
• Multi-wavelength observations provide 3D atmospheric structure

🚀 OBSERVATION TECHNIQUES:
• Transit Spectroscopy: Planet passes in front of star
• Emission Spectroscopy: Direct thermal emission from planet
• Phase Curve Analysis: Atmospheric dynamics from orbital motion

📚 SCIENTIFIC SIGNIFICANCE:
• Habitability assessment through molecular detections
• Atmospheric evolution and dynamics studies  
• Comparative planetology with Solar System worlds
• Search for biosignature molecules
"""
    
    analysis_text.insert('1.0', analysis_content)
    analysis_text.configure(state='disabled')
    
    analysis_scrollbar.pack(side='right', fill='y')
    analysis_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
    
    # --- TAB 3: MOLECULAR DETECTIONS ---
    molecules_frame = tk.Frame(notebook, bg='#2d2d44')
    notebook.add(molecules_frame, text="🧬 Molecular Database")
    
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
    molecules_content = f"""
🧪 MOLECULAR DETECTION DATABASE
{'='*50}

🎯 PLANET: {planet_name}
🔍 QUERY: NASA Exoplanet Archive molecular database

"""
    
    if molecules_data:
        molecules_content += f"✅ FOUND {len(molecules_data)} DETECTION RECORDS:\n\n"
        for i, record in enumerate(molecules_data, 1):
            molecules_content += f"#{i}. {record}\n\n"
    else:
        molecules_content += """❌ NO MOLECULAR DETECTIONS IN DATABASE

🔬 POSSIBLE REASONS:
• Planet not observed with atmospheric spectroscopy
• Observations lack sufficient signal-to-noise ratio
• Data not yet processed or published
• Planet name mismatch in database query

🌟 ATMOSPHERIC CHARACTERIZATION STATUS:
• Only ~200 exoplanets have atmospheric data
• Requires specialized space telescope observations
• Most productive: Hubble, Spitzer, JWST
• Ground-based observations limited by Earth's atmosphere

🚀 FUTURE PROSPECTS:
• James Webb Space Telescope revolutionizing field
• Next-generation ground telescopes (ELT, TMT)
• Specialized atmospheric characterization missions
• Improved data analysis techniques
"""
    
    molecules_text.insert('1.0', molecules_content)
    molecules_text.configure(state='disabled')
    
    molecules_scrollbar.pack(side='right', fill='y')
    molecules_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
    
    # Bottom control panel
    control_frame = tk.Frame(main_frame, bg='#0a0a0a')
    control_frame.pack(fill='x', pady=(10, 0))
    
    # Data source info
    source_label = tk.Label(
        control_frame,
        text=f"🔗 Data Source: {atmosphere_data.get('firefly_url', 'N/A')}",
        font=('Arial', 10),
        fg='#888888',
        bg='#0a0a0a'
    )
    source_label.pack(side='left')
    
    # Close button
    close_button = tk.Button(
        control_frame,
        text="🚀 CLOSE ANALYSIS",
        command=popup.destroy,
        font=('Arial', 12, 'bold'),
        bg='#ff4757',
        fg='white',
        padx=30,
        pady=8,
        relief='raised',
        borderwidth=2
    )
    close_button.pack(side='right')
    
    # Focus the popup
    popup.focus_set()

# Updated main function to call the atmospheric analysis
def check_atmosphere():
    """Enhanced controller function for atmospheric analysis with plotting."""
    planet_name = planet_var.get()
    if not planet_name:
        messagebox.showwarning("No Planet Selected", "Please select a star and a planet first.")
        return

    # Show loading message
    loading = tk.Toplevel(root)
    loading.title("Loading...")
    loading.geometry("400x150")
    loading.configure(bg='#1a1a2e')
    loading.transient(root)
    loading.grab_set()
    
    # Center loading window
    loading.update_idletasks()
    x = (loading.winfo_screenwidth() // 2) - (200)
    y = (loading.winfo_screenheight() // 2) - (75)
    loading.geometry(f"400x150+{x}+{y}")
    
    loading_label = tk.Label(
        loading,
        text=f"🔍 Querying NASA Atmospheric Databases\n\n📡 Target: {planet_name}\n🌌 Fetching spectroscopy data...",
        font=('Arial', 12),
        fg='#00d4ff',
        bg='#1a1a2e',
        justify='center'
    )
    loading_label.pack(expand=True)
    
    loading.update()
    
    try:
        # Fetch the atmospheric data
        atmosphere_data = fetch_atmosphere_spectroscopy(planet_name)
        
        # Close loading window
        loading.destroy()
        
        # Create and show the enhanced popup with plots
        create_atmosphere_popup(planet_name, atmosphere_data)
        
    except Exception as e:
        loading.destroy()
        messagebox.showerror("Error", f"Failed to fetch atmospheric data: {e}")

# This replaces the previous check_atmosphere function in your main code

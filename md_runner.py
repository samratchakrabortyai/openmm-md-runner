
#!/usr/bin/env python3
"""
Enhanced Modular OpenMM MD Simulation Script
=============================================
Author: Samrat Chakraborty
GitHub: https://github.com/samratchakrabortyai/openmm-md-runner
Version: 1.0.0
Date: January 2025
License: MIT
scssc3357@iacs.res.in
nooneknowme2004@gmail.com

A comprehensive molecular dynamics simulation toolkit built on OpenMM
with advanced features including checkpointing, analysis, and visualization.

Features:
    - Interactive and batch modes
    - Configuration file support (YAML/JSON)
    - Checkpoint/Restart capability
    - Comprehensive analysis with 9+ plot types
    - H-bond calculations
    - Energy conservation tracking
    - Automatic HTML report generation
    
Usage:
    python3 md_runner.py                           # Interactive mode
    python3 md_runner.py protein.pdb               # With PDB file
    python3 md_runner.py --config config.yaml      # Load from config
    python3 md_runner.py --analyze-only            # Analysis only
    python3 md_runner.py --batch                   # Non-interactive

Requirements:
    - OpenMM >= 7.5
    - NumPy >= 1.19
    - Matplotlib >= 3.3
    - PyYAML >= 5.0 (optional)

Citation:
    If you use this script in your research, please cite:
    Samrat Chakraborty (2025). OpenMM MD Runner: Enhanced Molecular 
    Dynamics Simulation Toolkit. 
    GitHub: https://github.com/samratchakrabortyai/openmm-md-runner
"""

__author__ = "Samrat Chakraborty"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import sys
import argparse
import math
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# ============================================================================
# OPTIONAL IMPORTS - Check availability
# ============================================================================
try:
    import json
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False
    print("⚠ JSON not available")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠ PyYAML not available - install with: pip install pyyaml")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠ NumPy not available - install with: pip install numpy")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ Matplotlib not available - install with: pip install matplotlib")

# ============================================================================
# OPENMM IMPORTS
# ============================================================================
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
except ImportError:
    try:
        from openmm.app import *
        from openmm import *
        from openmm.unit import *
    except ImportError:
        print("✗ OpenMM not found. Please install it.")
        sys.exit(1)

print("✓ OpenMM loaded successfully")


# ============================================================================
# SIMULATION CONFIGURATION CLASS
# ============================================================================
@dataclass
class SimConfig:
    """Simulation configuration container with save/load capability"""
    
    # Input files
    pdb_file: Optional[str] = None
    psf_file: Optional[str] = None
    ff_files: List[str] = field(default_factory=lambda: ['amber14-all.xml', 'amber14/tip3p.xml'])
    water_model: str = 'tip3p'
    
    # Box dimensions (nm)
    box_x: float = 5.0
    box_y: float = 5.0
    box_z: float = 5.0
    padding: Optional[float] = None
    
    # Conditions
    temperature: float = 300.0
    pressure: float = 1.0
    timestep: float = 2.0
    cutoff: float = 1.0
    
    # Simulation times (nanoseconds)
    nvt_time_ns: float = 1.0
    npt_time_ns: float = 10.0
    
    # Output frequency (picoseconds)
    log_freq_ps: float = 1.0
    save_freq_ps: float = 10.0
    print_freq_ps: float = 100.0
    
    # Checkpoint settings
    use_checkpoints: bool = False
    checkpoint_freq_ps: float = 100.0
    restart_from: Optional[str] = None
    
    # Analysis settings
    run_analysis: bool = True
    calc_hbonds: bool = False
    
    # Platform
    platform: str = 'CPU'
    
    # Output
    output_dir: str = ""
    
    # Calculated (auto-filled)
    nvt_steps: int = 0
    npt_steps: int = 0
    log_every: int = 0
    save_every: int = 0
    print_every: int = 0
    checkpoint_every: int = 0
    
    def calculate_steps(self):
        """Convert time to steps"""
        timestep_ps = self.timestep * 0.001  # fs -> ps
        
        # Simulation steps
        self.nvt_steps = int((self.nvt_time_ns * 1000) / timestep_ps)
        self.npt_steps = int((self.npt_time_ns * 1000) / timestep_ps)
        
        # Output frequency
        self.log_every = max(1, int(self.log_freq_ps / timestep_ps))
        self.save_every = max(1, int(self.save_freq_ps / timestep_ps))
        self.print_every = max(1, int(self.print_freq_ps / timestep_ps))
        self.checkpoint_every = max(1, int(self.checkpoint_freq_ps / timestep_ps))
    
    def save(self, filepath: str):
        """Save configuration to YAML/JSON file"""
        data = asdict(self)
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            if YAML_AVAILABLE:
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                print("  ⚠ YAML not available, saving as JSON")
                filepath = filepath.replace('.yaml', '.json').replace('.yml', '.json')
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"  ✓ Configuration saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file"""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            if YAML_AVAILABLE:
                with open(filepath) as f:
                    data = yaml.safe_load(f)
            else:
                raise ImportError("PyYAML required for .yaml files")
        else:
            with open(filepath) as f:
                data = json.load(f)
        
        # Remove calculated fields if present
        for key in ['nvt_steps', 'npt_steps', 'log_every', 'save_every', 
                    'print_every', 'checkpoint_every']:
            data.pop(key, None)
        
        config = cls(**data)
        config.calculate_steps()
        return config


# ============================================================================
# CHECKPOINT MANAGER CLASS
# ============================================================================
class CheckpointManager:
    """Manage simulation checkpoints for restart capability"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, sim, stage: str, step: int):
        """Save simulation checkpoint"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f'{stage}_{step:08d}.chk')
        sim.saveCheckpoint(checkpoint_file)
        
        # Save metadata
        metadata = {
            'stage': stage,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        meta_file = os.path.join(self.checkpoint_dir, f'{stage}_{step:08d}.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest checkpoint reference
        latest_file = os.path.join(self.checkpoint_dir, f'latest_{stage}.txt')
        with open(latest_file, 'w') as f:
            f.write(checkpoint_file)
        
        print(f"    💾 Checkpoint: {stage} step {step:,}")
        return checkpoint_file
    
    def load_checkpoint(self, sim, checkpoint_file: str):
        """Load simulation from checkpoint"""
        if not os.path.exists(checkpoint_file):
            # Try in checkpoint directory
            checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        sim.loadCheckpoint(checkpoint_file)
        print(f"  ✓ Loaded checkpoint: {checkpoint_file}")
        
        # Try to load metadata
        meta_file = checkpoint_file.replace('.chk', '.json')
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                metadata = json.load(f)
            return metadata
        return {}
    
    def get_latest_checkpoint(self, stage: str) -> Optional[str]:
        """Get the latest checkpoint for a stage"""
        latest_file = os.path.join(self.checkpoint_dir, f'latest_{stage}.txt')
        if os.path.exists(latest_file):
            with open(latest_file) as f:
                return f.read().strip()
        return None
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints"""
        checkpoints = []
        if os.path.exists(self.checkpoint_dir):
            for f in os.listdir(self.checkpoint_dir):
                if f.endswith('.chk'):
                    checkpoints.append(f)
        return sorted(checkpoints)


# ============================================================================
# ANALYSIS CLASS WITH ALL PLOTS
# ============================================================================
class SimulationAnalyzer:
    """Complete analysis toolkit for MD simulations with all graph types"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'analysis_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Data storage
        self.nvt_data = None
        self.npt_data = None
    
    def parse_log_file(self, log_file: str) -> Optional[Dict[str, Any]]:
        """Parse OpenMM log file and extract all data"""
        log_path = os.path.join(self.output_dir, log_file)
        
        if not os.path.exists(log_path):
            print(f"  ⚠ Log file not found: {log_path}")
            return None
        
        print(f"  Parsing: {log_file}")
        
        # Initialize data containers
        data = {
            'step': [],
            'time_ps': [],
            'temperature': [],
            'potential': [],
            'kinetic': [],
            'total': [],
            'density': [],
            'volume': [],
            'speed': []
        }
        
        with open(log_path, 'r') as f:
            header_line = None
            
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse header
                if line.startswith('#'):
                    header_line = line.strip('#').strip().replace('"', '')
                    continue
                
                # Parse data
                try:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        data['step'].append(int(float(parts[0])))
                        data['time_ps'].append(float(parts[1]))
                    if len(parts) >= 3:
                        data['temperature'].append(float(parts[2]))
                    if len(parts) >= 4:
                        data['potential'].append(float(parts[3]))
                    if len(parts) >= 5:
                        data['kinetic'].append(float(parts[4]))
                    if len(parts) >= 6:
                        data['total'].append(float(parts[5]))
                    if len(parts) >= 7:
                        data['density'].append(float(parts[6]))
                    if len(parts) >= 8:
                        data['volume'].append(float(parts[7]))
                    if len(parts) >= 9:
                        data['speed'].append(float(parts[8]))
                except (ValueError, IndexError):
                    continue
        
        # Convert to numpy arrays if available
        if NUMPY_AVAILABLE:
            for key in data:
                if data[key]:
                    data[key] = np.array(data[key])
                else:
                    data[key] = np.array([])
        
        print(f"    Found {len(data['step'])} data points")
        return data
    
    def plot_energy_components(self, data: Dict, title_prefix: str = ""):
        """Plot 1: Energy components (Potential, Kinetic, Total) vs Time"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            print("  ⚠ Matplotlib/NumPy required for plotting")
            return
        
        if len(data['time_ps']) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Energy components
        if len(data['potential']) > 0:
            ax1.plot(data['time_ps'], data['potential'], label='Potential Energy', 
                    color='cornflowerblue', alpha=0.8)
        if len(data['kinetic']) > 0:
            ax1.plot(data['time_ps'], data['kinetic'], label='Kinetic Energy', 
                    color='darkorange', alpha=0.8)
        if len(data['total']) > 0:
            ax1.plot(data['time_ps'], data['total'], label='Total Energy', 
                    color='black', linewidth=2, linestyle='--')
        
        ax1.set_ylabel('Energy (kJ/mol)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_title(f'{title_prefix}Energy Components vs Time', fontsize=14)
        
        # Temperature
        if len(data['temperature']) > 0:
            ax2.plot(data['time_ps'], data['temperature'], color='red', alpha=0.7)
            mean_temp = np.mean(data['temperature'])
            std_temp = np.std(data['temperature'])
            ax2.axhline(y=mean_temp, color='darkred', linestyle='--', 
                       label=f'Mean: {mean_temp:.1f} ± {std_temp:.1f} K')
            ax2.fill_between(data['time_ps'], mean_temp - std_temp, mean_temp + std_temp,
                            alpha=0.2, color='red')
            ax2.legend(loc='best')
        
        ax2.set_xlabel('Time (ps)', fontsize=12)
        ax2.set_ylabel('Temperature (K)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{title_prefix.lower().replace(" ", "_")}energy_components.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_energy_with_temperature(self, data: Dict, title_prefix: str = ""):
        """Plot 2: Energy and Temperature with dual Y-axis"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        if len(data['time_ps']) == 0:
            return
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Primary Y-axis: Energy
        color = 'tab:blue'
        ax1.set_xlabel('Time (ps)', fontsize=14)
        ax1.set_ylabel('Energy (kJ/mol)', color=color, fontsize=14)
        
        if len(data['potential']) > 0:
            ax1.plot(data['time_ps'], data['potential'], label='Potential Energy', 
                    color='cornflowerblue')
        if len(data['kinetic']) > 0:
            ax1.plot(data['time_ps'], data['kinetic'], label='Kinetic Energy', 
                    color='darkorange')
        if len(data['total']) > 0:
            ax1.plot(data['time_ps'], data['total'], label='Total Energy', 
                    color='black', linestyle='--', linewidth=2)
        
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Secondary Y-axis: Temperature
        if len(data['temperature']) > 0:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Temperature (K)', color=color, fontsize=14)
            ax2.plot(data['time_ps'], data['temperature'], label='Temperature', 
                    color=color, linestyle=':')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax1.legend(loc='best')
        
        plt.title(f'{title_prefix}Energy and Temperature vs Time', fontsize=16)
        fig.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, f'{title_prefix.lower().replace(" ", "_")}energy_temp_dual.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_energy_drift(self, data: Dict, title_prefix: str = ""):
        """Plot 3: Relative energy drift (E - E0) / E0"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        if len(data['total']) < 2:
            return
        
        E0 = data['total'][0]
        if E0 == 0:
            print("  ⚠ Initial energy is zero, skipping drift plot")
            return
        
        relative_drift = (data['total'] - E0) / abs(E0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['time_ps'], relative_drift * 1e6, color='darkred', linewidth=1.5)
        
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Relative Energy Drift (ppm)', fontsize=12)
        ax.set_title(f'{title_prefix}Energy Conservation - Relative Drift', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add statistics
        final_drift = relative_drift[-1] * 1e6
        drift_rate = final_drift / data['time_ps'][-1] if data['time_ps'][-1] > 0 else 0
        ax.text(0.02, 0.98, f'Final drift: {final_drift:.4f} ppm\nDrift rate: {drift_rate:.4f} ppm/ps',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{title_prefix.lower().replace(" ", "_")}energy_drift.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_cumulative_stddev(self, data: Dict, title_prefix: str = ""):
        """Plot 4: Cumulative standard deviation sqrt(<E^2> - <E>^2)"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        if len(data['total']) < 2:
            return
        
        # Calculate cumulative statistics
        cumulative_std = []
        times = []
        
        for i in range(1, len(data['total'])):
            current_energies = data['total'][:i+1]
            n = len(current_energies)
            
            # Formula: sqrt(<E^2> - <E>^2)
            mean_E = np.mean(current_energies)
            mean_E_sq = np.mean(current_energies**2)
            variance = mean_E_sq - mean_E**2
            
            if variance < 0:
                variance = 0
            
            std_dev = math.sqrt(variance)
            cumulative_std.append(std_dev)
            times.append(data['time_ps'][i])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, cumulative_std, color='blue', linewidth=1.5)
        
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel(r'Cumulative Std Dev $\sqrt{\langle E^2 \rangle - \langle E \rangle^2}$ (kJ/mol)', fontsize=12)
        ax.set_title(f'{title_prefix}Energy Standard Deviation Convergence', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{title_prefix.lower().replace(" ", "_")}cumulative_stddev.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_energy_fluctuations(self, data: Dict, title_prefix: str = ""):
        """Plot 5: Instantaneous fluctuations from cumulative mean"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        if len(data['total']) < 2:
            return
        
        # Calculate instantaneous fluctuation from cumulative mean
        fluctuations = []
        times = []
        
        for i in range(1, len(data['total'])):
            current_energies = data['total'][:i+1]
            mean_E = np.mean(current_energies)
            E = data['total'][i]
            
            # sqrt((E - <E>)^2) = |E - <E>|
            fluctuation = abs(E - mean_E)
            fluctuations.append(fluctuation)
            times.append(data['time_ps'][i])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, fluctuations, color='purple', linewidth=1.0, alpha=0.7)
        
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel(r'Fluctuation $\sqrt{(E - \langle E \rangle)^2}$ (kJ/mol)', fontsize=12)
        ax.set_title(f'{title_prefix}Instantaneous Energy Fluctuation', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{title_prefix.lower().replace(" ", "_")}fluctuations.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_density_volume(self, data: Dict, title_prefix: str = ""):
        """Plot 6: Density and Volume for NPT simulations"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        has_density = len(data.get('density', [])) > 0
        has_volume = len(data.get('volume', [])) > 0
        
        if not has_density and not has_volume:
            print("  ⚠ No density/volume data available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Density plot
        if has_density:
            ax1 = axes[0]
            ax1.plot(data['time_ps'], data['density'], color='blue', alpha=0.7)
            mean_density = np.mean(data['density'])
            std_density = np.std(data['density'])
            ax1.axhline(y=mean_density, color='darkblue', linestyle='--',
                       label=f'Mean: {mean_density:.4f} ± {std_density:.4f} g/mL')
            ax1.fill_between(data['time_ps'], mean_density - std_density, mean_density + std_density,
                            alpha=0.2, color='blue')
            ax1.set_ylabel('Density (g/mL)', fontsize=12)
            ax1.set_title(f'{title_prefix}System Density', fontsize=14)
            ax1.legend(loc='best')
            ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Volume plot
        if has_volume:
            ax2 = axes[1]
            ax2.plot(data['time_ps'], data['volume'], color='green', alpha=0.7)
            mean_volume = np.mean(data['volume'])
            std_volume = np.std(data['volume'])
            ax2.axhline(y=mean_volume, color='darkgreen', linestyle='--',
                       label=f'Mean: {mean_volume:.2f} ± {std_volume:.2f} nm³')
            ax2.set_ylabel('Volume (nm³)', fontsize=12)
            ax2.set_title(f'{title_prefix}Box Volume', fontsize=14)
            ax2.legend(loc='best')
            ax2.grid(True, linestyle='--', alpha=0.6)
        
        axes[1].set_xlabel('Time (ps)', fontsize=12)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{title_prefix.lower().replace(" ", "_")}density_volume.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_temperature_distribution(self, data: Dict, title_prefix: str = ""):
        """Plot 7: Temperature distribution histogram"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        if len(data['temperature']) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time series
        ax1.plot(data['time_ps'], data['temperature'], color='red', alpha=0.6)
        mean_temp = np.mean(data['temperature'])
        std_temp = np.std(data['temperature'])
        ax1.axhline(y=mean_temp, color='darkred', linestyle='--', linewidth=2)
        ax1.fill_between(data['time_ps'], mean_temp - std_temp, mean_temp + std_temp,
                        alpha=0.2, color='red')
        ax1.set_xlabel('Time (ps)', fontsize=12)
        ax1.set_ylabel('Temperature (K)', fontsize=12)
        ax1.set_title(f'{title_prefix}Temperature Time Series', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Histogram
        ax2.hist(data['temperature'], bins=50, color='red', alpha=0.7, edgecolor='darkred')
        ax2.axvline(x=mean_temp, color='black', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_temp:.2f} K')
        ax2.axvline(x=mean_temp - std_temp, color='gray', linestyle=':', linewidth=1.5)
        ax2.axvline(x=mean_temp + std_temp, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Std: {std_temp:.2f} K')
        ax2.set_xlabel('Temperature (K)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'{title_prefix}Temperature Distribution', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{title_prefix.lower().replace(" ", "_")}temperature_dist.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_equilibration_summary(self):
        """Plot 8: Combined equilibration check for NVT and NPT"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # NVT Temperature
        if self.nvt_data and len(self.nvt_data['temperature']) > 0:
            ax = axes[0, 0]
            ax.plot(self.nvt_data['time_ps'], self.nvt_data['temperature'], 
                   alpha=0.6, color='red')
            mean_t = np.mean(self.nvt_data['temperature'])
            ax.axhline(y=mean_t, color='darkred', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_t:.1f} K')
            ax.set_ylabel('Temperature (K)', fontsize=12)
            ax.set_title('NVT: Temperature Equilibration', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # NVT Energy
        if self.nvt_data and len(self.nvt_data['total']) > 0:
            ax = axes[0, 1]
            ax.plot(self.nvt_data['time_ps'], self.nvt_data['total'], 
                   alpha=0.6, color='blue')
            ax.set_ylabel('Total Energy (kJ/mol)', fontsize=12)
            ax.set_title('NVT: Energy Equilibration', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # NPT Density
        if self.npt_data and len(self.npt_data.get('density', [])) > 0:
            ax = axes[1, 0]
            ax.plot(self.npt_data['time_ps'], self.npt_data['density'], 
                   alpha=0.6, color='green')
            mean_d = np.mean(self.npt_data['density'])
            ax.axhline(y=mean_d, color='darkgreen', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_d:.4f} g/mL')
            ax.set_xlabel('Time (ps)', fontsize=12)
            ax.set_ylabel('Density (g/mL)', fontsize=12)
            ax.set_title('NPT: Density Equilibration', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # NPT Temperature
        if self.npt_data and len(self.npt_data['temperature']) > 0:
            ax = axes[1, 1]
            ax.plot(self.npt_data['time_ps'], self.npt_data['temperature'], 
                   alpha=0.6, color='red')
            mean_t = np.mean(self.npt_data['temperature'])
            ax.axhline(y=mean_t, color='darkred', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_t:.1f} K')
            ax.set_xlabel('Time (ps)', fontsize=12)
            ax.set_ylabel('Temperature (K)', fontsize=12)
            ax.set_title('NPT: Temperature Equilibration', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle('Equilibration Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, 'equilibration_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def plot_all_energies_combined(self):
        """Plot 9: Combined energy plot for both NVT and NPT"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        offset = 0
        colors = {'NVT': 'blue', 'NPT': 'green'}
        
        if self.nvt_data and len(self.nvt_data['total']) > 0:
            ax.plot(self.nvt_data['time_ps'], self.nvt_data['total'], 
                   color=colors['NVT'], label='NVT Total Energy', alpha=0.7)
            offset = self.nvt_data['time_ps'][-1]
        
        if self.npt_data and len(self.npt_data['total']) > 0:
            npt_times = self.npt_data['time_ps'] + offset
            ax.plot(npt_times, self.npt_data['total'], 
                   color=colors['NPT'], label='NPT Total Energy', alpha=0.7)
            
            # Mark transition
            ax.axvline(x=offset, color='gray', linestyle='--', linewidth=2, 
                      label='NVT → NPT')
        
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Total Energy (kJ/mol)', fontsize=12)
        ax.set_title('Combined Simulation Energy Profile', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'combined_energy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def calculate_statistics(self, data: Dict, stage: str) -> Dict:
        """Calculate and print statistics for a simulation stage"""
        stats = {}
        
        print(f"\n  {stage} Statistics:")
        print("  " + "-" * 40)
        
        if len(data['temperature']) > 0:
            stats['temp_mean'] = float(np.mean(data['temperature']))
            stats['temp_std'] = float(np.std(data['temperature']))
            print(f"    Temperature: {stats['temp_mean']:.2f} ± {stats['temp_std']:.2f} K")
        
        if len(data['total']) > 0:
            stats['energy_mean'] = float(np.mean(data['total']))
            stats['energy_std'] = float(np.std(data['total']))
            print(f"    Total Energy: {stats['energy_mean']:.2f} ± {stats['energy_std']:.2f} kJ/mol")
        
        if len(data.get('density', [])) > 0:
            stats['density_mean'] = float(np.mean(data['density']))
            stats['density_std'] = float(np.std(data['density']))
            print(f"    Density: {stats['density_mean']:.4f} ± {stats['density_std']:.4f} g/mL")
        
        if len(data['time_ps']) > 0:
            stats['simulation_time'] = float(data['time_ps'][-1])
            print(f"    Simulation time: {stats['simulation_time']:.2f} ps")
        
        return stats
    
    def generate_html_report(self, stats_nvt: Dict, stats_npt: Dict):
        """Generate HTML report with all plots"""
        
        # Get list of generated plots
        plots = []
        if os.path.exists(self.plots_dir):
            plots = sorted([f for f in os.listdir(self.plots_dir) if f.endswith('.png')])
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MD Simulation Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
        .stats {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .stats table {{ width: 100%; border-collapse: collapse; }}
        .stats td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        .stats td:first-child {{ font-weight: bold; width: 200px; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .plot-title {{ font-weight: bold; margin-bottom: 5px; color: #333; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 MD Simulation Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Simulation Statistics</h2>
        <div class="grid">
            <div class="stats">
                <h3>NVT Equilibration</h3>
                <table>
                    <tr><td>Temperature</td><td>{stats_nvt.get('temp_mean', 'N/A'):.2f} ± {stats_nvt.get('temp_std', 0):.2f} K</td></tr>
                    <tr><td>Total Energy</td><td>{stats_nvt.get('energy_mean', 'N/A'):.2f} ± {stats_nvt.get('energy_std', 0):.2f} kJ/mol</td></tr>
                    <tr><td>Duration</td><td>{stats_nvt.get('simulation_time', 'N/A'):.2f} ps</td></tr>
                </table>
            </div>
            <div class="stats">
                <h3>NPT Production</h3>
                <table>
                    <tr><td>Temperature</td><td>{stats_npt.get('temp_mean', 'N/A'):.2f} ± {stats_npt.get('temp_std', 0):.2f} K</td></tr>
                    <tr><td>Total Energy</td><td>{stats_npt.get('energy_mean', 'N/A'):.2f} ± {stats_npt.get('energy_std', 0):.2f} kJ/mol</td></tr>
                    <tr><td>Density</td><td>{stats_npt.get('density_mean', 'N/A'):.4f} ± {stats_npt.get('density_std', 0):.4f} g/mL</td></tr>
                    <tr><td>Duration</td><td>{stats_npt.get('simulation_time', 'N/A'):.2f} ps</td></tr>
                </table>
            </div>
        </div>
        
        <h2>📈 Analysis Plots</h2>
"""
        
        for plot in plots:
            plot_title = plot.replace('_', ' ').replace('.png', '').title()
            html_content += f"""
        <div class="plot">
            <div class="plot-title">{plot_title}</div>
            <img src="analysis_plots/{plot}" alt="{plot_title}">
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        report_path = os.path.join(self.output_dir, 'analysis_report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"  ✓ HTML Report: {report_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 60)
        print("  RUNNING ANALYSIS")
        print("=" * 60)
        
        stats_nvt = {}
        stats_npt = {}
        
        # Parse NVT log
        print("\n  [1] Analyzing NVT Equilibration...")
        self.nvt_data = self.parse_log_file('02_nvt.log')
        if self.nvt_data:
            stats_nvt = self.calculate_statistics(self.nvt_data, "NVT")
            self.plot_energy_components(self.nvt_data, "NVT_")
            self.plot_energy_with_temperature(self.nvt_data, "NVT_")
            self.plot_temperature_distribution(self.nvt_data, "NVT_")
        
        # Parse NPT log
        print("\n  [2] Analyzing NPT Production...")
        self.npt_data = self.parse_log_file('03_production.log')
        if self.npt_data:
            stats_npt = self.calculate_statistics(self.npt_data, "NPT")
            self.plot_energy_components(self.npt_data, "NPT_")
            self.plot_energy_with_temperature(self.npt_data, "NPT_")
            self.plot_energy_drift(self.npt_data, "NPT_")
            self.plot_cumulative_stddev(self.npt_data, "NPT_")
            self.plot_energy_fluctuations(self.npt_data, "NPT_")
            self.plot_density_volume(self.npt_data, "NPT_")
            self.plot_temperature_distribution(self.npt_data, "NPT_")
        
        # Combined plots
        print("\n  [3] Generating Combined Plots...")
        self.plot_equilibration_summary()
        self.plot_all_energies_combined()
        
        # Generate HTML report
        print("\n  [4] Generating Report...")
        self.generate_html_report(stats_nvt, stats_npt)
        
        print("\n" + "=" * 60)
        print("  ✓ ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"  Plots saved in: {self.plots_dir}/")
        
        return stats_nvt, stats_npt


# ============================================================================
# HYDROGEN BOND ANALYZER
# ============================================================================
class HBondAnalyzer:
    """Analyze hydrogen bonds in the simulation"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'analysis_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # H-bond criteria
        self.distance_cutoff = 0.35  # nm
        self.angle_cutoff = 150.0    # degrees
    
    def analyze_structure(self, pdb_file: str) -> int:
        """Count H-bonds in a PDB structure"""
        pdb_path = os.path.join(self.output_dir, pdb_file)
        
        if not os.path.exists(pdb_path):
            print(f"  ⚠ PDB file not found: {pdb_path}")
            return 0
        
        try:
            pdb = PDBFile(pdb_path)
            positions = pdb.getPositions(asNumpy=True)
            topology = pdb.topology
            
            # Find donors and acceptors
            donors = []  # (heavy_atom_idx, h_atom_idx)
            acceptors = []  # atom_idx
            
            for atom in topology.atoms():
                element = atom.element.symbol if atom.element else ''
                
                # Find H atoms bonded to N or O (donors)
                if element == 'H':
                    for bond in topology.bonds():
                        if atom in bond:
                            other = bond[0] if bond[1] == atom else bond[1]
                            other_element = other.element.symbol if other.element else ''
                            if other_element in ['N', 'O']:
                                donors.append((other.index, atom.index))
                
                # Find N and O atoms (acceptors)
                if element in ['N', 'O']:
                    acceptors.append(atom.index)
            
            # Count H-bonds
            n_hbonds = 0
            for donor_heavy, donor_h in donors:
                donor_pos = positions[donor_heavy]
                h_pos = positions[donor_h]
                
                for acceptor_idx in acceptors:
                    if acceptor_idx == donor_heavy:
                        continue
                    
                    acceptor_pos = positions[acceptor_idx]
                    
                    # Check H-A distance
                    h_a_dist = np.linalg.norm(h_pos - acceptor_pos)
                    if h_a_dist > self.distance_cutoff:
                        continue
                    
                    # Check D-H-A angle
                    vec_dh = h_pos - donor_pos
                    vec_ha = acceptor_pos - h_pos
                    
                    norm_dh = np.linalg.norm(vec_dh)
                    norm_ha = np.linalg.norm(vec_ha)
                    
                    if norm_dh > 0 and norm_ha > 0:
                        cos_angle = np.dot(vec_dh, vec_ha) / (norm_dh * norm_ha)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        
                        if angle >= self.angle_cutoff:
                            n_hbonds += 1
            
            return n_hbonds
            
        except Exception as e:
            print(f"  ⚠ Error analyzing H-bonds: {e}")
            return 0
    
    def analyze_trajectory(self):
        """Analyze H-bonds across available structures"""
        print("\n  Analyzing Hydrogen Bonds...")
        
        structures = [
            ('00_initial.pdb', 'Initial'),
            ('01_minimized.pdb', 'Minimized'),
            ('02_nvt_final.pdb', 'NVT Final'),
            ('03_final.pdb', 'NPT Final')
        ]
        
        results = []
        labels = []
        
        for pdb_file, label in structures:
            n_hbonds = self.analyze_structure(pdb_file)
            if n_hbonds > 0 or os.path.exists(os.path.join(self.output_dir, pdb_file)):
                results.append(n_hbonds)
                labels.append(label)
                print(f"    {label}: {n_hbonds} H-bonds")
        
        if results and MATPLOTLIB_AVAILABLE:
            self.plot_hbonds(labels, results)
        
        return dict(zip(labels, results))
    
    def plot_hbonds(self, labels: List[str], counts: List[int]):
        """Plot H-bond counts"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))
        bars = ax.bar(labels, counts, color=colors, edgecolor='darkblue')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Simulation Stage', fontsize=12)
        ax.set_ylabel('Number of Hydrogen Bonds', fontsize=12)
        ax.set_title('Hydrogen Bond Count Across Simulation', fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'hbond_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")


# ============================================================================
# FORCE FIELD PRESETS
# ============================================================================
FF_PRESETS = {
    '1': {
        'name': 'AMBER14 + TIP3P',
        'files': ['amber14-all.xml', 'amber14/tip3p.xml'],
        'water': 'tip3p'
    },
    '2': {
        'name': 'AMBER14 + SPC/E',
        'files': ['amber14-all.xml', 'amber14/spce.xml'],
        'water': 'spce'
    },
    '3': {
        'name': 'AMBER14 + TIP4P-Ew',
        'files': ['amber14-all.xml', 'amber14/tip4pew.xml'],
        'water': 'tip4pew'
    },
    '4': {
        'name': 'AMBER99SB-ILDN + TIP3P',
        'files': ['amber99sbildn.xml', 'tip3p.xml'],
        'water': 'tip3p'
    },
    '5': {
        'name': 'CHARMM36 + TIP3P',
        'files': ['charmm36.xml', 'charmm36/water.xml'],
        'water': 'tip3p'
    },
    '6': {
        'name': 'CHARMM36 + TIP4P-Ew',
        'files': ['charmm36.xml', 'charmm36/tip4pew.xml'],
        'water': 'tip4pew'
    },
}


# ============================================================================
# INPUT HELPER FUNCTIONS
# ============================================================================
def get_float(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Get float input with validation"""
    while True:
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            value = float(raw) if raw else default
            
            if min_val is not None and value < min_val:
                print(f"    ⚠ Must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"    ⚠ Must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("    ⚠ Invalid number")


def get_choice(prompt: str, options: List[str], default: str = None) -> str:
    """Get choice from options"""
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip().lower()
        choice = raw if raw else default
        if choice in options:
            return choice
        print(f"    ⚠ Choose from: {', '.join(options)}")


def get_yn(prompt: str, default: bool = True) -> bool:
    """Get yes/no input"""
    default_str = 'y' if default else 'n'
    raw = input(f"  {prompt} (y/n) [{default_str}]: ").strip().lower()
    if raw == '':
        return default
    return raw in ['y', 'yes', '1', 'true']


def file_exists(path: str) -> bool:
    """Check if file exists"""
    return path is not None and os.path.exists(path)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Modular OpenMM MD Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 md_sim.py                                    # Interactive water box
  python3 md_sim.py protein.pdb                        # With PDB
  python3 md_sim.py protein.pdb --psf protein.psf      # With PSF
  python3 md_sim.py --config config.yaml               # Load from config file
  python3 md_sim.py --analyze-only                     # Run analysis only
  python3 md_sim.py --batch                            # Non-interactive defaults
        """
    )
    
    parser.add_argument('pdb', nargs='?', default=None,
                        help='PDB file (optional)')
    parser.add_argument('--psf', '-p', default=None,
                        help='PSF file (optional)')
    parser.add_argument('--ff', '-f', nargs='+', default=None,
                        help='Force field XML files')
    parser.add_argument('--water', '-w', default='tip3p',
                        choices=['tip3p', 'tip4pew', 'tip5p', 'spce'],
                        help='Water model')
    parser.add_argument('--platform', default='CPU',
                        choices=['CPU', 'CUDA', 'OpenCL'],
                        help='Compute platform')
    parser.add_argument('--config', '-c', default=None,
                        help='Load configuration from YAML/JSON file')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Batch mode (no prompts)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='Run analysis only on existing simulation')
    
    return parser.parse_args()


# ============================================================================
# INTERACTIVE SETUP
# ============================================================================
def interactive_setup(args) -> SimConfig:
    """Interactive configuration setup"""
    
    config = SimConfig()
    
    print("\n" + "=" * 60)
    print("  SIMULATION SETUP")
    print("=" * 60)
    
    # -------------------------
    # SECTION 0: CONFIG FILE
    # -------------------------
    print("\n" + "-" * 60)
    print("  [0] CONFIGURATION FILE (Optional)")
    print("-" * 60)
    
    if get_yn("Load settings from a config file?", default=False):
        while True:
            config_path = input("  Enter config file path: ").strip()
            if file_exists(config_path):
                try:
                    config = SimConfig.load(config_path)
                    print(f"  ✓ Loaded configuration from: {config_path}")
                    if get_yn("Modify loaded settings?", default=False):
                        break  # Continue with interactive setup
                    else:
                        return config
                except Exception as e:
                    print(f"  ✗ Error loading config: {e}")
            else:
                print("  ✗ File not found")
            if not get_yn("Try another file?", default=True):
                break
    
    # -------------------------
    # SECTION 1: INPUT FILES
    # -------------------------
    print("\n" + "-" * 60)
    print("  [1] INPUT FILES")
    print("-" * 60)
    
    if args.pdb and file_exists(args.pdb):
        config.pdb_file = args.pdb
        print(f"  ✓ PDB: {args.pdb}")
    elif args.pdb:
        print(f"  ✗ PDB not found: {args.pdb}")
        config.pdb_file = None
    else:
        print("  No PDB provided - will create water box only")
        use_pdb = get_yn("Load a PDB file?", default=False)
        if use_pdb:
            while True:
                pdb_path = input("  Enter PDB path: ").strip()
                if file_exists(pdb_path):
                    config.pdb_file = pdb_path
                    print(f"  ✓ PDB: {pdb_path}")
                    break
                print("    ⚠ File not found")
                if not get_yn("Try again?", default=True):
                    break
    
    # PSF file
    if args.psf and file_exists(args.psf):
        config.psf_file = args.psf
        print(f"  ✓ PSF: {args.psf}")
    elif config.pdb_file:
        use_psf = get_yn("Use PSF topology file?", default=False)
        if use_psf:
            while True:
                psf_path = input("  Enter PSF path: ").strip()
                if file_exists(psf_path):
                    config.psf_file = psf_path
                    break
                if not get_yn("File not found. Try again?", default=True):
                    break
    
    # -------------------------
    # SECTION 2: FORCE FIELD
    # -------------------------
    print("\n" + "-" * 60)
    print("  [2] FORCE FIELD")
    print("-" * 60)
    
    if args.ff:
        config.ff_files = args.ff
        config.water_model = args.water
        print(f"  ✓ FF: {config.ff_files}")
        print(f"  ✓ Water: {config.water_model}")
    else:
        print("  Available presets:")
        for key, preset in FF_PRESETS.items():
            print(f"    [{key}] {preset['name']}")
        print("    [C] Custom (enter files manually)")
        
        choice = input("  Select [1]: ").strip().upper() or '1'
        
        if choice == 'C':
            ff_input = input("  Enter FF files (space-separated): ").strip()
            config.ff_files = ff_input.split()
            config.water_model = input("  Water model name [tip3p]: ").strip() or 'tip3p'
        elif choice in FF_PRESETS:
            preset = FF_PRESETS[choice]
            config.ff_files = preset['files']
            config.water_model = preset['water']
            print(f"  ✓ Selected: {preset['name']}")
        else:
            print("  ⚠ Invalid, using default (AMBER14 + TIP3P)")
    
    # -------------------------
    # SECTION 3: BOX DIMENSIONS
    # -------------------------
    print("\n" + "-" * 60)
    print("  [3] BOX DIMENSIONS (nanometers)")
    print("-" * 60)
    
    if config.pdb_file:
        auto_box = get_yn("Auto-calculate box from molecule?", default=True)
        if auto_box:
            config.padding = get_float("Padding around molecule (nm)", 1.0, 0.5, 5.0)
            print(f"  ✓ Will add {config.padding} nm padding")
        else:
            config.box_x = get_float("Box X (nm)", 8.0, 2.0, 100.0)
            config.box_y = get_float("Box Y (nm)", 8.0, 2.0, 100.0)
            config.box_z = get_float("Box Z (nm)", 8.0, 2.0, 100.0)
    else:
        config.box_x = get_float("Box X (nm)", 5.0, 2.0, 50.0)
        config.box_y = get_float("Box Y (nm)", 5.0, 2.0, 50.0)
        config.box_z = get_float("Box Z (nm)", 5.0, 2.0, 50.0)
    
    # -------------------------
    # SECTION 4: CUTOFF
    # -------------------------
    print("\n" + "-" * 60)
    print("  [4] NONBONDED CUTOFF")
    print("-" * 60)
    
    config.cutoff = get_float("Cutoff distance (nm)", 1.0, 0.8, 1.4)
    
    # -------------------------
    # SECTION 5: CONDITIONS
    # -------------------------
    print("\n" + "-" * 60)
    print("  [5] SIMULATION CONDITIONS")
    print("-" * 60)
    
    config.temperature = get_float("Temperature (K)", 300.0, 100.0, 500.0)
    config.pressure = get_float("Pressure (bar)", 1.0, 0.1, 10.0)
    config.timestep = get_float("Timestep (fs)", 2.0, 0.5, 4.0)
    
    # -------------------------
    # SECTION 6: SIMULATION TIME
    # -------------------------
    print("\n" + "-" * 60)
    print("  [6] SIMULATION TIME (nanoseconds)")
    print("-" * 60)
    
    config.nvt_time_ns = get_float("NVT equilibration (ns)", 1.0, 0.001, 1000.0)
    config.npt_time_ns = get_float("NPT production (ns)", 10.0, 0.01, 10000.0)
    
    # -------------------------
    # SECTION 7: OUTPUT FREQUENCY
    # -------------------------
    print("\n" + "-" * 60)
    print("  [7] OUTPUT FREQUENCY (picoseconds)")
    print("-" * 60)
    
    config.log_freq_ps = get_float("Log file write (ps)", 1.0, 0.01, 1000.0)
    config.save_freq_ps = get_float("Trajectory save (ps)", 10.0, 0.1, 10000.0)
    config.print_freq_ps = get_float("Screen print (ps)", 100.0, 1.0, 100000.0)
    
    # -------------------------
    # SECTION 8: ADVANCED OPTIONS
    # -------------------------
    print("\n" + "-" * 60)
    print("  [8] ADVANCED OPTIONS")
    print("-" * 60)
    
    # Checkpointing
    config.use_checkpoints = get_yn("Enable checkpointing?", default=False)
    if config.use_checkpoints:
        config.checkpoint_freq_ps = get_float("Checkpoint frequency (ps)", 100.0, 10.0, 10000.0)
    
    # Analysis
    config.run_analysis = get_yn("Run analysis after simulation?", default=True)
    if config.run_analysis:
        config.calc_hbonds = get_yn("Calculate hydrogen bonds?", default=False)
    
    # Save configuration
    if get_yn("Save this configuration for later use?", default=False):
        save_path = input("  Save as [config.yaml]: ").strip() or "config.yaml"
        config.save(save_path)
    
    # Platform
    config.platform = args.platform
    
    # Calculate steps
    config.calculate_steps()
    
    return config


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def print_header():
    """Print header"""
    print("\n" + "=" * 60)
    print("        ENHANCED OpenMM MD SIMULATION")
    print("=" * 60)
    print(f"  OpenMM Version: {Platform.getOpenMMVersion()}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def print_config(config: SimConfig):
    """Print configuration summary"""
    
    total_ns = config.nvt_time_ns + config.npt_time_ns
    total_steps = config.nvt_steps + config.npt_steps
    
    print("\n" + "=" * 60)
    print("  CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print("\n  INPUT FILES:")
    print(f"    PDB File:       {config.pdb_file or 'None (water box)'}")
    print(f"    PSF File:       {config.psf_file or 'None'}")
    print(f"    Force Field:    {config.ff_files}")
    print(f"    Water Model:    {config.water_model}")
    
    print("\n  BOX:")
    if config.padding:
        print(f"    Mode:           Auto (padding: {config.padding} nm)")
    else:
        print(f"    Dimensions:     {config.box_x} x {config.box_y} x {config.box_z} nm")
    
    print("\n  CONDITIONS:")
    print(f"    Temperature:    {config.temperature} K")
    print(f"    Pressure:       {config.pressure} bar")
    print(f"    Timestep:       {config.timestep} fs")
    print(f"    Cutoff:         {config.cutoff} nm")
    
    print("\n  SIMULATION TIME:")
    print(f"    NVT:            {config.nvt_time_ns} ns  ({config.nvt_steps:,} steps)")
    print(f"    NPT:            {config.npt_time_ns} ns  ({config.npt_steps:,} steps)")
    print(f"    TOTAL:          {total_ns} ns  ({total_steps:,} steps)")
    
    print("\n  OUTPUT FREQUENCY:")
    print(f"    Log:            every {config.log_freq_ps} ps  ({config.log_every} steps)")
    print(f"    Trajectory:     every {config.save_freq_ps} ps  ({config.save_every} steps)")
    print(f"    Screen:         every {config.print_freq_ps} ps  ({config.print_every} steps)")
    
    print("\n  ADVANCED OPTIONS:")
    print(f"    Checkpointing:  {'Enabled' if config.use_checkpoints else 'Disabled'}")
    if config.use_checkpoints:
        print(f"    Checkpoint freq: every {config.checkpoint_freq_ps} ps")
    print(f"    Run Analysis:   {'Yes' if config.run_analysis else 'No'}")
    print(f"    H-bond Calc:    {'Yes' if config.calc_hbonds else 'No'}")
    
    print(f"\n  PLATFORM:         {config.platform}")
    print("=" * 60)


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================
def create_output_dir(config: SimConfig) -> str:
    """Create output directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if config.pdb_file:
        name = os.path.splitext(os.path.basename(config.pdb_file))[0]
    else:
        name = f"water_{config.water_model}"
    
    config.output_dir = f"sim_{name}_{timestamp}"
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"\n  Output directory: {config.output_dir}")
    return config.output_dir


def load_forcefield(config: SimConfig):
    """Load force field"""
    print("\n" + "-" * 60)
    print("  Loading Force Field...")
    
    ff = ForceField(*config.ff_files)
    print(f"  ✓ Loaded: {config.ff_files}")
    return ff


def load_or_create_structure(config: SimConfig, ff):
    """Load PDB or create water box"""
    print("\n" + "-" * 60)
    print("  Creating System Structure...")
    
    if config.pdb_file:
        pdb = PDBFile(config.pdb_file)
        print(f"  ✓ Loaded PDB: {config.pdb_file}")
        
        if config.psf_file:
            try:
                from openmm.app import CharmmPsfFile
                psf = CharmmPsfFile(config.psf_file)
                topology = psf.topology
                print(f"  ✓ Loaded PSF topology: {config.psf_file}")
            except Exception as e:
                print(f"  ⚠ PSF load failed: {e}")
                topology = pdb.topology
        else:
            topology = pdb.topology
        
        positions = pdb.positions
        modeller = Modeller(topology, positions)
        
        if config.padding:
            print(f"  Adding solvent (padding: {config.padding} nm)...")
            modeller.addSolvent(
                ff,
                model=config.water_model,
                padding=config.padding * nanometer
            )
        else:
            print(f"  Adding solvent ({config.box_x}x{config.box_y}x{config.box_z} nm)...")
            modeller.addSolvent(
                ff,
                model=config.water_model,
                boxSize=Vec3(config.box_x, config.box_y, config.box_z) * nanometer
            )
    else:
        print(f"  Creating water box: {config.box_x}x{config.box_y}x{config.box_z} nm")
        modeller = Modeller(Topology(), [] * nanometer)
        modeller.addSolvent(
            ff,
            model=config.water_model,
            boxSize=Vec3(config.box_x, config.box_y, config.box_z) * nanometer
        )
    
    topology = modeller.getTopology()
    positions = modeller.getPositions()
    
    n_atoms = topology.getNumAtoms()
    n_residues = topology.getNumResidues()
    n_waters = len([r for r in topology.residues() if r.name in ['HOH', 'WAT', 'TIP3', 'SOL']])
    
    print(f"  ✓ Total atoms:    {n_atoms:,}")
    print(f"  ✓ Total residues: {n_residues:,}")
    print(f"  ✓ Water molecules: {n_waters:,}")
    
    return topology, positions


def create_system(config: SimConfig, ff, topology):
    """Create OpenMM system"""
    print("\n" + "-" * 60)
    print("  Creating OpenMM System...")
    
    system = ff.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=config.cutoff * nanometer,
        constraints=HBonds,
        rigidWater=True
    )
    
    print(f"  ✓ Nonbonded: PME")
    print(f"  ✓ Cutoff: {config.cutoff} nm")
    print(f"  ✓ Constraints: HBonds")
    print(f"  ✓ Rigid water: Yes")
    
    return system


def create_simulation(config: SimConfig, topology, system, positions):
    """Create simulation object"""
    print("\n" + "-" * 60)
    print("  Creating Simulation...")
    
    integrator = LangevinIntegrator(
        config.temperature * kelvin,
        1.0 / picosecond,
        config.timestep * femtosecond
    )
    
    try:
        platform = Platform.getPlatformByName(config.platform)
        print(f"  ✓ Platform: {config.platform}")
    except:
        platform = Platform.getPlatformByName('CPU')
        print(f"  ⚠ {config.platform} unavailable, using CPU")
    
    sim = Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    
    print(f"  ✓ Integrator: Langevin")
    print(f"  ✓ Temperature: {config.temperature} K")
    print(f"  ✓ Friction: 1.0 /ps")
    
    return sim


def run_minimization(sim, config: SimConfig, topology):
    """Energy minimization"""
    print("\n" + "-" * 60)
    print("  Energy Minimization...")
    
    state = sim.context.getState(getEnergy=True)
    e_initial = state.getPotentialEnergy()
    print(f"  Initial energy: {e_initial}")
    
    sim.minimizeEnergy()
    
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e_final = state.getPotentialEnergy()
    print(f"  Final energy:   {e_final}")
    
    pdb_path = os.path.join(config.output_dir, '01_minimized.pdb')
    with open(pdb_path, 'w') as f:
        PDBFile.writeFile(topology, state.getPositions(), f)
    print(f"  ✓ Saved: {pdb_path}")


def run_nvt(sim, config: SimConfig, topology, checkpoint_mgr=None):
    """NVT equilibration with optional checkpointing"""
    print("\n" + "-" * 60)
    print(f"  NVT Equilibration: {config.nvt_time_ns} ns ({config.nvt_steps:,} steps)")
    print("-" * 60)
    
    sim.context.setVelocitiesToTemperature(config.temperature * kelvin)
    
    log_path = os.path.join(config.output_dir, '02_nvt.log')
    dcd_path = os.path.join(config.output_dir, '02_nvt.dcd')
    
    sim.reporters = [
        StateDataReporter(log_path, config.log_every,
                          step=True, time=True, temperature=True,
                          potentialEnergy=True, kineticEnergy=True,
                          totalEnergy=True, speed=True),
        StateDataReporter(sys.stdout, config.print_every,
                          step=True, temperature=True, speed=True),
        DCDReporter(dcd_path, config.save_every)
    ]
    
    print()
    
    # Run with checkpointing
    if config.use_checkpoints and checkpoint_mgr:
        steps_done = 0
        while steps_done < config.nvt_steps:
            steps_to_run = min(config.checkpoint_every, config.nvt_steps - steps_done)
            sim.step(steps_to_run)
            steps_done += steps_to_run
            
            if steps_done % config.checkpoint_every == 0 and steps_done < config.nvt_steps:
                checkpoint_mgr.save_checkpoint(sim, 'nvt', steps_done)
    else:
        sim.step(config.nvt_steps)
    
    print()
    
    state = sim.context.getState(getPositions=True)
    pdb_path = os.path.join(config.output_dir, '02_nvt_final.pdb')
    with open(pdb_path, 'w') as f:
        PDBFile.writeFile(topology, state.getPositions(), f)
    
    print(f"  ✓ Log: {log_path}")
    print(f"  ✓ Trajectory: {dcd_path}")
    print(f"  ✓ Structure: {pdb_path}")


def run_npt(sim, config: SimConfig, system, topology, checkpoint_mgr=None):
    """NPT production with optional checkpointing"""
    print("\n" + "-" * 60)
    print(f"  NPT Production: {config.npt_time_ns} ns ({config.npt_steps:,} steps)")
    print("-" * 60)
    
    barostat = MonteCarloBarostat(
        config.pressure * bar,
        config.temperature * kelvin
    )
    system.addForce(barostat)
    sim.context.reinitialize(preserveState=True)
    print(f"  ✓ Barostat: {config.pressure} bar")
    
    log_path = os.path.join(config.output_dir, '03_production.log')
    dcd_path = os.path.join(config.output_dir, '03_production.dcd')
    
    sim.reporters = [
        StateDataReporter(log_path, config.log_every,
                          step=True, time=True, temperature=True,
                          potentialEnergy=True, kineticEnergy=True,
                          totalEnergy=True, density=True, volume=True,
                          speed=True),
        StateDataReporter(sys.stdout, config.print_every,
                          step=True, time=True, temperature=True,
                          density=True, speed=True),
        DCDReporter(dcd_path, config.save_every)
    ]
    
    print()
    
    # Run with checkpointing
    if config.use_checkpoints and checkpoint_mgr:
        steps_done = 0
        while steps_done < config.npt_steps:
            steps_to_run = min(config.checkpoint_every, config.npt_steps - steps_done)
            sim.step(steps_to_run)
            steps_done += steps_to_run
            
            if steps_done % config.checkpoint_every == 0 and steps_done < config.npt_steps:
                checkpoint_mgr.save_checkpoint(sim, 'npt', steps_done)
    else:
        sim.step(config.npt_steps)
    
    print()
    
    state = sim.context.getState(getPositions=True)
    pdb_path = os.path.join(config.output_dir, '03_final.pdb')
    with open(pdb_path, 'w') as f:
        PDBFile.writeFile(topology, state.getPositions(), f)
    
    print(f"  ✓ Log: {log_path}")
    print(f"  ✓ Trajectory: {dcd_path}")
    print(f"  ✓ Structure: {pdb_path}")


def save_info(config: SimConfig, n_atoms: int = 0):
    """Save simulation info"""
    info_path = os.path.join(config.output_dir, 'simulation_info.txt')
    
    with open(info_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("SIMULATION CONFIGURATION\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Date: {datetime.now()}\n\n")
        
        f.write("INPUT FILES:\n")
        f.write(f"  PDB: {config.pdb_file}\n")
        f.write(f"  PSF: {config.psf_file}\n")
        f.write(f"  Force Field: {config.ff_files}\n")
        f.write(f"  Water Model: {config.water_model}\n\n")
        
        f.write("BOX:\n")
        if config.padding:
            f.write(f"  Padding: {config.padding} nm\n\n")
        else:
            f.write(f"  Dimensions: {config.box_x} x {config.box_y} x {config.box_z} nm\n\n")
        
        f.write("CONDITIONS:\n")
        f.write(f"  Temperature: {config.temperature} K\n")
        f.write(f"  Pressure: {config.pressure} bar\n")
        f.write(f"  Timestep: {config.timestep} fs\n")
        f.write(f"  Cutoff: {config.cutoff} nm\n\n")
        
        f.write("SIMULATION:\n")
        f.write(f"  NVT: {config.nvt_time_ns} ns ({config.nvt_steps} steps)\n")
        f.write(f"  NPT: {config.npt_time_ns} ns ({config.npt_steps} steps)\n")
        f.write(f"  Total: {config.nvt_time_ns + config.npt_time_ns} ns\n\n")
        
        f.write("OUTPUT:\n")
        f.write(f"  Log every: {config.log_freq_ps} ps ({config.log_every} steps)\n")
        f.write(f"  Save every: {config.save_freq_ps} ps ({config.save_every} steps)\n")
        f.write(f"  Platform: {config.platform}\n\n")
        
        f.write("ADVANCED:\n")
        f.write(f"  Checkpointing: {config.use_checkpoints}\n")
        f.write(f"  Analysis: {config.run_analysis}\n")
        f.write(f"  H-bonds: {config.calc_hbonds}\n")
    
    # Also save as YAML/JSON for programmatic access
    config.save(os.path.join(config.output_dir, 'config.yaml'))
    
    print(f"  ✓ Info: {info_path}")


def print_completion(config: SimConfig):
    """Print completion message"""
    print("\n" + "=" * 60)
    print("  ✓ SIMULATION COMPLETE!")
    print("=" * 60)
    print(f"  Output: {config.output_dir}/")
    print("\n  Files:")
    
    for f in sorted(os.listdir(config.output_dir)):
        fpath = os.path.join(config.output_dir, f)
        if os.path.isdir(fpath):
            n_files = len(os.listdir(fpath))
            print(f"    {f}/                                     ({n_files} files)")
        else:
            size = os.path.getsize(fpath)
            
            if size >= 1024 * 1024 * 1024:
                s = f"{size / (1024**3):.1f} GB"
            elif size >= 1024 * 1024:
                s = f"{size / (1024**2):.1f} MB"
            elif size >= 1024:
                s = f"{size / 1024:.1f} KB"
            else:
                s = f"{size} B"
            
            print(f"    {f:40s} {s:>10s}")
    
    print("\n" + "=" * 60)


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main entry point"""
    
    print_header()
    
    # Parse arguments
    args = parse_args()
    
    # =========================================
    # ANALYZE ONLY MODE
    # =========================================
    if args.analyze_only:
        print("\n" + "=" * 60)
        print("  ANALYSIS ONLY MODE")
        print("=" * 60)
        
        output_dir = input("  Enter simulation output directory: ").strip()
        if not os.path.exists(output_dir):
            print(f"  ✗ Directory not found: {output_dir}")
            sys.exit(1)
        
        # Run analysis
        analyzer = SimulationAnalyzer(output_dir)
        analyzer.run_full_analysis()
        
        # Optional H-bond analysis
        if get_yn("\n  Run H-bond analysis?", default=False):
            hbond_analyzer = HBondAnalyzer(output_dir)
            hbond_analyzer.analyze_trajectory()
        
        print("\n  ✓ Analysis complete!")
        sys.exit(0)
    
    # =========================================
    # LOAD FROM CONFIG FILE
    # =========================================
    if args.config and file_exists(args.config):
        config = SimConfig.load(args.config)
        print(f"\n  ✓ Loaded configuration from: {args.config}")
        config.platform = args.platform
        config.calculate_steps()
    # =========================================
    # BATCH MODE
    # =========================================
    elif args.batch:
        config = SimConfig()
        if args.pdb and file_exists(args.pdb):
            config.pdb_file = args.pdb
        if args.psf and file_exists(args.psf):
            config.psf_file = args.psf
        if args.ff:
            config.ff_files = args.ff
        config.water_model = args.water
        config.platform = args.platform
        config.calculate_steps()
    # =========================================
    # INTERACTIVE MODE
    # =========================================
    else:
        config = interactive_setup(args)
    
    # Print summary
    print_config(config)
    
    # Confirm
    if not args.batch:
        if not get_yn("\n  Proceed with simulation?", default=True):
            print("\n  Cancelled.\n")
            sys.exit(0)
    
    # Create output directory
    create_output_dir(config)
    
    # Initialize checkpoint manager
    checkpoint_mgr = None
    if config.use_checkpoints:
        checkpoint_mgr = CheckpointManager(config.output_dir)
        print(f"  ✓ Checkpointing enabled")
    
    try:
        # Load force field
        ff = load_forcefield(config)
        
        # Create structure
        topology, positions = load_or_create_structure(config, ff)
        
        # Save initial
        pdb_path = os.path.join(config.output_dir, '00_initial.pdb')
        with open(pdb_path, 'w') as f:
            PDBFile.writeFile(topology, positions, f)
        print(f"  ✓ Initial structure: {pdb_path}")
        
        # Create system
        system = create_system(config, ff, topology)
        
        # Create simulation
        sim = create_simulation(config, topology, system, positions)
        
        # Run stages
        run_minimization(sim, config, topology)
        run_nvt(sim, config, topology, checkpoint_mgr)
        run_npt(sim, config, system, topology, checkpoint_mgr)
        
        # Save info
        save_info(config)
        
        # Print completion
        print_completion(config)
        
        # =========================================
        # RUN ANALYSIS (if enabled)
        # =========================================
        if config.run_analysis:
            analyzer = SimulationAnalyzer(config.output_dir)
            analyzer.run_full_analysis()
            
            if config.calc_hbonds:
                hbond_analyzer = HBondAnalyzer(config.output_dir)
                hbond_analyzer.analyze_trajectory()
        
        print("\n" + "=" * 60)
        print("  ✓ ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

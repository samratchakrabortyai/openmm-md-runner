# OpenMM MD Runner 🧬

<p align="center">
  <img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/OpenMM-7.5+-green.svg" alt="OpenMM">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/github/stars/samratchakrabortyai/openmm-md-runner?style=social" alt="GitHub stars">
</p>

<p align="center">
  <b>A comprehensive, user-friendly molecular dynamics simulation toolkit built on OpenMM</b>
</p>

---

## ✨ Features

- **🎯 Interactive Mode** - Step-by-step guided setup for beginners
- **📁 Configuration Files** - Save and reuse simulation parameters (YAML/JSON)
- **💾 Checkpointing** - Save progress and restart failed simulations
- **📊 Advanced Analysis** - 9+ publication-quality plot types
- **🔗 H-bond Analysis** - Automatic hydrogen bond counting across trajectory
- **📈 Energy Tracking** - Drift analysis, fluctuations, and convergence metrics
- **📄 HTML Reports** - Professional analysis summaries with all plots
- **🚀 Multiple Modes** - Interactive, batch, and analysis-only workflows

## 🚀 Quick Start

### Installation

# Clone the repository
git clone https://github.com/samratchakrabortyai/openmm-md-runner.git
cd openmm-md-runner

# Install dependencies
pip install openmm numpy matplotlib pyyaml
# Interactive mode (recommended for first-time users)
python3 md_runner.py

# Run with a PDB file
python3 md_runner.py protein.pdb

# Use a configuration file
python3 md_runner.py --config simulation.yaml

# Analysis only (on completed simulation)
python3 md_runner.py --analyze-only
Example 1: Water Box Simulation


python3 md_runner.py
# Follow the prompts to create a simple water box
Example 2: Protein Simulation
Bash

python3 md_runner.py my_protein.pdb
# The script will guide you through solvation and equilibration
Example 3: Batch Mode with Config


# Create config.yaml with your parameters
python3 md_runner.py --config config.yaml --batch
📊 Analysis Outputs
The tool automatically generates:

Energy Components - Potential, kinetic, and total energy evolution
Temperature Profiles - With mean and standard deviation
Energy Drift - Relative energy conservation in ppm
Standard Deviation Convergence - Energy equilibration check
Energy Fluctuations - Instantaneous deviations from mean
Density & Volume - NPT equilibration metrics
Temperature Distribution - Statistical analysis
Equilibration Summary - Combined NVT/NPT overview
Combined Energy Timeline - Full simulation profile
H-bond Analysis - Hydrogen bond counts (optional)
All plots are compiled into a beautiful HTML report with statistics.

📁 Output Structure
text

sim_protein_20250115_143022/
├── 00_initial.pdb              # Starting structure
├── 01_minimized.pdb            # Energy minimized
├── 02_nvt.log                  # NVT equilibration data
├── 02_nvt.dcd                  # NVT trajectory
├── 02_nvt_final.pdb            # NVT final structure
├── 03_production.log           # NPT production data
├── 03_production.dcd           # NPT trajectory
├── 03_final.pdb                # Final structure
├── config.yaml                 # Saved configuration
├── simulation_info.txt         # Simulation summary
├── analysis_report.html        # ⭐ Open this in browser!
└── analysis_plots/             # All PNG plots
    ├── nvt_energy_components.png
    ├── npt_energy_drift.png
    ├── equilibration_summary.png
    └── ... (9+ plots total)
🔧 Advanced Features
Configuration File Example
Create my_simulation.yaml:

YAML

pdb_file: protein.pdb
padding: 1.2
temperature: 310.0
nvt_time_ns: 2.0
npt_time_ns: 50.0
run_analysis: true
calc_hbonds: true
Run:

Bash

python3 md_runner.py --config my_simulation.yaml
Force Field Options
AMBER14 + TIP3P/SPC-E/TIP4P-Ew
AMBER99SB-ILDN + TIP3P
CHARMM36 + TIP3P/TIP4P-Ew
Custom XML files
Platform Selection
Bash

python3 md_runner.py --platform CPU    # Default
python3 md_runner.py --platform CUDA   # NVIDIA GPU
python3 md_runner.py --platform OpenCL # AMD/Intel GPU
📋 Requirements
Python 3.7+
OpenMM >= 7.5.0
NumPy >= 1.19.0
Matplotlib >= 3.3.0
PyYAML >= 5.0 (optional, for YAML config files)
Install all at once:

Bash

pip install -r requirements.txt
🐛 Troubleshooting
OpenMM not found?

Bash

conda install -c conda-forge openmm
CUDA not available?

Bash

conda install -c conda-forge openmm cudatoolkit
Import errors?

Bash

pip install --upgrade numpy matplotlib pyyaml
📚 Documentation
Command Line Options
text

usage: md_runner.py [-h] [--psf PSF] [--ff FF [FF ...]]
                    [--water {tip3p,tip4pew,tip5p,spce}]
                    [--platform {CPU,CUDA,OpenCL}]
                    [--config CONFIG] [--batch] [--analyze-only]
                    [pdb]

positional arguments:
  pdb                   PDB file (optional)

optional arguments:
  -h, --help            show this help message and exit
  --psf PSF, -p PSF     PSF file (optional)
  --ff FF [FF ...]      Force field XML files
  --water {tip3p,tip4pew,tip5p,spce}
                        Water model
  --platform {CPU,CUDA,OpenCL}
                        Compute platform
  --config CONFIG, -c CONFIG
                        Load configuration from YAML/JSON
  --batch, -b           Batch mode (no prompts)
  --analyze-only, -a    Run analysis only on existing simulation
🤝 Contributing
Contributions, issues, and feature requests are welcome!

Feel free to check the issues page.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Samrat Chakraborty

GitHub: @samratchakrabortyai
Repository: openmm-md-runner
🙏 Acknowledgments
Built with OpenMM - High-performance MD toolkit
Force fields from the OpenMM community
Inspired by the need for accessible molecular dynamics tools
📖 Citation
If you use this tool in your research, please cite:

bibtex

@software{chakraborty2025openmm,
  author = {Chakraborty, Samrat},
  title = {OpenMM MD Runner: Enhanced Molecular Dynamics Simulation Toolkit},
  year = {2025},
  url = {https://github.com/samratchakrabortyai/openmm-md-runner},
  version = {1.0.0}
}
⭐ Star This Repository
If you find this tool useful, please consider giving it a star! It helps others discover the project.



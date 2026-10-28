# 🚀 Rocket Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Pygame 2.5+](https://img.shields.io/badge/pygame-2.5+-4B8BBE.svg?style=flat-square&logo=python&logoColor=white)](https://www.pygame.org/)
[![NumPy](https://img.shields.io/badge/numpy-64--bit_precision-013243.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/tests-117%20passed-brightgreen.svg?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

An interactive, high-precision **2D Aerospace & Orbital Mechanics Flight Simulator** built in Python, Pygame, and NumPy. Design multi-stage rockets in the **Vehicle Assembly Building (VAB)**, launch into orbit with realistic aerodynamics and gravity fields, and monitor missions with real-time flight telemetry.

---

## ⚡ Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LFM57/rocket-simulation.git
   cd "Rocket Simulation"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   # or: .venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulator

```bash
# Open Vehicle Assembly Building (VAB) editor (Default)
python main.py

# Launch directly onto the pad with the Falcon 9 preset
python main.py --launch

# Optional flags: custom resolution or fullscreen
python main.py --width 1600 --height 900
python main.py --fullscreen
```

---

## 🎮 Controls & Keybindings

### 🚀 Flight View & Orbital Map

| Key / Input | Action |
|:---|:---|
| <kbd>Space</kbd> | **Stage Rocket** (Ignite engines / Decouple active stage) |
| <kbd>Z</kbd> / <kbd>X</kbd> | **Full Throttle** (100%) / **Cut Throttle** (0%) |
| <kbd>Shift</kbd> / <kbd>Ctrl</kbd> (or <kbd>↑</kbd> / <kbd>↓</kbd>) | **Throttle Up** / **Throttle Down** (±5%) |
| <kbd>A</kbd> / <kbd>D</kbd> (or <kbd>←</kbd> / <kbd>→</kbd>) | **Steer CCW** / **Steer CW** (Gimbal / Pitch) |
| <kbd>M</kbd> | **Toggle Orbital Map** / Flight View |
| <kbd>C</kbd> | **Toggle Camera Mode** (Horizon-Up $\leftrightarrow$ Inertial) |
| <kbd>.</kbd> / <kbd>,</kbd> | **Increase / Decrease Time Warp** (1x to 1000x) |
| <kbd>R</kbd> | **Revert Flight** to launchpad |
| <kbd>V</kbd> or <kbd>Esc</kbd> | **Toggle Post-Flight Telemetry Modal** |
| <kbd>Scroll Wheel</kbd> | **Zoom In / Out** |
| <kbd>Right Click Drag</kbd> | **Pan Camera** |

### 🛠️ Vehicle Assembly Building (VAB)

| Key / Input | Action |
|:---|:---|
| <kbd>Space</kbd> or <kbd>Enter</kbd> | **Launch Rocket** to flight view |
| <kbd>Delete</kbd> | **Delete Selected Part** |
| <kbd>Left Click</kbd> | **Select Part** for property inspection |
| <kbd>C</kbd> | **Recenter Blueprint Canvas** |
| <kbd>Scroll Wheel</kbd> | **Zoom Canvas** |
| <kbd>Middle / Right Click</kbd> | **Pan Canvas** |
| **Top Menu** | Load Falcon 9 preset, Clear stack, Save/Load blueprint JSON |

---

## ✨ Features

- **Vehicle Assembly Building (VAB)**: Modular blueprint editor with part snapping, Merlin 1D / MVac engines, fuel tanks, decouplers, and payload fairings. Displays live Center of Mass (CoM), Center of Pressure (CoP), stage $\Delta v$, and Thrust-to-Weight Ratios (TWR).
- **Realistic Flight Dynamics**: Sub-stepped 4th-order Runge-Kutta (RK4) integration, variable angle-of-attack drag and lift, atmospheric lapse rate and pressure profiles, and thrust vectoring control (gimbaling).
- **Astrodynamics & Gravity Fields**: Inverse-square $1/r^2$ gravity fields for Earth and Moon with planetary rotation, real-time Keplerian orbital decomposition ($r_a, r_p, a, e$), and trajectory conic projections.
- **Dual-Mode Time Warp**: Sub-stepped physics warp (1x–4x) for powered flight and atmospheric passes; analytical Keplerian on-rails warp (10x–1000x) for vacuum coasting with automatic atmospheric safety drop.
- **SAS Autopilot**: PID attitude controller with `HOLD_ATTITUDE`, `PROGRADE`, and `RETROGRADE` vector tracking modes.
- **Telemetry & Mission Analysis**: Anti-jitter 64-bit camera, dynamic pressure tracking (Max-Q), G-force monitoring, post-flight multi-channel plots, and CSV export.

---

## 📐 Physics Foundations

- **Planar 3-DOF Equations of Motion**:
  $$\mathbf{a} = \frac{\mathbf{F}_{\text{thrust}} + \mathbf{F}_{\text{aero}}}{m(t)} + \mathbf{g}(\mathbf{r}), \quad \alpha = \frac{\tau_{\text{thrust}} + \tau_{\text{aero}}}{I_{zz}}$$
- **Thrust & Mass Flow**:
  Engine thrust scales with ambient barometric pressure:
  $$F(p_{\text{amb}}) = F_{\text{vac}} - \left(\frac{p_{\text{amb}}}{p_0}\right)(F_{\text{vac}} - F_{\text{sl}}), \quad \dot{m} = \frac{F_{\text{vac}}}{g_0 I_{\text{sp,vac}}}$$
- **Stage Delta-V (Tsiolkovsky Rocket Equation)**:
  $$\Delta v_i = g_0 I_{\text{sp}}^{(i)} \ln\left(\frac{m_{0,i}}{m_{f,i}}\right)$$
- **Keplerian Orbital Conics**:
  Specific orbital energy $\varepsilon$ and eccentricity vector $\mathbf{e}$ are extracted from instantaneous position $\mathbf{r}$ and velocity $\mathbf{v}$ for exact conic trajectory rendering and fast analytical propagation:
  $$\varepsilon = \frac{|\mathbf{v}|^2}{2} - \frac{\mu}{|\mathbf{r}|}, \quad a = -\frac{\mu}{2\varepsilon}, \quad r_a = a(1 + |\mathbf{e}|), \quad r_p = a(1 - |\mathbf{e}|)$$

---

## 🧪 Testing

Run the automated test suite with pytest:

```bash
pytest
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

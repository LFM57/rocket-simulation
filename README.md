# 🚀 Spaceflight Simulator (SFS-2D)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Pygame 2.5+](https://img.shields.io/badge/pygame-2.5+-4B8BBE.svg?style=flat-square&logo=python&logoColor=white)](https://www.pygame.org/)
[![NumPy](https://img.shields.io/badge/numpy-64--bit_precision-013243.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/tests-99%20passed-brightgreen.svg?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![Architecture 6-DOF / 3-DOF](https://img.shields.io/badge/dynamics-Planar%203--DOF%20%2F%206--DOF-blueviolet.svg?style=flat-square)]()
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

An interactive, high-precision **2D Aerospace & Orbital Mechanics Flight Simulator** featuring an interactive **Vehicle Assembly Building (VAB)** blueprint editor, 64-bit numerical physics integration, dual-mode time warp (RK4 Physics & Analytical On-Rails Keplerian Warp), compressible barometric aerodynamics, PID-driven Stability Augmentation (SAS), and real-time mission telemetry strip charts.

---

## 📸 Overview

```text
       +-------------------------------------------------------------------------+
       | [VAB Blueprint Editor]                   [Real-Time Flight View]        |
       | +--------------------+-----------------+ +----------------------------+ |
       | | Part Catalog       | Rocket Canvas   | | Visual: Particles & Plumes | |
       | | - Merlin 1D / MVac | Auto-Snapping   | | HUD: Altitude, V-Speed, Q  | |
       | | - Modular Tanks    | Live CoM & CoP  | | SAS: Prograde / Retrograde | |
       | | - Decouplers       | TWR Rollups     | | Map: Kepler Orbit Conics   | |
       | | - Fairing/Capsule  | Stage Delta-V   | | Warp: 1x - 1000x Dual-Mode | |
       | +--------------------+-----------------+ +----------------------------+ |
       +-------------------------------------------------------------------------+
```

Spaceflight Simulator (SFS-2D) couples orbital mechanics and multi-stage rocket engineering with an intuitive, modern aerospace dark-themed GUI written entirely in Python, Pygame, and NumPy.

---

## 🌟 Key Highlights

### 🪐 Planetary Gravity & Astrodynamics
- **Newtonian $1/r^2$ Gravity Field**: Gravitational models for Earth ($M = 5.972 \times 10^{24}\text{ kg}, R = 6,371\text{ km}$) and Moon ($M = 7.342 \times 10^{22}\text{ kg}, R = 1,737\text{ km}$) with planetary rotation and surface co-rotation velocity clamps.
- **Orbital State Decomposition**: Instantaneous extraction of classical Keplerian orbital elements:
  - Apoapsis ($r_a$) & Periapsis ($r_p$)
  - Semi-Major Axis ($a$) & Orbital Eccentricity ($e$)
  - True Anomaly ($\nu$) & Flight Path Angle ($\gamma$)
  - Specific Orbital Energy ($\varepsilon$) & Orbital Period ($T$)
- **Kepler Universal Anomaly Propagation**: Exact analytical on-rails conic trajectory propagation solving Kepler's equation for fast-forwarding coast phases in vacuum.

### 🌪️ Compressible Aerodynamics & Atmosphere
- **Barometric Atmosphere**: Continuous exponential air density profile with smooth cubic tapering to the $100\text{ km}$ Kármán line.
- **Lapse Rate Temperature Model**: Tropospheric lapse rate with stratospheric isothermal clamp, local speed of sound $c = \sqrt{\gamma R T}$, and ambient pressure calculations.
- **Cross-Flow Aerodynamics**:
  - Angle-of-attack dependent drag: $C_d(\alpha) = C_{d0} + (C_{d,\text{max}} - C_{d0})\sin^2\alpha$
  - Slender-body lift: $C_L(\alpha) = C_{L\alpha}\sin(2\alpha)$
  - Dynamic pressure ($q = \frac{1}{2} \rho v_{\text{rel}}^2$) tracking and Max-Q identification.
  - Aerodynamic restoring moment computed from instantaneous Center of Pressure (CoP) relative to Center of Mass (CoM).

### ⚡ Dual-Mode Time Acceleration Engine
- **Physics Warp (1x, 2x, 4x)**: Sub-stepped 4th-order Runge-Kutta (RK4) numerical integrator with mid-step mass flow and fuel depletion modeling. Runs whenever thrust is firing or the rocket operates within planetary atmospheres ($h \le 100\text{ km}$).
- **On-Rails Warp (10x, 50x, 100x, 1000x)**: Pure analytical Keplerian orbit propagation for coasting through planetary orbits in vacuum without numerical integration drift.
- **Safety Interlocks**: Automatic warp drops when entering atmospheres, igniting engines, or approaching collision courses.

### 🛠️ Interactive Vehicle Assembly Building (VAB)
- **Modular Part Catalog**: Merlin 1D sea-level engines, Merlin 1D Vacuum engines, booster thrusters, modular propellant tanks (Stage 1, Upper Stage, Medium, Small), pneumatic interstage decouplers, Dragon capsule, and payload fairings.
- **Blueprint Canvas**: Pan, zoom, vertical auto-snapping, parallel engine clustering, and part property inspection.
- **Real-Time Engineering Metrics**:
  - Dynamic Center of Mass (CoM) and Center of Pressure (CoP) markers.
  - Moment of inertia ($I_{zz}$) calculated via the Parallel Axis Theorem.
  - Stage-by-stage and vehicle total Tsiolkovsky $\Delta v = g_0 I_{\text{sp}} \ln(m_0 / m_f)$.
  - Sea-level and vacuum Thrust-to-Weight Ratios (TWR).
- **Preset & Serialization**: Quick-load Falcon 9 two-stage orbital rocket preset, JSON rocket blueprint export and import.

### 🎮 Flight Control & Autopilot (SAS)
- **Nozzle Gimbal Thrust Vectoring**: Real-time steerable engine bells with angular deflection limits and gimbal-induced steering torque.
- **PID Stability Augmentation System (SAS)**: Anti-windup PID attitude controller supporting:
  - `OFF`: Manual attitude and gimbal steering.
  - `HOLD_ATTITUDE`: Locks and maintains vehicle attitude heading.
  - `PROGRADE`: Automatically tracks orbital velocity vector for ascent and burn optimization.
  - `RETROGRADE`: Automatically tracks counter-velocity vector for de-orbit burns.

### 📊 Telemetry, Visuals & Analytics
- **64-bit Anti-Jitter Camera**: World-to-screen coordinate transforms executed in 64-bit floating point precision before pixel rasterization, eliminating float32 truncation jitter at planetary distances. Supports **Horizon-Up** and **Inertial** reference frames.
- **Procedural Vector Graphics**: Multi-layered supersonic exhaust plumes, dynamic engine gimbal pivoting, atmospheric sky gradient transitions (day/night/space), curved terrain rendering, and Keplerian conic orbit overlays.
- **Post-Flight Mission Modal**: Mission telemetry ring buffer with live interactive strip charts (Altitude, Airspeed, Dynamic Pressure, G-force, Stage Mass) and one-click CSV export.

---

## 📐 Mathematical & Physics Foundations

### 1. State Vector & Equations of Motion
The vehicle state vector $\mathbf{S} \in \mathbb{R}^6$ in planar inertial space is defined as:
$$\mathbf{S} = [x,\; y,\; v_x,\; v_y,\; \theta,\; \omega]^T$$

The derivative $\dot{\mathbf{S}}$ is governed by:
$$\dot{\mathbf{S}} = [v_x,\; v_y,\; a_x,\; a_y,\; \omega,\; \alpha]^T$$

where:
$$\mathbf{a} = \frac{\mathbf{F}_{\text{thrust}} + \mathbf{F}_{\text{aero}}}{m(t)} + \mathbf{g}(\mathbf{r}), \quad \alpha = \frac{\tau_{\text{thrust}} + \tau_{\text{aero}}}{I_{zz}}$$

### 2. Thrust & Propellant Depletion
Engines adjust thrust according to ambient barometric pressure $p_{\text{amb}}$:
$$F(p_{\text{amb}}) = F_{\text{vac}} - \left(\frac{p_{\text{amb}}}{p_0}\right) (F_{\text{vac}} - F_{\text{sl}})$$
$$\dot{m} = \frac{F_{\text{vac}}}{g_0 I_{\text{sp,vac}}}$$

### 3. Tsiolkovsky Rocket Equation
For each individual stage $i$ with initial mass $m_{0,i}$ (including all upper stages/payload) and burnout mass $m_{f,i}$:
$$\Delta v_i = g_0 \, I_{\text{sp,vac}}^{(i)} \ln\left(\frac{m_{0,i}}{m_{f,i}}\right)$$

### 4. Keplerian Orbital Decomposition
From inertial position $\mathbf{r}$ and velocity $\mathbf{v}$ with planetary gravitational parameter $\mu = G M$:
$$\mathbf{h} = \mathbf{r} \times \mathbf{v}, \quad \varepsilon = \frac{|\mathbf{v}|^2}{2} - \frac{\mu}{|\mathbf{r}|}, \quad a = -\frac{\mu}{2 \varepsilon}$$
$$\mathbf{e} = \frac{1}{\mu}\left((|\mathbf{v}|^2 - \frac{\mu}{|\mathbf{r}|})\mathbf{r} - (\mathbf{r} \cdot \mathbf{v})\mathbf{v}\right)$$
$$r_a = a(1 + |\mathbf{e}|), \quad r_p = a(1 - |\mathbf{e}|)$$

---

## 📁 Repository Structure

```text
Rocket Simulation/
├── core/                       # Core numerical algorithms and primitives
│   ├── constants.py            # CODATA 2018 physical & astronomical constants
│   ├── vector2d.py             # 64-bit precision immutable 2D vector suite
│   └── integrator.py           # RK4 and Velocity-Verlet differential solvers
├── physics/                    # Aerodynamics, orbital mechanics, and atmosphere
│   ├── atmosphere.py           # Exponential barometric & lapse rate models
│   ├── aerodynamics.py         # Angle-of-attack drag, lift, and restoring torque
│   ├── celestial.py            # Earth and Moon planetary bodies & gravity fields
│   └── orbit.py                # Keplerian orbital elements & universal propagation
├── models/                     # Vehicle component models and staging
│   ├── parts.py                # Engine, FuelTank, Decoupler, and Payload classes
│   ├── rocket.py               # Multi-stage rocket, CoM, CoP, and moment of inertia
│   └── presets.py              # Falcon 9 launch vehicle two-stage configuration
├── simulation/                 # Simulation loop orchestrator and flight dynamics
│   ├── autopilot.py            # PID controller and SAS attitude control modes
│   ├── flight_dynamics.py      # 3-DOF coupled planar flight dynamics solver
│   ├── simulation.py           # Dual-mode (Physics / On-Rails) time warp coordinator
│   └── telemetry.py            # High-rate mission logging, downsampling & CSV export
├── ui/                         # User interface, rendering, and views
│   ├── app.py                  # Application state machine (VAB <-> Flight)
│   ├── camera.py               # 64-bit anti-jitter camera (Horizon-Up & Inertial)
│   ├── flight_view.py          # Real-time flight cockpit, HUD, map, and modal
│   ├── renderer.py             # Pure Pygame procedural renderer (rockets, plumes, orbits)
│   ├── vab_view.py             # Blueprint editor, parts catalog, staging & TWR readouts
│   └── widgets.py              # Aerospace dark-theme buttons, sliders, panels, plots
├── tests/                      # Comprehensive automated test suite
│   ├── test_atmosphere.py      # Atmosphere profile, pressure, and lapse rate tests
│   ├── test_core.py            # Vector2D math and RK4 integrator precision tests
│   ├── test_flight.py          # Flight dynamics, liftoff, and thrust integration tests
│   ├── test_orbit.py           # Keplerian decomposition & orbital propagation tests
│   ├── test_rocket.py          # CoM, CoP, mass depletion, and staging tests
│   ├── test_simulation.py      # Time warp switching and safety interlock tests
│   └── test_ui.py              # Camera projections, widgets, and view lifecycle tests
├── main.py                     # Main application entrypoint & CLI argument parser
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **pip** and **virtualenv** (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/rocket-simulation.git
   cd rocket-simulation
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

---

## 💻 Running the Application

### Launch VAB Blueprint Editor (Default)
Start in the Vehicle Assembly Building to customize, inspect, or build your rocket:
```bash
python main.py
```

### Launch Directly onto the Pad
Skip the editor and initiate on the launchpad with the default Falcon 9 preset:
```bash
python main.py --launch
```

### Custom Resolution & Fullscreen
```bash
# Custom window resolution
python main.py --width 1600 --height 900

# Fullscreen mode
python main.py --fullscreen
```

---

## 🎮 Controls & Keybindings

### 🚀 Flight View & Orbital Map

| Key / Input | Action |
|:---|:---|
| <kbd>Space</kbd> | **Trigger Stage** (Ignite active engines / Decouple stage) |
| <kbd>Z</kbd> | **Max Throttle** (100%) |
| <kbd>X</kbd> | **Cut Throttle** (0%) |
| <kbd>Shift</kbd> / <kbd>↑</kbd> | **Throttle Up** (+5%) |
| <kbd>Ctrl</kbd> / <kbd>↓</kbd> | **Throttle Down** (-5%) |
| <kbd>A</kbd> / <kbd>←</kbd> | **Steer CCW** (Pitch/Yaw Left) |
| <kbd>D</kbd> / <kbd>→</kbd> | **Steer CW** (Pitch/Yaw Right) |
| <kbd>M</kbd> | **Toggle Orbital Map View** / Flight View |
| <kbd>C</kbd> | **Toggle Camera Mode** (Horizon-Up $\leftrightarrow$ Inertial) |
| <kbd>.</kbd> (Period) | **Increase Time Warp** (1x $\to$ 2x $\to$ 4x $\to$ 10x $\to$ 1000x) |
| <kbd>,</kbd> (Comma) | **Decrease Time Warp** |
| <kbd>R</kbd> | **Revert Flight** back to launchpad |
| <kbd>V</kbd> or <kbd>Esc</kbd> | **Open / Close Post-Flight Telemetry Modal** |
| <kbd>Scroll Wheel</kbd> | **Zoom In / Zoom Out** |
| <kbd>Right Click Drag</kbd> | **Pan Camera** across world space |

### 🛠️ Vehicle Assembly Building (VAB)

| Key / Input | Action |
|:---|:---|
| <kbd>Space</kbd> or <kbd>Enter</kbd> | **Launch Rocket** to flight view |
| <kbd>Delete</kbd> | **Delete Selected Part** |
| <kbd>C</kbd> | **Recenter Blueprint Canvas** |
| <kbd>Left Click</kbd> | **Select Part** for property inspection |
| <kbd>Catalog Buttons</kbd> | **Spawn Parts** (Engines, Tanks, Decouplers, Payloads) |
| <kbd>Scroll Wheel</kbd> | **Zoom In / Zoom Out Canvas** |
| <kbd>Middle / Right Click</kbd> | **Pan Canvas** |
| **Top Menu** | Load Falcon 9 Preset, Clear Stack, Save/Load Blueprint JSON |

---

## 🧪 Testing

The codebase includes **99 automated unit and regression tests** verifying all mathematical, physical, and UI systems:

```bash
pytest
```

Output:
```text
============================= test session starts ==============================
collected 99 items

tests/test_atmosphere.py ...........                                     [ 11%]
tests/test_core.py ...........                                           [ 22%]
tests/test_flight.py ....................                                [ 42%]
tests/test_orbit.py ..............                                       [ 56%]
tests/test_rocket.py ..............                                      [ 70%]
tests/test_simulation.py ..............                                  [ 84%]
tests/test_ui.py ...............                                         [100%]

======================== 99 passed in 8.30s ====================================
```

---

## 📈 Post-Flight Telemetry & Analysis

During flight, high-rate simulation snapshots are captured by the `TelemetryLogger`. Opening the Telemetry Modal (<kbd>V</kbd> or <kbd>Esc</kbd>) displays:
- **Flight Summary**: Peak apogee, peak speed, maximum aerodynamic dynamic pressure (**Max-Q**), maximum G-force experienced, total $\Delta v$ expended, and flight status.
- **Interactive Multi-Channel Graphs**:
  - Altitude vs. Time ($h(t)$)
  - True Airspeed vs. Time ($v(t)$)
  - Dynamic Pressure vs. Time ($q(t)$)
  - Vehicle Gross Mass vs. Time ($m(t)$)
- **Data Export**: Export complete telemetry history to CSV for post-processing in NumPy, Pandas, or MATLAB.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

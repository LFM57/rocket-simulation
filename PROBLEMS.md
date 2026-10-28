# UI, Camera & Flight Control Issues (`PROBLEMS.md`)

## 1. UI Layout & Overlapping Elements ("Too Packed / Unusable")

### 1.1. Flight View HUD Overlaps (`ui/flight_view.py`)
- **`Menu [V]` Overlaps SAS Mode Buttons**: `b_end` (`Menu [V]`) is fixed at `y = 12`, while the SAS selector buttons (`OFF`, `HOLD`, `PRO`, `RET`) are positioned relative to vertical center (`slider_y - 84 = height // 2 - 224`). On any window height below ~600 px, the SAS row collides directly with the `Menu [V]` button.
- **Telemetry Header Text Collision**: In `_render_telemetry_panel()`, `str(snap.state)` formats the `(str, Enum)` as `"VehicleState.ON_PAD"` instead of `"ON_PAD"` (`snap.state.value`), causing it to overlap the `"TELEMETRY"` title on the 216 px wide panel.
- **Propellant ProgressBar Text Overflow**: The label `"Propellant (410.00 t): 100.0%"` is wider than the 176 px `ProgressBar` container and bleeds outside its borders.
- **Uncentered Top Warp Bar**: The 7 warp buttons (`1x`–`1000x`) start at `width // 2 - 240` (total width 336 px), shifting them toward the left where they crowd the SAS / Menu buttons on smaller resolutions.

### 1.2. VAB Blueprint Editor Overlaps (`ui/vab_view.py`)
- **Top Toolbar Collisions**:
  - `"Rocket: Falcon 9"` is drawn at `top_bar_rect.left + 16`, which is directly underneath the `Falcon 9 Preset` button at `top_bar_rect.left + 12`.
  - The 4 toolbar buttons require 422 px of horizontal space and get clipped by the right Engineering Panel (`Load JSON` overflows into the right sidebar).
- **Part Inspector Collides with `LAUNCH` Button**: In `_render_engineering_panel()`, the `PART INSPECTOR` section begins at `py = max(py, 420)` and renders 6 lines downward, overlapping the green `LAUNCH` button anchored at `sidebar_right_rect.bottom - 54`.
- **Unreadable & Stacked Canvas Part Labels**:
  - Dark part labels (`(20, 25, 35)`) on narrow tanks overflow horizontally onto the dark blueprint background (`(16, 22, 34)`), making the text nearly invisible.
  - `_render_rocket_stack()` does not laterally offset multi-engine clusters, drawing all 9 Stage 1 `Merlin 1D` engines and their labels on the exact same pixels.

---

## 2. Camera Tracking & Scrolling Issues

### 2.1. Rocket Off-Center / Off-Screen Due to Camera Lerp Velocity Lag (`ui/camera.py`)
- **Root Cause**: `Camera.update()` uses a first-order exponential position filter (`p_factor = 1.0 - exp(-position_smoothing * dt)` with `position_smoothing = 12.0`).
- **Effect**: For a vehicle moving at speed $v$, exponential position smoothing introduces a steady-state spatial lag of $\Delta r = v / k$.
  - Even while sitting stationary on the launchpad, Earth's surface co-rotation ($v \approx 464.6\text{ m/s}$) causes the camera to lag **~38.7 meters behind the rocket**. At the default zoom (`scale = 12.0 px/m`), the rocket is pushed **~465 pixels to the left of screen center** (underneath the Throttle HUD).
  - Once the rocket accelerates in flight ($v > 1000\text{ m/s}$) or the user zooms in, the lag pushes the rocket completely off-screen.

### 2.2. Scrolling & Zoom Input Issues (`ui/flight_view.py`, `ui/vab_view.py`)
- **Missing `pygame.MOUSEWHEEL` Support**: Both `FlightView.handle_event()` and `VABView.handle_event()` only check legacy `MOUSEBUTTONDOWN` buttons `4` and `5` and ignore Pygame 2's `pygame.MOUSEWHEEL` event (`event.y`), breaking trackpad and mouse-wheel scrolling on many platforms.
- **No Scrolling in VAB Sidebars**: Neither the left `PART CATALOG` nor the right `ENGINEERING READOUT` / `STAGING STACK` supports vertical scrolling when content exceeds the window height.
- **No Camera Panning or Keyboard Zoom in Flight/Map View**: `FlightView.update()` overwrites `camera.set_target(self.vehicle.position)` every frame with no support for manual camera panning or keyboard zoom shortcuts (`+`/`-`).
- **VAB Canvas Not Recentered on Resize**: `VABView.recenter_canvas()` is only called once in `__init__` (using initial 1280x720 bounds) and is never updated in `update_layout()`, leaving the 70 m Falcon 9 clipped at the top and bottom of the viewport.

---

## 3. Rocket Steering & Uncontrollable Spin Issues

The difficulty controlling the rocket is not just due to realistic physics; several modeling and control defects actively cause the vehicle to spin out of control:

### 3.1. Falcon 9 Preset is Aerodynamically Unstable at Liftoff (`models/rocket.py`, `models/presets.py`)
- At launch, the Falcon 9 preset has its Center of Pressure (`CoP = 32.18 m`) located **3.31 m ahead** of its Center of Mass (`CoM = 28.87 m`).
- Because $y_{\text{CoP}} > y_{\text{CoM}}$, any slight steering angle of attack ($\alpha \neq 0$) in the lower atmosphere generates a large destabilizing aerodynamic torque that flips the nose away from the velocity vector and overwhelms the $\pm 5^\circ$ engine gimbal authority.

### 3.2. Zero Aerodynamic Pitch Damping (`physics/aerodynamics.py`)
- `calculate_aero_forces()` computes torque solely from static translational angle of attack and includes no rotational damping term ($\tau_{\text{damp}} \propto -\omega$). Once the rocket begins rotating, the atmosphere provides no resistance to angular velocity.

### 3.3. SAS Target Attitude Windup & Under-Damped PID Gains (`ui/flight_view.py`, `simulation/autopilot.py`)
- **Blind Target Windup**: Holding `A`/`D` (`LEFT`/`RIGHT`) in `HOLD_ATTITUDE` mode increments `autopilot.target_attitude` at `0.4 rad/s` ($23^\circ/\text{s}$) faster than the heavy rocket can physically rotate, with no HUD indicator showing the commanded target angle.
- **Insufficient Rate Damping (`kp = 2.0, kd = 2.0`)**: Due to the Falcon 9's high rotational inertia ($I_{zz} \approx 1.15 \times 10^8\text{ kg}\cdot\text{m}^2$) and bounded gimbal torque ($\alpha_{\max} \approx 0.16\text{ rad/s}^2$), a $kd/kp$ ratio of $1.0$ starts braking far too late, causing severe overshoot and continuous spin oscillation after a turn.

### 3.4. Manual Gimbal Never Resets & Zero Control Authority at 0% Throttle (`ui/flight_view.py`)
- **Sticky Gimbal in `SASMode.OFF`**: In manual mode (`SASMode.OFF`), pressing `A`/`D` increments `eng.current_gimbal`, but releasing the key only sets `steer_rate = 0.0` without resetting `eng.current_gimbal` back to `0.0`. The nozzles remain deflected and spin the rocket indefinitely.
- **No Attitude Control Without Thrust**: Steering relies exclusively on engine thrust vectoring (no RCS thrusters, reaction wheels, or aerodynamic control surfaces), so the player cannot stop a spin whenever throttle is low or fuel is depleted.

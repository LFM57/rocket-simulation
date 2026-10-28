# Codebase Mathematical, Physical & Software Audit (`PROBLEMS.md`)

This document catalogs the critical bugs, mathematical/physical inaccuracies, and edge-case defects identified in the 2D rocket simulation codebase.

## Status Summary

| # | Bug Title | Status | Commit |
|---|-----------|--------|--------|
| 1.1 | Zero Thrust Applied in RK4 Integration on Propellant Burnout Step | **DONE** | `4dfa9f1` |
| 1.2 | Rectilinear / Radial Trajectories Misclassified as Parabolic Escape Orbits | **DONE** | `347461c` |
| 1.3 | `propagate_kepler` Parabolic Singularity & Teleportation Bug | **DONE** | `eb5a4a3` |
| 1.4 | On-Rails Time Warp Tunnels Through Atmosphere and Surface | **DONE** | `c32d864` |
| 1.5 | Empty Stage Liftoff Bug on Launchpad | **DONE** | `210df22` |
| 2.1 | First-Order $O(\Delta t)$ Mass Discretization Bias in RK4 Integration | **DONE** | `1cec00f` |
| 2.2 | Aerodynamic Cross-Flow Force Direction, Lift Slope, and Area Overcounting | **DONE** | `f4a8aaa` |
| 2.3 | Spurious `CollisionWarning` Emitted While `ON_PAD` or `LANDED` | **DONE** | `ecdc141` |
| 2.4 | Surface Co-Rotation Velocity Bug on Airless Rotating Bodies | **DONE** | `17d2bd6` |
| 3.1 | `TelemetryLogger` Overestimates Expended $\Delta v$ on Sub-Stepped Burnout and Staging | **DONE** | `d65685e` |
| 3.2 | `TelemetryLogger.get_summary()` Ignores Aerodynamic G-Forces | **DONE** | Solved |
| 3.3 | `Simulation.run()` Misplaced Docstring and `wall_dt <= 0` Infinite Loop | **TODO** | Open |
| 3.4 | Unused `Decoupler.separation_impulse` & Docstring Sign Mismatch | **TODO** | Open |
| 3.5 | Missing `Part` import for static type checkers in `simulation/flight_dynamics.py` | **TODO** | Open |
| 3.6 | Partial-burn liftoff check on `ON_PAD` | **TODO** | Open |

---

## 1. Critical Bugs (Physics & Numerical Failures)

### 1.1. [DONE] Zero Thrust Applied in RK4 Integration on the Propellant Burnout Step
- **Status**: Solved (commit `4dfa9f1`)
- **Severity**: Critical
- **Locations**:
  - `simulation/flight_dynamics.py` (lines 407–410, 515–538)
  - `models/rocket.py` (lines 307–320)
  - `models/parts.py` (lines 294–295)
- **Description**:
  In `FlightVehicle.step()`, propellant is drained for the full time step `dt_val` **before** calling `rk4_step`:
  ```python
  actual_thrust, _, _ = self.rocket.burn_fuel(dt_val, p_amb)
  self._step_thrust_scale = (actual_thrust / nom_thrust) if nom_thrust > 0.0 else 0.0
  self._firing_engines = firing_engines if self._step_thrust_scale > 0.0 else []
  self._last_step_thrust = actual_thrust
  ```
  When the remaining propellant in the active stage is less than `prop_needed` for `dt_val` (`0 < avail_prop < prop_needed`), `Rocket.burn_fuel()` drains the remaining fuel, returns the fractional `actual_thrust > 0`, and **immediately shuts down all engines**:
  ```python
  # models/rocket.py:318-320
  for eng in ignited_engines:
      eng.shutdown()  # Sets eng.is_ignited = False
  ```
  Next, `rk4_step` calls `FlightVehicle.compute_derivatives()`, which loops over `self._firing_engines` and calls `eng.get_thrust_and_mass_flow(p_amb)`:
  ```python
  # simulation/flight_dynamics.py:407-409
  for eng in engines_to_eval:
      t_val, _ = eng.get_thrust_and_mass_flow(p_amb)
      t_val *= self._step_thrust_scale
  ```
  However, `Engine.get_thrust_and_mass_flow()` checks `self.is_ignited`:
  ```python
  # models/parts.py:294-295
  if not self.is_ignited or self.throttle <= 0.0:
      return 0.0, 0.0
  ```
- **Impact**:
  Because `burn_fuel()` already set `eng.is_ignited = False`, `get_thrust_and_mass_flow()` returns `(0.0, 0.0)` for every engine in `self._firing_engines`. The entire `_step_thrust_scale` mechanism is rendered dead code: on the burnout step of any stage, the remaining propellant disappears from the tank and `TelemetrySnapshot` (as well as `TelemetryLogger._delta_v_expended`) records `actual_thrust > 0`, **but the RK4 physics integration applies zero thrust force and zero gimbal torque**.
- **Fix**:
  Either pass `ignore_ignited=True` (or evaluate nominal thrust before `burn_fuel()`) when iterating over `self._firing_engines`, or defer `eng.shutdown()` until after `rk4_step` completes.

---

### 1.2. [DONE] Rectilinear / Radial Trajectories Misclassified as Parabolic Escape Orbits
- **Status**: Solved (commit `347461c`)
- **Severity**: Critical
- **Location**: `physics/orbit.py` (lines 201 and 395)
- **Mathematical Root Cause**:
  In two-body Keplerian mechanics, eccentricity $e$, specific orbital energy $\epsilon = \frac{v^2}{2} - \frac{\mu}{r}$, and specific angular momentum $h = \vec{r} \times \vec{v}$ satisfy:
  $$e = \sqrt{1 + \frac{2\epsilon h^2}{\mu^2}}$$
  There are **two completely distinct physical regimes** where $e = 1$:
  1. **Parabolic escape trajectory** ($\epsilon = 0, h \neq 0$): $v = v_{\text{esc}} = \sqrt{2\mu/r}$, $a = \infty$, $r_a = \infty$, $r_p = \frac{h^2}{2\mu}$.
  2. **Rectilinear (radial) trajectory** ($h = 0$, i.e. $\vec{r} \parallel \vec{v}$ or $\vec{v} = \vec{0}$): Evaluating the Laplace–Runge–Lenz vector when $\vec{v} = v_r \hat{r}$ gives:
     $$\vec{e} = \frac{1}{\mu}\left[\left(v_r^2 - \frac{\mu}{r}\right)\vec{r} - (r v_r)(v_r \hat{r})\right] = -\frac{\vec{r}}{r} = -\hat{r} \implies e = \|\vec{e}\| = 1.0 \quad \forall \epsilon$$
     When $\epsilon < 0$ and $h = 0$ (e.g., an object dropped from rest or a rocket ascending vertically on a non-rotating body like the Moon), the trajectory is a **bound rectilinear ellipse** with finite semi-major axis $a = -\frac{\mu}{2\epsilon} > 0$, finite apoapsis radius $r_a = a(1+e) = 2a = -\frac{\mu}{\epsilon}$, and $r_p = a(1-e) = 0$.
- **What happens in the code**:
  Line 201 checks:
  ```python
  if math.isclose(energy, 0.0, abs_tol=1e-12) or math.isclose(e, 1.0, abs_tol=1e-7):
  ```
  Because of `or math.isclose(e, 1.0, abs_tol=1e-7)`, any vehicle with zero velocity or purely vertical velocity ($h = 0$) is misclassified as a **parabolic escape trajectory** with `semi_major_axis = float("inf")` and `apoapsis_altitude = float("inf")`.
  *(Note: `math.isclose(energy, 0.0, abs_tol=1e-12)` failed at $v = v_{\text{esc}}$ because $\frac{\mu}{r} \approx 6 \times 10^7\text{ J/kg}$, so 1 ULP in IEEE 754 `float64` is $\sim 1.3 \times 10^{-8}\text{ J/kg} \gg 10^{-12}$.)*
- **Fix**:
  Classify conic sections strictly by dimensionless specific energy $\frac{|\epsilon|}{\mu / r} < 10^{-7}$ (i.e., `math.isclose(0.5 * v_sq, mu / r, rel_tol=1e-7)`), and allow bound rectilinear trajectories ($\epsilon < 0, e = 1$) to use the standard elliptic branch ($a = -\frac{\mu}{2\epsilon}, r_p = a(1-e) = 0, r_a = a(1+e) = 2a$).

---

### 1.3. [DONE] `propagate_kepler` Parabolic Singularity & Teleportation Bug
- **Status**: Solved (commit `eb5a4a3`)
- **Severity**: Critical
- **Location**: `physics/orbit.py` (lines 395–475)
- **Description**:
  In `propagate_kepler()`, lines 395–401 attempt to avoid the parabolic singularity ($e = 1, \epsilon = 0$) by biasing eccentricity:
  ```python
  if math.isclose(energy0, 0.0, abs_tol=1e-12) or math.isclose(e, 1.0, abs_tol=1e-7):
      if energy0 < 0.0:
          e = 1.0 - 1e-8
      else:
          e = 1.0 + 1e-8
  ```
  However, `energy0` is **not** updated to match the biased eccentricity, and lines 404 and 444 immediately compute:
  ```python
  semi_major_axis = float(np.float64(-mu / (2.0 * energy0)))
  ```
  1. If `energy0 == 0.0` exactly, line 444 raises **`ZeroDivisionError: float division by zero`**.
  2. If `v = math.sqrt(2.0 * mu / r0)` has 1-ULP floating-point rounding (`energy0 ~ 1e-8 J/kg`), then `abs_a = mu / (2 * energy0) ~ 2e22 m`. Combining `abs_a ~ 2e22 m` with `e = 1.0 + 1e-8` gives a semi-latus rectum `p = abs_a * (e^2 - 1) ~ 4e14 m` (off by a factor of $3 \times 10^7$) and teleports the spacecraft to `r_target ~ 2e14 m` ($\sim 1300\text{ AU}$) in a single step.
  3. Additionally, for rectilinear ellipses ($h = 0, e \to 1^-$), line 440 computes velocity via $\sqrt{\mu / p}$ where $p = a(1-e^2) \to 0$, which blows up instead of using the non-singular eccentric anomaly velocity formula $\vec{v} = \frac{\sqrt{\mu a}}{r}(-\sin E\,\hat{p} + \sqrt{1-e^2}\cos E\,\hat{q})$.
- **Fix**:
  When perturbing a near-parabolic orbit with $h \neq 0$ to $e = 1 \pm 10^{-8}$, compute `semi_major_axis` consistently from the conserved semi-latus rectum $p = h^2 / \mu$ via $a = \frac{p}{1 - e^2}$ (or solve Barker's equation directly).

---

### 1.4. [DONE] On-Rails Time Warp Tunnels Through Atmosphere and Planet Surface
- **Status**: Solved (commit `c32d864`)
- **Severity**: Critical
- **Location**: `simulation/simulation.py` (lines 142, 206–260)
- **Description**:
  There are three separate bugs in the `WarpMode.ON_RAILS` propagation logic in `Simulation.step()`:
  1. **Hyperbolic trajectories bypass entry interception (line 206)**:
     ```python
     if orbit.periapsis_radius < r_atm and orbit.eccentricity < 1.0:
     ```
     If a spacecraft is on a hyperbolic return or flyby trajectory ($e \ge 1.0$) whose periapsis intersects the atmosphere or planet surface ($r_p < r_{\text{atm}}$), the entry time calculation is completely skipped. At high time warp, `propagate_kepler` steps right through the atmosphere and planet in a single macro-step without dropping to physics warp or detecting impact.
  2. **Boundary tolerance wrap-around bug (lines 142 & 225)**:
     `set_time_warp()` (line 142) allows engaging `ON_RAILS` warp when `alt >= self.cutoff_altitude - 1e-4` (i.e. $r \in [r_{\text{atm}} - 10^{-4}, r_{\text{atm}})$). If an inbound spacecraft is in this $0.1\text{ mm}$ tolerance band (or $1\text{ ULP}$ below $r_{\text{atm}}$ due to float rounding), its current true anomaly $\nu_{\text{curr}} \in (\pi, 2\pi)$ has **already passed** $\nu_{\text{entry}}$, so $M_{\text{curr}} > M_{\text{entry}}$. Line 225 then computes:
     ```python
     dt_to_entry = (m_entry - m_curr) % (2.0 * math.pi) / n
     ```
     Because `m_entry - m_curr` is a tiny negative number $-\delta$, modulo $2\pi$ wraps it to $2\pi - \delta$, yielding `dt_to_entry` equal to nearly a **full orbital period** ($\sim 5400\text{ s}$). Since `dt_to_entry > dt_sim`, `step_dt` is not clamped, and the spacecraft propagates on rails straight through the atmosphere/planet for the full `dt_sim`.
  3. **No surface collision state transition on airless bodies**:
     On airless bodies (`self.body.atmosphere is None`, e.g. `MOON`), `cutoff_altitude` is `0.0` (`r_atm = body.radius`). Even when `ON_RAILS` drops to `PHYSICS` at $r = R_{\text{body}}$, `ON_RAILS` never evaluates touchdown/crash conditions (`VehicleState.DESTROYED` / `LANDED`).

---

### 1.5. [DONE] Empty Stage Liftoff Bug on Launchpad
- **Status**: Solved (commit `210df22`)
- **Severity**: High
- **Location**: `simulation/flight_dynamics.py` (lines 486–504)
- **Description**:
  In `FlightVehicle.step()` under `if self.state == VehicleState.ON_PAD:`, the liftoff check computes `f_thrust_vec` by summing `eng.get_thrust_and_mass_flow(p_sl)` over all ignited engines in `self.rocket.active_stage` **without checking `self.rocket.active_stage.propellant_mass() > 0.0`**.
  Because `Engine` has no knowledge of `FuelTank` propellant levels, an ignited engine on an empty stage returns full nominal thrust.
- **Impact**:
  `radial_thrust > weight` evaluates to `True`, transitioning an empty rocket from `ON_PAD` to `IN_FLIGHT`. In the `IN_FLIGHT` block on the very same step, `burn_fuel()` returns `0.0` thrust, the rocket free-falls into the ground (`alt_new <= 0.0`), and immediately transitions to `VehicleState.LANDED` (or `DESTROYED`).
- **Fix**:
  Guard the `ON_PAD` thrust loop with `if self.rocket.active_stage.propellant_mass() > 0.0:`, matching `compute_derivatives()` and `get_telemetry_snapshot()`.

---

## 2. Mathematical & Physical Modeling Issues

### 2.1. [DONE] First-Order $O(\Delta t)$ Mass Discretization Bias in RK4 Integration
- **Status**: Solved (commit `1cec00f`)
- **Severity**: Medium
- **Location**: `simulation/flight_dynamics.py` (lines 522–534)
- **Mathematics**:
  `FlightVehicle.step()` drains the entire step's propellant via `self.rocket.burn_fuel(dt_val, p_amb)` *before* caching `m = self.rocket.total_mass()`, `i_zz`, and `y_com` for `rk4_step`. Consequently, all 4 stages of the RK4 integrator evaluate linear and angular accelerations using the **end-of-step mass** $m(t + \Delta t)$ instead of the time-varying mass $m(t + c_i \Delta t)$ (or even the midpoint mass $m_{\text{mid}} = \frac{1}{2}[m(t) + m(t + \Delta t)]$).
  Because $m(t + \Delta t) < m(t)$, this introduces a systematic $O(\Delta t)$ positive bias in acceleration and $\Delta v$, reducing the effective order of the integrator from $O(\Delta t^4)$ to $O(\Delta t)$ during powered flight.

---

### 2.2. [DONE] Aerodynamic Cross-Flow Force Direction, `lift_coefficient` Slope, and Area Overcounting
- **Status**: Solved (commit `f4a8aaa`)
- **Severity**: Medium
- **Locations**:
  - `physics/aerodynamics.py` (lines 119–137, 296–315)
  - `models/presets.py` (lines 36–89)
  - `models/rocket.py` (lines 213–239)
- **Issues**:
  1. **Cross-flow force applied along $-\hat{v}_{\text{rel}}$ instead of body normal (`physics/aerodynamics.py:296-308`)**:
     ```python
     f_drag_mag = q * (cd_axial * reference_area * cos_a_sq + cd_transverse * a_trans * sin_a_sq)
     f_drag = -v_rel_hat * float(np.float64(f_drag_mag))
     ```
     With `cl_alpha = 0.0` (the default), the entire lateral planform cross-flow term $q C_{d,\text{trans}} A_{\text{trans}} \sin^2\alpha$ is applied parallel to $-\hat{v}_{\text{rel}}$ (pure drag) with zero force component perpendicular to $\vec{v}_{\text{rel}}$. For a slender rocket where $A_{\text{trans}} \approx 262\text{ m}^2$ and $A_{\text{ref}} \approx 10.75\text{ m}^2$, pitching to $\alpha = 90^\circ$ increases drag along $\vec{v}_{\text{rel}}$ by $\sim 100\times$, while generating zero normal lift force unless `cl_alpha` is manually enabled.
  2. **Factor of 2 in `lift_coefficient` slope (`physics/aerodynamics.py:127`)**:
     `C_L(alpha) = cl_alpha * sin(2 * alpha)` has a small-angle slope of:
     $$\left.\frac{dC_L}{d\alpha}\right|_{\alpha=0} = 2 \, c_{l,\alpha}$$
     With the default `cl_alpha = 2 * math.pi`, the small-angle lift slope is $4\pi\text{ rad}^{-1}$ (twice the theoretical thin-airfoil slope of $2\pi\text{ rad}^{-1}$; the standard formula over $[-\pi, \pi]$ is $\frac{1}{2} c_{l,\alpha} \sin(2\alpha)$).
  3. **Multi-engine cluster and enclosed upper-stage engine area overcounting (`models/presets.py:36-89`, `models/rocket.py:213-239`)**:
     `Rocket.center_of_pressure()` and `FlightVehicle._get_transverse_area()` sum `p.projected_area = 2 * r * L` across every part in all non-separated stages. In `create_falcon_9()`:
     - All 9 Merlin 1D engines sit side-by-side at `position_offset = 1.35 m`, yet their areas are summed as if stacked in series ($9 \times 3.24 = 29.16\text{ m}^2$, equivalent to adding $7.9\text{ m}$ of full-diameter booster hull at the tail).
     - The Stage 2 `mvac` engine (`7.2 m^2`) is enclosed inside the interstage prior to staging, yet its area is included in Stage 1 aerodynamics.
     - Even with the 9-engine tail bias, at liftoff $y_{\text{CoP}} = 30.32\text{ m} > y_{\text{CoM}} = 28.87\text{ m}$ (aerodynamically unstable until $\sim 15\text{ s}$ into the burn when fuel depletion shifts $y_{\text{CoM}}$ forward of $y_{\text{CoP}}$).

---

### 2.3. [DONE] Spurious `CollisionWarning` Emitted While `ON_PAD` or `LANDED`
- **Status**: Solved (commit `ecdc141`)
- **Severity**: Medium
- **Locations**:
  - `physics/orbit.py` (line 177)
  - `physics/celestial.py` (line 235)
  - `simulation/flight_dynamics.py` (lines 510, 583, 591)
- **Description**:
  `FlightVehicle.get_telemetry_snapshot()` calls `calculate_orbit(self.position, self.velocity, self.body)` on every simulation step, which unconditionally calls `body.is_colliding(pos)` (`dist < self.radius`).
  When the vehicle is `ON_PAD` or `LANDED`, its position is updated via:
  ```python
  self.position = self.position.rotate(d_phi).normalized() * self.body.radius
  ```
  Due to standard IEEE 754 `float64` rounding in `math.hypot(x, y)`, `(self.position - body.position).magnitude()` is frequently $1\text{ ULP}$ ($\sim 9 \times 10^{-10}\text{ m}$) strictly less than `self.body.radius`. This triggers false `CollisionWarning` warnings while the rocket is stationary on the launchpad or landed.
- **Fix**:
  Use a small numerical tolerance in `is_colliding` (e.g., `dist < self.radius - 1e-6`), or pass `warn=False` when calling `calculate_orbit` from telemetry snapshots.

---

### 2.4. [DONE] Surface Co-Rotation Velocity Bug on Airless Rotating Bodies
- **Status**: Solved (commit `17d2bd6`)
- **Severity**: Medium
- **Locations**:
  - `simulation/flight_dynamics.py` (lines 270–273 vs. lines 511, 557, 584)
  - `physics/celestial.py` (lines 275–285)
- **Description**:
  In `FlightVehicle._clamp_to_surface()`, the surface velocity is computed directly from the body's sidereal rotation rate `omega_p`:
  ```python
  omega_p = self._get_planet_omega()
  self.velocity = Vector2D(-omega_p * self.position.y, omega_p * self.position.x)
  ```
  However, in `FlightVehicle.step()` (lines 511, 557, 584), the surface velocity for `ON_PAD`, `LANDED`, and touchdown impact speed checks is obtained via `self.body.wind_velocity_at(self.position)`.
  Because `CelestialBody.wind_velocity_at()` returns `Vector2D(0.0, 0.0)` whenever `self.atmosphere is None` (line 275 of `physics/celestial.py`), any rotating celestial body without an atmosphere has its surface position rotated at `omega_p` while its surface velocity is set to `(0.0, 0.0)`.
- **Fix**:
  Add a dedicated `CelestialBody.surface_velocity_at(pos)` method that depends only on `rotation_period`, and use it for surface clamping and landing checks.

---

## 3. Telemetry & Simulation Orchestrator Defects

### 3.1. [DONE] `TelemetryLogger` Overestimates Expended $\Delta v$ on Sub-Stepped Burnout and Staging
- **Status**: Solved (commit `d65685e`)
- **Severity**: Low / Medium
- **Location**: `simulation/telemetry.py` (lines 95–103)
- **Description**:
  `TelemetryLogger.record()` integrates expended $\Delta v$ via the trapezoidal rule between consecutive logged snapshots:
  ```python
  dt = snapshot.time - self._prev_snapshot.time
  if dt > 0.0:
      m_mid = 0.5 * (self._prev_snapshot.mass + snapshot.mass)
      t_mid = 0.5 * (self._prev_snapshot.thrust + snapshot.thrust)
      if m_mid > 0.0 and t_mid > 0.0:
          self._delta_v_expended += (t_mid / m_mid) * dt
  ```
  1. When `Simulation.step(wall_dt)` executes multiple physics sub-steps (e.g., `wall_dt = 1.0 s` with `max_physics_substep = 0.05 s` = 20 sub-steps), `logger.record(snap)` is only called once at the end of the 20 sub-steps. If Stage 1 burns out on sub-step 1 (after $0.05\text{ s}$), `prev_snapshot.thrust` is full thrust ($\sim 8.2\text{ MN}$) and `snapshot.thrust` is $0\text{ N}$, so `TelemetryLogger` integrates $0.5 \times 8.2\text{ MN}$ across the full $1.0\text{ s}$ macro-step, overestimating the burnout impulse by $10\times$.
  2. When stage separation occurs between snapshots, `m_mid = 0.5 * (self._prev_snapshot.mass + snapshot.mass)` averages the pre-staging full stack mass with the post-staging upper-stage mass.

---

### 3.2. [DONE] `TelemetryLogger.get_summary()` Ignores Aerodynamic G-Forces
- **Status**: Solved
- **Severity**: Low / Medium
- **Location**: `simulation/telemetry.py` (lines 232–235)
- **Description**:
  ```python
  max_g = max(
      (s.thrust / (s.mass * g0) if s.mass > 0.0 else 0.0 for s in self._snapshots),
      default=0.0,
  )
  ```
  `max_g_force` only measures thrust-to-weight ratio ($\frac{T}{m g_0}$) and completely ignores aerodynamic drag and lift forces. During unpowered atmospheric reentry (`s.thrust == 0.0`), `max_g_force` reports `0.0 g` even if the vehicle experiences $>10 g$ of aerodynamic deceleration.
- **Fix**:
  Record net non-gravitational acceleration magnitude $\|\vec{F}_{\text{thrust}} + \vec{F}_{\text{aero}}\| / (m g_0)$ in `TelemetrySnapshot`.

---

### 3.3. [TODO] `Simulation.run()` Misplaced Docstring and `wall_dt <= 0` Infinite Loop
- **Status**: Open
- **Severity**: Low
- **Location**: `simulation/simulation.py` (lines 269–299)
- **Description**:
  ```python
  def run(
      self,
      duration: float,
      wall_dt: float = 0.1,
      target_pitch: Optional[float] = None,
      throttle: Optional[float] = None,
  ) -> list[TelemetrySnapshot]:
      # Guard against freeze when simulation is paused (time_warp == 0)
      if self.time_warp <= 1e-9:
          snap = self.step(wall_dt, target_pitch=target_pitch, throttle=throttle)
          return [snap]
      """Run the simulation for a specified simulation duration in seconds.
  ```
  1. The `if self.time_warp <= 1e-9:` block is placed **before** the docstring, turning the docstring into an unassigned string literal (`Simulation.run.__doc__` is `None`).
  2. If `wall_dt <= 0.0` is passed while `duration > 0.0` and `time_warp > 0.0`, `self.step(0.0)` does not advance `self.elapsed_time`, causing `while self.elapsed_time < target_time - 1e-9:` to hang in an infinite loop.

---

### 3.4. [TODO] Unused `Decoupler.separation_impulse` & Docstring Sign Mismatch
- **Status**: Open
- **Severity**: Low
- **Locations**:
  - `models/parts.py` (lines 334–341), `models/rocket.py` (lines 393–432)
  - `models/rocket.py` (line 346 vs. line 387)
- **Description**:
  1. `Decoupler` defines `separation_impulse: float = 0.0` (configured to `5000.0 N s` in `create_falcon_9()`), but neither `Rocket.stage()` nor `FlightVehicle.stage()` applies the impulse $\Delta v = J / m$ upon stage separation.
  2. The docstring of `Rocket.thrust_force_and_torque()` (line 346) states `tau_thrust = (y_engine - y_CoM) * (T * sin(delta))`, whereas the implementation (line 387) computes `total_tau += -arm * fx = -(y_eng - y_com) * (T * sin(delta))`.

### 3.5 **Missing `Part` import for static type checkers in `simulation/flight_dynamics.py`**:

Line 251 of `simulation/flight_dynamics.py` uses the type annotation `axial_groups: dict[float, list[Part]] = {}`, but only `Engine` is imported from `models.parts` on line 25 (`from models.parts import Engine`). Because `from __future__ import annotations` is active, Python does not fail at runtime, but `mypy`/`pyright` will flag `Part` as undefined unless changed to `from models.parts import Engine, Part`.

### 3.6 **Partial-burn liftoff check on `ON_PAD` (`simulation/flight_dynamics.py:536-560`)**:
If a rocket is `ON_PAD` with only a tiny trace of propellant left (enough for e.g. $0.01\text{ s}$ of thrust during a $1.0\text{ s}$ step), the `ON_PAD` liftoff check uses unscaled instantaneous thrust (`radial_thrust > weight`), transitions to `IN_FLIGHT`, and then immediately scales the step's thrust down to $1\%$ in `IN_FLIGHT`, causing the vehicle to sink below `altitude <= 0.0` and transition to `LANDED`. Scaling `f_thrust_vec` by `min(1.0, avail_prop / (total_mdot * dt_val))` during the `ON_PAD` check prevents liftoff when propellant cannot sustain $T > W$ across the step.

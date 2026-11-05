# Next practical simulation idea: connected-vehicle platoon under V2V spoofing

## Why this is a better practical demonstration

The current test bench is a clean adversarial-communication experiment: one
vehicle/agent sends false state telemetry, and the comparison shows how each
neighbor-dependent controller reacts with and without W-MSR. It is useful for
isolating the message-filtering mechanism, but its states are abstract and all
normal agents are ultimately trying to converge to the origin.

A more intuitive follow-up is a **connected autonomous-vehicle platoon**. Cars
travel along a road while maintaining prescribed spacing and a common speed.
A compromised vehicle broadcasts false position/speed information to its V2V
neighbors. The simulation shows whether the normal vehicles maintain their
desired spacing when the message stream is falsified.

## State and formation coordinates

For vehicle i, use the idealized longitudinal model

\[
\begin{aligned}
p_i(k+1)&=p_i(k)+h v_i(k),\\
v_i(k+1)&=d_i v_i(k)+b_i u_i(k),
\end{aligned}
\]

or

\[
A_i=\begin{bmatrix}1&h\\0&d_i\end{bmatrix},\qquad
B_i=\begin{bmatrix}0\\b_i\end{bmatrix}.
\]

Let vehicle i's desired position be `p_ref(k) + r_i`, where `r_i` is its
fixed spacing offset from a common reference vehicle. Define the transformed
formation error state

\[
\xi_i(k)=\begin{bmatrix}
p_i(k)-p_{\rm ref}(k)-r_i\\
v_i(k)-v_{\rm ref}(k)
\end{bmatrix}.
\]

With the reference position update

\[
p_{\rm ref}(k+1)=p_{\rm ref}(k)+h v_{\rm ref}(k)
\]

and known reference speed, the feedforward input

\[
u_i^{\rm ff}(k)=
\frac{v_{\rm ref}(k+1)-d_i v_{\rm ref}(k)}{b_i}
\]

cancels the reference motion in the transformed velocity dynamics. The
feedback part can then use the existing local matching and learned consensus
correction on `xi_i`:

\[
u_i(k)=u_i^{\rm ff}(k)+K_i\xi_i(k)+\phi(W^T e_i(k)).
\]

For a stationary desired convoy speed, the feedforward term can be constant.
The target matrix and local matching constraints must be checked for the chosen
`h`, `d_i`, and `b_i`; do not assume every parameter choice is compatible.

## Communication and adversary model

Use two distinct graphs:

- **Physical order graph:** vehicles' front/rear relationships used to calculate
desired gaps and display the convoy.
- **V2V communication graph:** redundant communication links used by the
  distributed controller and W-MSR filter.

Do not use only a line/path communication graph if you intend to demonstrate a
W-MSR guarantee. A path is connected but generally fails the required
robustness condition for F=1. Build a redundant V2V mesh and explicitly run the
source library's `assess_wmsr_topology()` check on the normal-node induced graph.

A Byzantine vehicle can send different fake position/speed reports to
neighbouring cars. Compare:

1. Paper GNN, raw V2V reports.
2. Paper GNN + W-MSR.
3. Small-gain controller, raw V2V reports.
4. Small-gain controller + W-MSR.
5. Local-only/ACC-like reference, clearly labelled as a non-message-consuming
   reference rather than the primary comparator.

Use the same plant parameters, reference trajectory, desired spacing, attack,
initial conditions and communication graph for every method.

## What to display and measure

- Animated vehicle positions on a road, each with its vehicle ID and speed.
- Dashed outlines for desired gap/formation positions.
- A red or amber marker for the compromised transmitter.
- A side panel showing the forged V2V values and which values W-MSR trims.
- Per-vehicle spacing error and speed error.
- Maximum normal-vehicle spacing error, RMS formation error and settling/recovery
  time after the attack begins.
- Minimum physical inter-vehicle gap and a **safety-violation indicator**.
- Actual actuator commands and cumulative control effort.

A separate collision-avoidance/safety layer would be required before claiming
collision safety. W-MSR filters communicated values; it does not stop a
compromised vehicle from physically braking, accelerating, or jamming links.
Similarly, the component-wise W-MSR rule on position and velocity is a
coordinate-wise extension, not automatically a vector-state resilient-control
theorem.

## Why it helps communicate the result

The current experiment shows that corrupted neighbour information can cause
raw message-based controllers to retain a steady disagreement, while W-MSR
filtering can reject extreme forged reports on a topology that meets the stated
robustness condition. A vehicle platoon makes that mechanism tangible: an
attacker lies about its position/speed, and the viewer can see the other cars'
spacing response with and without filtering.

This is a **new model and training task**. The current consensus checkpoint
should not be reused blindly: retrain the paper GNN and small-gain controller
on the transformed formation-error system using the same training/validation
streams and evaluate them on matched held-out attacks. The transformed dynamics
retain the existing two-state structure if every physical plant satisfies the
first-row matching requirement and the selected Schur-stable target is used.

## Why this should be the next simulation

The current GUI is useful for isolating message filtering, but it remains an
abstract consensus experiment in which all normal states aim for zero. A
platoon gives the viewer a concrete task: preserve desired inter-vehicle gaps
while moving at a reference speed. The attack is also easy to explain: one car
transmits false position/speed reports to its V2V neighbours.

The primary plots should be normal-car spacing error over time, maximum gap
error, settling/recovery after attack onset, control effort, minimum physical
gap, and the W-MSR trimmed-message audit. Compare paper GNN raw/W-MSR and
small-gain raw/W-MSR under matched runs; keep local-only/ACC as a secondary
reference. Add a separate safety controller if collision avoidance is
required—W-MSR alone is not a collision-avoidance algorithm.

# Part B: Pre-trained Walking Policy Deployment

## Changes Made for Circular Locomotion

In `unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py`, the default setup sends a straight forward velocity command:

```python
cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
```

To make the robot walk in a continuous circle, I modified the command vector override line to:

```python
cmd = np.array([0.5, 0.0, 0.5], dtype=np.float32)
```

### Why this produces circular motion

The velocity command array is formatted as $[v_x, v_y, \omega_z]^T$:

* $v_x = 0.5 \text{ m/s}$: Forward walking velocity.
* $v_y = 0.0 \text{ m/s}$: Lateral (sideways) velocity.
* $\omega_z = 0.5 \text{ rad/s}$: Yaw angular velocity (turning rate).

By combining a constant linear forward velocity ($v_x$) with a continuous angular turning rate ($\omega_z$), the policy drives the robot forward while turning its heading angle at a steady rate, resulting in a smooth circular path on the ground plane.

---

## Policy Architecture & Control Questions

### 1. What inputs does the policy receive, and where do they come from?

The PyTorch policy network receives a 47-dimensional observation vector (`obs`) constructed inside the decimation control loop of `deploy_mujoco.py`:

1. **Angular Velocity ($\mathbf{\omega}$)**: Read from the base velocity tensor `d.qvel[3:6]` in MuJoCo, scaled by `ang_vel_scale`.
2. **Projected Gravity Vector**: Computed from the IMU root quaternion (`d.qpos[3:7]`) via `get_gravity_orientation()` to inform the network of body tilt relative to vertical gravity.
3. **Command Vector ($\mathbf{v}$)**: The target velocity vector $[v_x, v_y, \omega_z]$, scaled by `cmd_scale`.
4. **Joint Positions ($q$)**: Derived from current joint positions `d.qpos[7:]`, normalized relative to default standing pose (`default_angles`), and scaled by `dof_pos_scale`.
5. **Joint Velocities ($\dot{q}$)**: Read from joint velocities `d.qvel[6:]` and scaled by `dof_vel_scale`.
6. **Previous Action History ($a_{t-1}$)**: The action array output from the network's previous inference step.
7. **Phase Clock ($\sin(\phi), \cos(\phi)$)**: Evaluated from current simulation time (`counter * simulation_dt`) and gait period ($0.8 \text{ s}$) to give the policy an internal temporal rhythm for foot swing and stance timing.

---

### 2. What does the policy output, and how is that converted into joint torques?

* **Policy Output**: The policy neural network outputs a vector of target normalized joint position offsets ($a_t$).
* **Conversion to Joint Torques**:
  1. **Scaling**: The raw network output is converted into target joint angles (`target_dof_pos`) by scaling the action vector and adding nominal standing angles:

     $$\text{target\_dof\_pos} = a_t \times \text{action\_scale} + \text{default\_angles}$$

  2. **PD Control**: A Proportional-Derivative (PD) controller converts position targets into individual motor torques ($\tau$):

     $$\tau = K_p \times (\text{target\_dof\_pos} - q) - K_d \times \dot{q}$$

     where $q$ is current joint position (`d.qpos[7:]`), $\dot{q}$ is current joint velocity (`d.qvel[6:]`), $K_p = 100$ is proportional gain, and $K_d = 5$ is derivative gain.
  3. **Actuator Execution**: The resulting torque vector $\tau$ is applied directly to MuJoCo actuators (`d.ctrl[:] = tau`) before invoking `mujoco.mj_step()` to advance the physical simulation.
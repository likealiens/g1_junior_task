# Part A: Human Motion Retargeting (Unitree G1)

This folder contains the setup for retargeting human motion onto the 29-DoF Unitree G1 humanoid robot using MuJoCo physics simulation.

---

## Motion Overview & Prompt

- **Source**: Motion sequence generated via UltimateBots Studio.
- **Text Prompt**:
  > *"Humanoid robot motion sequence: Keep feet planted completely flat and stationary on the floor at all times (zero foot displacement, no walking or stepping). The robot extends both arms straight forward, performs two smooth squats, flexes its right bicep, brings its right hand up to its face/lips to blow a fly kiss, and opens its right arm wide to send the kiss."*

---

## How to Reproduce the Result

### 1. Prerequisites

Ensure you have the required Python packages installed:

```bash
pip install mujoco opencv-python pandas numpy
```

### 2. Execution

1. Place `motion.csv`, `play_motion.py`, and the `unitree_rl_gym` repository folder in the same root directory.
2. Run the playback script:

```bash
python play_motion.py
```

3. The script will initialize the Unitree G1 model in MuJoCo, execute a two-pass foot-ground anchoring algorithm to keep the soles anchored to $z = 0$, and render the performance into `motion.mp4`.

---

## CSV Column Mapping Explanation

### How did you determine what the columns in the CSV represent?

The exported LAFAN CSV file maps directly to the generalized coordinates of the humanoid robot in MuJoCo:

1. **Columns 0 to 2**: Represent the root base translation ($x, y, z$) in 3D world space coordinates.
2. **Columns 3 to 6**: Represent the root base orientation quaternion ($w, x, y, z$).
3. **Columns 7 to 35**: Represent the individual target joint angles across the robot's 29 Degrees of Freedom (DoFs).

We verified this alignment by inspecting the G1 robot XML structure in MuJoCo (`model.nq`). In MuJoCo, a free-joint base requires 7 root coordinates (3 translation + 4 rotation quaternion), while indices `7` through `35` map directly to the leg, waist, torso, and arm joint angle positions.
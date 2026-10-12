import os
import mujoco
import numpy as np
import pandas as pd
import cv2

# 1. Path to motion CSV
csv_path = "motion.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("motion.csv was not found in robotics_internship folder!")

# 2. Locate full G1 robot model
xml_path = "unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml"
if not os.path.exists(xml_path):
    for root, dirs, files in os.walk("unitree_rl_gym"):
        for file in files:
            if "g1" in file.lower() and file.endswith(".xml") and "12dof" not in file:
                xml_path = os.path.join(root, file)
                break

print(f"Loading full-body robot model from: {xml_path}")

# 3. Load MuJoCo model & renderer
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# Configure camera facing the front
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2.4
camera.elevation = -10
camera.azimuth = 0
camera.lookat = [0, 0, 0.75]

# Identify foot/ankle body IDs to anchor to ground plane
foot_body_ids = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and ("ankle" in name.lower() or "foot" in name.lower()):
        foot_body_ids.append(i)

if not foot_body_ids:
    # Fallback to lower leg links if specific foot name isn't found
    foot_body_ids = [i for i in range(model.nbody) if "link" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()][-2:]

# 4. Read motion CSV data
df = pd.read_csv(csv_path).select_dtypes(include=[np.number])
motion_data = df.to_numpy()

frames = []
fps = 30
sole_ground_offset = 0.035  # Thickness of sole to floor plane

print(f"Rendering {len(motion_data)} frames with Foot-Ground Anchoring...")

# 5. Playback loop with 2-Pass Foot Anchoring
for row in motion_data:
    if model.nq >= 7:
        # Lock horizontal position (X, Y) so robot doesn't drift sideways
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        
        # Set dummy root Z height for pass 1
        data.qpos[2] = 1.0
        
        # Apply torso orientation quaternion from CSV
        if np.linalg.norm(row[3:7]) > 0.1:
            data.qpos[3:7] = row[3:7]
        else:
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        # Apply joint angles from CSV
        num_joints = min(len(row) - 7, model.nq - 7)
        if num_joints > 0:
            data.qpos[7:7+num_joints] = row[7:7+num_joints]

    # Pass 1: Compute kinematics to see where feet land
    mujoco.mj_forward(model, data)
    
    # Calculate current lowest foot position
    lowest_foot_z = min(data.xpos[bid][2] for bid in foot_body_ids)
    
    # Pass 2: Adjust root Z position so lowest foot sits exactly on floor surface
    required_root_z = 1.0 - (lowest_foot_z - sole_ground_offset)
    data.qpos[2] = required_root_z
    
    # Final forward kinematics pass
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    frames.append(pixels)

# 6. Export MP4 Video
output_file = "motion.mp4"
height, width, _ = frames[0].shape
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
print(f"Success! Anchored video saved to {output_file}")
import ikpy
from ikpy.chain import Chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import numpy as np

# Joint angles for all joints. Number of joints must equal number of links, so first joint here
# appears to be a dummy whose value has no effect. Each joint in our chain is a revolute joint,
# which takes an angle in radians, or a fixed joint, which cannot be changed.
joint_angles = [0, 0, 0, 0, 0, 0, 0, 0]
#joint_angles = [0, 0, 0, np.deg2rad(-10), 0, np.deg2rad(-45), 0, 0]

# The target position for the last link in our chain
target = [-120, 150, 50]

# Load our robot
my_chain = Chain.from_urdf_file("robot-arm.urdf", active_links_mask=[ False, False, True, True, False, True, True, False ])

# Forward kinematics computes a transform matrix for the final link
m = my_chain.forward_kinematics(joints=joint_angles)
print(m)

# Inverse kinematics returns a list of joint angles
joint_angles = my_chain.inverse_kinematics(target_position=target)

# Plot
fig, ax = plot_utils.init_3d_figure()
my_chain.plot(joints=joint_angles, ax=ax, target=target)
plt.xlim(-100,100)
plt.ylim(-100,100)
ax.set_zlim(0, 200)
plt.show()

from franky import Affine, JointWaypointMotion, JointWaypoint, CartesianMotion, Robot, Gripper, ReferenceType

robot = Robot("172.16.3.1")
gripper = Gripper("172.16.3.1")

gripper.open(0.02)
# gripper.move(0.0, 0.02)


# robot.relative_dynamics_factor = 0.05

# imp_value = 300
# # robot.set_cartesian_impedance([2000., 2000, 2000, 200., 200, 200])
# robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

# m1 = JointWaypointMotion([JointWaypoint([0.15, -0.61, -0.29, -2.12, -0.0, 1.79, 0.82])])
# robot.move(m1)

# z = 0.05
# for i in range(20):
#     motion = CartesianMotion(Affine([0.0, 0.0, z]), ReferenceType.Relative)
#     z = -z
#     robot.move(motion)
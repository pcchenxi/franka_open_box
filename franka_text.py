from franky import Affine, JointWaypointMotion, JointWaypoint, CartesianMotion, Robot, ReferenceType

robot = Robot("172.16.3.1")
robot.relative_dynamics_factor = 0.05

robot.set_cartesian_impedance([10, 10, 10, 200, 200, 200])

m1 = JointWaypointMotion([JointWaypoint([0.15, -0.61, -0.29, -2.12, -0.0, 1.79, 0.82])])
robot.move(m1)

z = 0.05
for i in range(10):
    motion = CartesianMotion(Affine([0.0, 0.0, z]), ReferenceType.Relative)
    z = -z
    robot.move(motion)
import time
import numpy as np
import pybullet as p
import pybullet_data
import math
import pandas as pd
import os
from abc import ABC, abstractmethod

# ---------- Constants ----------
BASE_POSITION = np.array([0, 0, 0.75])
BASE_ORIENTATION = np.array([0, np.pi/4, 0])
INITIAL_POSITIONS = [0.550569, 0.0, 0.549657, 0.0]
GRASP_DISTANCE = 0.5
GRASP_VALID_TOTAL=100

# Path of the folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative path: src → data
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Relative path: src → urdf
URDF_BASE_PATH = os.path.join(BASE_DIR, "..", "urdf")

# ---------- Base Interfaces ----------
class SceneObject(ABC):
    """Base interface for any object in the PyBullet scene."""
    def __init__(self, urdf_file, position, orientation=(0, 0, 0)):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.id = None
        self.name = None
        self.grasp_height = 0.03  # Default grasp height

    def load(self):
        """Load the object into the scene."""
        urdf_path = os.path.join(URDF_BASE_PATH, self.urdf_file)
        self.id = p.loadURDF(urdf_path, self.position, self.orientation)
        return self.id
    
    @abstractmethod
    def update_name(self):
        """Update object name based on ID."""
        pass

    def get_position(self):
        """Get current object position."""
        return p.getBasePositionAndOrientation(self.id)[0]

    def reset_position(self, position=None, orientation=None):
        """Reset object to initial or specified position."""
        pos = position if position is not None else self.position
        ori = orientation if orientation is not None else self.orientation
        p.resetBasePositionAndOrientation(self.id, pos, ori)


class GraspableObject(SceneObject):
    """Interface for objects that can be grasped."""
    @abstractmethod
    def get_grasp_point(self):
        """Get the optimal grasp point for this object."""
        pass


class GripperInterface(ABC):
    """Base interface for all grippers."""
    def __init__(self, urdf_path, base_position):
        self.urdf_path = urdf_path
        self.base_position = base_position
        self.id = None
        self.constraint_id = None
        self.num_joints = 0

    @abstractmethod
    def load(self):
        """Load gripper into the PyBullet world."""
        pass

    @abstractmethod
    def open(self):
        """Open the gripper."""
        pass

    @abstractmethod
    def close(self):
        """Close the gripper."""
        pass

    @abstractmethod
    def grasp_and_lift(self, obj, position, orientation, lift_height=0.4, lift_steps=150):
        """Execute grasp and lift sequence."""
        pass

    def move(self, position=None, orientation=None):
        """Move gripper to a new position and orientation."""
        if self.constraint_id is None:
            raise ValueError("Gripper must be fixed before moving.")
        p.changeConstraint(
            self.constraint_id,
            jointChildPivot=position,
            jointChildFrameOrientation=orientation,
            maxForce=200
        )

    def attach_fixed(self, offset):
        """Attach gripper to a fixed world position."""
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=self.base_position
        )

    def move_forward(self, distance):
        """Move gripper forward along its current orientation."""
        if self.constraint_id is None:
            raise ValueError("Gripper must be fixed before moving.")

        current_pos, current_quat = p.getBasePositionAndOrientation(self.id)
        rotation_matrix = p.getMatrixFromQuaternion(current_quat)
        forward_vector = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
        target_position = [
            current_pos[0] + distance * forward_vector[0],
            current_pos[1] + distance * forward_vector[1], 
            current_pos[2] + distance * forward_vector[2]
        ]
        print(target_position)
        p.changeConstraint(
            self.constraint_id,
            jointChildPivot=target_position,
            jointChildFrameOrientation=current_quat,
            maxForce=5
        )
        return target_position

    @staticmethod
    def update_camera():
        """Update debug visualizer camera."""
        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=135,
            cameraPitch=-30,
            cameraTargetPosition=[-0.1, -0.1, 0.3]
        )


# ---------- Object Implementations ----------
class Box(GraspableObject):
    def __init__(self, position):
        super().__init__("cube_small.urdf", position, (0, 0, 0))
        self.grasp_height = 0.03

    def update_name(self):
        self.name = f"Box_{self.id}"

    def get_grasp_point(self):
        return self.position


class Cylinder(GraspableObject):
    def __init__(self, position):
        super().__init__("cylinder.urdf", position, (0, 0, 0))
        self.grasp_height = 0.1

    def update_name(self):
        self.name = f"Cylinder_{self.id}"

    def get_grasp_point(self):
        return self.position


# ---------- Gripper Implementations ----------
class TwoFingerGripper(GripperInterface):
    def __init__(self, base_position=BASE_POSITION):
        super().__init__("pr2_gripper.urdf", base_position)

    def load(self):
        self.id = p.loadURDF(self.urdf_path, *self.base_position)
        self.num_joints = p.getNumJoints(self.id)
        return self.id

    def start(self):
        """Open gripper at start."""
        for i, pos in enumerate(INITIAL_POSITIONS):
            p.resetJointState(self.id, i, pos)

    def open(self):
        """Open the gripper fingers."""
        for joint in [0, 2]:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=0.0, maxVelocity=1, force=100
            )

    def close(self):
        """Close the gripper fingers."""
        for joint in [0, 2]:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=0.1, maxVelocity=1, force=10
            )

    def grasp_and_lift(self, obj, position, orientation, lift_height=0.4):
        """Control Sequence, with smooth approach and lift."""
        # Record initial position
        cube_center = obj.get_grasp_point()
        obj_pos_before = obj.get_position()
        
        # Set contact parameters
        p.changeDynamics(obj.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
        p.changeDynamics(self.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
        
        # Initial stabilization
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./720.)
        
        # Move to start position
        self.move(position, orientation)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./720.)

        # Smooth downward movement to grasp position
        current_pos, current_quat = p.getBasePositionAndOrientation(self.id)
        transform = p.getMatrixFromQuaternion(current_quat)
        forward_vector = [transform[0], transform[3], transform[6]]
        
        approach_pos = [
            current_pos[0] + GRASP_DISTANCE * forward_vector[0],
            current_pos[1] + GRASP_DISTANCE * forward_vector[1], 
            current_pos[2] + GRASP_DISTANCE * forward_vector[2]
        ]
        
        # Use interpolation for smooth movement (key part)
        for interp in np.linspace(current_pos, approach_pos, 100):
            p.changeConstraint(
                self.constraint_id,
                jointChildPivot=interp,
                jointChildFrameOrientation=current_quat,
                maxForce=100  # Increase constraint force
            )
            p.stepSimulation()
            time.sleep(1./720.)  # Control movement speed

        # Close gripper
        for joint in [0, 2]:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=0.12, force=300, maxVelocity=10
            )
        
        # Wait for contact stabilization
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./720.)

        # Smooth lifting
        current_pos, current_quat = p.getBasePositionAndOrientation(self.id)
        lift_pos = [current_pos[0], current_pos[1], lift_height]
        
        # Use interpolation for smooth lifting
        for interp in np.linspace(current_pos, lift_pos, 200):
            p.changeConstraint(
                self.constraint_id,
                jointChildPivot=interp,
                jointChildFrameOrientation=current_quat,
                maxForce=100
            )
            p.stepSimulation()
            time.sleep(0.01)

        # Record results
        obj_pos_after = obj.get_position()
        return self._evaluate_grasp_result(obj_pos_before, obj_pos_after, position, orientation)

    def _evaluate_grasp_result(self, pos_before, pos_after, grasp_position, grasp_orientation):
        """Evaluate grasp result"""
        # Calculate position change
        pos_change = np.linalg.norm(np.array(pos_after) - np.array(pos_before))
        z_change = pos_after[2] - pos_before[2]
        
        # Get Euler angles
        euler_angles = p.getEulerFromQuaternion(grasp_orientation)
        roll, pitch, yaw = euler_angles
        
        print(f"Grasp position: X={grasp_position[0]:.3f}, Y={grasp_position[1]:.3f}, Z={grasp_position[2]:.3f}")
        print(f"Grasp angles: Roll={roll:.3f}, Pitch={pitch:.3f}, Yaw={yaw:.3f}")
        print(f"Initial position: {pos_before}")
        print(f"Final position: {pos_after}")
        print(f"Position change: {pos_change:.4f} m")
        print(f"Z-axis change: {z_change:.4f} m")
        
        # Evaluate result
        if pos_change < 0.003:  # Position basically unchanged
            print("Invalid test: Gripper did not touch the object")
            print("")
            return "invalid"
        elif z_change > 0.15:  # Z-axis raised more than 0.2 meters
            print("Grasp successful: Object effectively lifted")
            print("")
            return "success"
        else:
            print("Grasp failed: Object lift height insufficient")
            print("")
            return "failure"
        

class ThreeFingerHand(GripperInterface):
    """Three-finger hand implementation."""
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self, urdf_path="./grippers/threeFingers/sdh/sdh.urdf", 
                 base_position=(-0.5, -0.5, 0.5)):
        super().__init__(urdf_path, base_position)
        self.open_state = True
        self.hand_base_constraint = None
        self.gripper_id = None
        self.init_pos = base_position
        self.init_ori = None

    def load(self):
        """Load three-finger hand with specific orientation."""
        self.init_ori = p.getQuaternionFromEuler([3 * math.pi / 4, -1 * math.pi / 4, 3 * math.pi / 4])
        urdf_path = os.path.join(URDF_BASE_PATH, self.urdf_path)
        self.id = p.loadURDF(urdf_path, self.base_position, self.init_ori,
                            globalScaling=1, useFixedBase=False)

        self.num_joints = p.getNumJoints(self.id)
        
        self.constraint_id = p.createConstraint(
            self.id, -1, -1, -1,
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.init_ori
        )
        
        # Backward compatibility
        self.hand_base_constraint = self.constraint_id
        self.gripper_id = self.id
        
        return self.id

    def connect(self):
        """Connect to simulation and setup environment."""
        self.load()
        p.setRealTimeSimulation(0)
        return self.id

    def open(self):
        """Open the three-finger hand."""
        self.open_gripper()

    def close(self):
        """Close the three-finger hand."""
        self.preshape()

    def preshape(self):
        """Set hand to pre-shape configuration."""
        for joint in self.PRESHAPE_JOINTS:
            p.setJointMotorControl2(
                self.id, joint, p.POSITION_CONTROL,
                targetPosition=0.4, maxVelocity=2, force=1
            )
        self.open_state = False

    def open_gripper(self):
        """Open the three-finger hand with iterative approach."""
        closed, iteration = True, 0
        while closed and not self.open_state:
            joints = self.get_joint_positions()
            closed = False
            for k in range(self.num_joints):
                if k in [2, 5, 8] and joints[k] >= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.05)
                    closed = True
                elif k in [3, 6, 9] and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.05)
                    closed = True
                elif k in [1, 4, 7] and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.05)
                    closed = True
            iteration += 1
            if iteration > 10000:
                break
            p.stepSimulation()
        self.open_state = True

    def _apply_joint_command(self, joint, target, max_velocity=2, force=5):
        """Apply joint motor control command."""
        p.setJointMotorControl2(
            self.id, joint, p.POSITION_CONTROL,
            targetPosition=target, maxVelocity=max_velocity, force=force
        )

    def get_joint_positions(self):
        """Get current joint positions."""
        return [p.getJointState(self.id, i)[0] for i in range(self.num_joints)]

    def grasp_and_lift(self, obj, position, orientation):
        """Use original three-finger hand grasp program"""
        cube_center = obj.get_grasp_point()
        # Move to start position
        p.changeConstraint(self.hand_base_constraint, position,
                        jointChildFrameOrientation=orientation, maxForce=300)
        time.sleep(0.3)

        # Open gripper
        self.open_state = False
        self.open_gripper()

        # --- Calculate approach point ---
        dir_vec = cube_center - position
        dist0 = np.linalg.norm(dir_vec)
        dir_vec = dir_vec / dist0

        target_dist = max(0.05, dist0 - 0.3)
        approach_pos = position + dir_vec * (dist0 - target_dist)

        # Record object initial position
        obj_pos_before = obj.get_position()

        # Move to approach position
        for interp in np.linspace(position, approach_pos, 150):
            p.changeConstraint(self.hand_base_constraint, interp,
                            jointChildFrameOrientation=orientation, maxForce=50)
            p.stepSimulation()
            time.sleep(0.005)

        # --- Close gripper ---
        for j in [1, 4, 7]:
            p.setJointMotorControl2(self.gripper_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.05, maxVelocity=4, force=40)
        for j in [2, 5, 8]:
            p.setJointMotorControl2(self.gripper_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.3, maxVelocity=4, force=20)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        # --- Lift back ---
        for interp in np.linspace(approach_pos, position, 200):
            p.changeConstraint(self.hand_base_constraint, interp,
                            jointChildFrameOrientation=orientation, maxForce=50)
            p.stepSimulation()
            time.sleep(0.005)

        # Maintain grasp state for 2 seconds
        for _ in range(int(2 / 0.01)):
            p.stepSimulation()
            time.sleep(0.01)

        # --- Record results and evaluate ---
        obj_pos_after = obj.get_position()
        # Reset gripper shape
        self.open_state = False
        self.open_gripper()
        return self._evaluate_grasp_result(obj_pos_before, obj_pos_after, position, orientation)

    def _evaluate_grasp_result(self, pos_before, pos_after, grasp_position, grasp_orientation):
        """Evaluate grasp result"""
        # Calculate position change
        pos_change = np.linalg.norm(np.array(pos_after) - np.array(pos_before))
        z_change = pos_after[2] - pos_before[2]
        
        # Get Euler angles
        euler_angles = p.getEulerFromQuaternion(grasp_orientation)
        roll, pitch, yaw = euler_angles
        
        print(f"Grasp position: X={grasp_position[0]:.3f}, Y={grasp_position[1]:.3f}, Z={grasp_position[2]:.3f}")
        print(f"Grasp angles: Roll={roll:.3f}, Pitch={pitch:.3f}, Yaw={yaw:.3f}")
        print(f"Initial position: {pos_before}")
        print(f"Final position: {pos_after}")
        print(f"Position change: {pos_change:.4f} m")
        print(f"Z-axis change: {z_change:.4f} m")
        
        # Evaluate result
        if pos_change < 0.003:  # Position basically unchanged
            print("Invalid test: Gripper did not touch the object")
            return "invalid"
        elif z_change > 0.2:  # Z-axis raised more than 0.2 meters
            print("Grasp successful: Object effectively lifted")
            return "success"
        else:
            print("Grasp failed: Object lift height insufficient")
            return "failure"

# ---------- Utility Functions ----------
def generate_random_gripper_pose(object_center=np.array([0, 0, 0.05]),
                                 radius_min=0.4, radius_max=0.47, angle_limit_deg=45):
    

    """Generate random gripper pose around an object."""
    r = np.random.uniform(radius_min, radius_max)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi * angle_limit_deg / 180)

    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    position = object_center + np.array([x, y, z])
    return position, None


def compute_gripper_orientation_towards_object(gripper_pos, object_pos, x_rot=0, y_rot=0, z_rot=0):
    """Compute gripper orientation towards object with optional rotations."""
    forward = np.array(object_pos) - np.array(gripper_pos)
    forward /= np.linalg.norm(forward)
    
    world_up = np.array([0, 0, 1])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0, 1, 0])
    
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    
    rot_matrix = np.array([
        [forward[0], right[0], up[0]],
        [forward[1], right[1], up[1]], 
        [forward[2], right[2], up[2]]
    ])

    # Convert rotation matrix to quaternion
    trace = rot_matrix[0,0] + rot_matrix[1,1] + rot_matrix[2,2]
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_matrix[2,1] - rot_matrix[1,2]) * s
        y = (rot_matrix[0,2] - rot_matrix[2,0]) * s  
        z = (rot_matrix[1,0] - rot_matrix[0,1]) * s
    else:
        if rot_matrix[0,0] > rot_matrix[1,1] and rot_matrix[0,0] > rot_matrix[2,2]:
            s = 2.0 * math.sqrt(1.0 + rot_matrix[0,0] - rot_matrix[1,1] - rot_matrix[2,2])
            w = (rot_matrix[2,1] - rot_matrix[1,2]) / s
            x = 0.25 * s
            y = (rot_matrix[0,1] + rot_matrix[1,0]) / s
            z = (rot_matrix[0,2] + rot_matrix[2,0]) / s
        elif rot_matrix[1,1] > rot_matrix[2,2]:
            s = 2.0 * math.sqrt(1.0 + rot_matrix[1,1] - rot_matrix[0,0] - rot_matrix[2,2])
            w = (rot_matrix[0,2] - rot_matrix[2,0]) / s
            x = (rot_matrix[0,1] + rot_matrix[1,0]) / s
            y = 0.25 * s
            z = (rot_matrix[1,2] + rot_matrix[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + rot_matrix[2,2] - rot_matrix[0,0] - rot_matrix[1,1])
            w = (rot_matrix[1,0] - rot_matrix[0,1]) / s
            x = (rot_matrix[0,2] + rot_matrix[2,0]) / s
            y = (rot_matrix[1,2] + rot_matrix[2,1]) / s
            z = 0.25 * s

    original_orientation = [x, y, z, w]
    original_euler = p.getEulerFromQuaternion(original_orientation)
    
    final_euler = [
        original_euler[0] + x_rot,
        original_euler[1] + y_rot,
        original_euler[2] + z_rot
    ]
    
    return p.getQuaternionFromEuler(final_euler)


def setup_environment():
    """Setup PyBullet simulation environment."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf")


# ---------- Main Demo Functions ----------
def run_two_finger_demo(object_type="box"):
    """Run demo with two-finger gripper."""
    setup_environment()
    
    # Create object based on selection
    if object_type == "box":
        obj = Box([0, 0,0.05])
    elif object_type == "cylinder":
        obj = Cylinder([0, 0, 0.05])
    else:
        print("Invalid object type, defaulting to box")
        obj = Box([0, 0, 0.05])
        
    obj_id = obj.load()
    obj.update_name()
    
    # Statistics results
    results = {
        "success": 0,
        "failure": 0,
        "invalid": 0
    }
    
    valid_grasp_data = []  # Store valid test grasp data
    valid_grasp_data_for_csv=[]
    total_valid_grasps = GRASP_VALID_TOTAL  # Target: 10 valid grasps
    current_attempt = 0
    

    while len(valid_grasp_data) < total_valid_grasps:
        current_attempt += 1
        
        # Create and setup gripper
        gripper = TwoFingerGripper()
        gripper.load()
        gripper.start()
        gripper.attach_fixed(offset=[0.2, 0, 0])
        gripper.update_camera()

        # Generate random pose
        radius_offset = -0.04 if object_type == "box" else 0.0
        position, _ = generate_random_gripper_pose(
            object_center=obj.get_grasp_point(),
            radius_min=0.4 + radius_offset,
            radius_max=0.47 + radius_offset,
            angle_limit_deg=75
        )

        orientation = compute_gripper_orientation_towards_object(position, obj.get_grasp_point())

        # Print grasp info
        obj_pos = obj.get_grasp_point()
        dist = np.linalg.norm(position - obj_pos)
        euler = p.getEulerFromQuaternion(orientation)
        
        print(f"Gripper position: {position}")
        print(f"Object position: {obj_pos}")
        print(f"Grasp orientation (quaternion): {orientation}")

        # Execute grasp and get result
        result = gripper.grasp_and_lift(obj, position, orientation)
        results[result] += 1
        
        # Skip adding to valid dataset if one class already has enough samples
        if result == "success" and results["success"] > GRASP_VALID_TOTAL / 2:
            if gripper.constraint_id is not None:
                p.removeConstraint(gripper.constraint_id)
                p.removeBody(gripper.id)
            # Reset object for next attempt
            obj.reset_position()
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./720.)
            continue
        if result == "failure" and results["failure"] > GRASP_VALID_TOTAL / 2:
            if gripper.constraint_id is not None:
                p.removeConstraint(gripper.constraint_id)
                p.removeBody(gripper.id)
            # Reset object for next attempt
            obj.reset_position()
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./720.)
            continue

        # If valid test, record grasp data
        if result != "invalid":
            euler_angles = p.getEulerFromQuaternion(orientation)
            grasp_data = {
                "position": position,
                "orientation": orientation,
                "euler_angles": euler_angles,
                "result": result,
                "attempt_number": current_attempt
            }
            valid_grasp_data.append(grasp_data)
            valid_grasp_data_for_csv.append({
                "Result": result,
                "PosX": position[0],
                "PosY": position[1],
                "PosZ": position[2],
                "AngX": orientation[0],
                "AngY": orientation[1],
                "AngZ": orientation[2],
                "AngW": orientation[3],
            })
            print(f"Obtained {len(valid_grasp_data)} valid grasps")
        else:
            print("This attempt is invalid, continuing...")

        # Cleanup
        if gripper.constraint_id is not None:
            p.removeConstraint(gripper.constraint_id)
        p.removeBody(gripper.id)

        # Reset object for next attempt
        obj.reset_position()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./720.)

    # Print final statistics
    print(f"Total attempts: {current_attempt}")
    print(f"Success count: {results['success']}")
    print(f"Failure count: {results['failure']}") 
    print(f"Invalid tests: {results['invalid']}")
    print(f"Total valid tests: {len(valid_grasp_data)}")
    print(f"Success rate: {(results['success']/current_attempt)*100:.1f}%")
    print(f"Valid grasp rate: {len(valid_grasp_data)/current_attempt*100:.1f}%")
    
    # Print valid test grasp data
    if valid_grasp_data:
        print("\n")
        for i, data in enumerate(valid_grasp_data):
            pos = data["position"]
            result_symbol = "√" if data["result"] == "success" else "X"
            ori = data["orientation"]
            print(f"Valid test {i+1} {result_symbol} (Attempt {data['attempt_number']}): "
                f"Position(X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}), "
                f"Orientation(quaternion) = {ori}")
    df = pd.DataFrame(valid_grasp_data_for_csv)
    output_file = f"grasp_results_two_finger_{object_type}.csv"
    df.to_csv(os.path.join(DATA_DIR, output_file), index=False)
    p.removeBody(obj.id)
    p.disconnect()

def run_three_finger_demo(object_type="box"):
    """Run three-finger hand demo using original grasp program"""
    setup_environment()
    
    # Create three-finger hand instance and connect
    hand = ThreeFingerHand()
    hand.connect()
    
    # Create object based on selection
    if object_type == "box":
        obj = Box([0, 0, 0.05])
    elif object_type == "cylinder":
        obj = Cylinder([0, 0, 0.05])
    else:
        print("Invalid object type, defaulting to box")
        obj = Box([0, 0, 0.05])
        
    obj_id = obj.load()
    obj.update_name()
    
    # Statistics results
    results = {
        "success": 0,
        "failure": 0,
        "invalid": 0
    }
    
    valid_grasp_data = []  # Store valid test grasp data
    valid_grasp_data_for_csv=[]
    total_valid_grasps = GRASP_VALID_TOTAL  # Target: 10 valid grasps
    current_attempt = 0
    

    while len(valid_grasp_data) < total_valid_grasps:
        current_attempt += 1
        
        # Generate random start pose
        object_center = obj.get_grasp_point()
        radius_offset = -0.03 if object_type == "box" else 0.0
        start_pos = generate_random_gripper_pose(
            object_center=object_center,
            radius_min=0.45 + radius_offset,
            radius_max=0.60 + radius_offset,
            angle_limit_deg=75
        )[0]

        hand_ori = compute_gripper_orientation_towards_object(start_pos, object_center, y_rot=np.pi/2)

        # Print grasp info
        dist = np.linalg.norm(start_pos - object_center)
        print(f"Gripper position: {start_pos}")
        print(f"Object position: {object_center}")
        print(f"Grasp orientation (quaternion): {hand_ori}")


        # Execute grasp and get result
        result = hand.grasp_and_lift(obj, start_pos, hand_ori)
        results[result] += 1

        # Skip adding to valid dataset if one class already has enough samples
        if result == "success" and results["success"] > GRASP_VALID_TOTAL / 2:
            # Reset object for next attempt
            obj.reset_position()
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./720.)
            continue
        if result == "failure" and results["failure"] > GRASP_VALID_TOTAL / 2:
            # Reset object for next attempt
            obj.reset_position()
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./720.)
            continue
        
        # If valid test, record grasp data
        if result != "invalid":
            euler_angles = p.getEulerFromQuaternion(hand_ori)
            grasp_data = {
                "position": start_pos,
                "orientation": hand_ori,
                "euler_angles": euler_angles,
                "result": result,
                "attempt_number": current_attempt
            }
            valid_grasp_data_for_csv.append({
                "Result": result,
                "PosX": start_pos[0],
                "PosY": start_pos[1],
                "PosZ": start_pos[2],
                "AngX": hand_ori[0],
                "AngY": hand_ori[1],
                "AngZ": hand_ori[2],
                "AngW": hand_ori[3],
            })
            valid_grasp_data.append(grasp_data)
            print(f"Obtained {len(valid_grasp_data)} valid grasps")
        else:
            print(f"This attempt is invalid, continuing...")
        
        # Reset object for next attempt
        obj.reset_position()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./720.)

    # Print final statistics
    print("\n=== Final Statistics ===")
    print(f"Total attempts: {current_attempt}")
    print(f"Success count: {results['success']}")
    print(f"Failure count: {results['failure']}") 
    print(f"Invalid tests: {results['invalid']}")
    print(f"Total valid tests: {len(valid_grasp_data)}")
    print(f"Success rate: {results['success']/current_attempt*100:.1f}%")
    print(f"Valid grasp rate: {len(valid_grasp_data)/current_attempt*100:.1f}%")
    
    # Print valid test grasp data
    if valid_grasp_data:
        print("\n=== Valid Test Grasp Data ===")
        for i, data in enumerate(valid_grasp_data):
            pos = data["position"]
            result_symbol = "√" if data["result"] == "success" else "X"
            ori = data["orientation"]
            print(f"Valid test {i+1} {result_symbol} (Attempt {data['attempt_number']}): "
                f"Position(X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}), "
                f"Orientation(quaternion) = {ori}")


    print(f"\nCompleted {total_valid_grasps} valid grasps, total attempts: {current_attempt}.")
    df = pd.DataFrame(valid_grasp_data_for_csv)
    output_file = f"grasp_results_three_finger_{object_type}.csv"
    df.to_csv(os.path.join(DATA_DIR, output_file), index=False)
    p.removeBody(obj.id)
    p.disconnect()


# ---------- Main Entry Point ----------
if __name__ == "__main__":
    print("1. Two-finger gripper")
    print("2. Three-finger gripper")
    gripper_choice = input("Select gripper type (1 or 2): ")
    print("1. Box")
    print("2. Cylinder")
    object_choice = input("Select object type (1 or 2): ")
    # Map choices to parameters
    object_type = "box" if object_choice == "1" else "cylinder"
    
    if gripper_choice == "1":
        run_two_finger_demo(object_type)
    elif gripper_choice == "2":
        run_three_finger_demo(object_type)
    else:
        print("Invalid gripper choice, defaulting to two-finger gripper demo")
        run_two_finger_demo(object_type)
"""URDF arm kinematics in a fixed torso frame for measured-body diagnostics."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
    "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
)


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("rotation axis must be nonzero")
    x, y, z = axis / norm
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


class TorsoArmKinematics:
    """Use the robot's actual joint origins; never predict unobserved base motion.

    Input: [...,14] measured arm joint positions in radians, seven per arm.
    Output: [...,2,9,3] positions in meters relative to torso_link, including
    the shoulder joints, elbow/wrist joints, and fixed palm endpoint.
    """

    def __init__(self, urdf_path: str | Path):
        root = ET.parse(urdf_path).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        self.chains = []
        self.link_names = []
        for side in ("left", "right"):
            names = [f"{side}_{suffix}" for suffix in ARM_JOINT_SUFFIXES]
            names.append(f"{side}_hand_palm_joint")
            parent = "torso_link"
            chain = []
            links = []
            for name in names:
                joint = joints[name]
                if joint.find("parent").attrib["link"] != parent:
                    raise ValueError(f"URDF arm chain is discontinuous at {name}")
                parent = joint.find("child").attrib["link"]
                links.append(parent)
                origin = joint.find("origin")
                xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                roll, pitch, yaw = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
                frame = np.eye(4)
                frame[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
                frame[:3, 3] = xyz
                axis = joint.find("axis")
                if joint.attrib["type"] not in {"revolute", "continuous", "fixed"}:
                    raise ValueError(f"unsupported URDF joint type: {name}")
                chain.append((frame, None if axis is None else np.fromstring(axis.attrib["xyz"], sep=" ")))
            self.chains.append(chain)
            self.link_names.append(tuple(links))

    def positions(self, arm_angles):
        return self.chain_transforms(arm_angles)[..., :3, 3]

    def link_transforms(self, arm_angles):
        frames = self.chain_transforms(arm_angles)
        result = {"torso_link": frames[..., 0, 0, :, :]}
        for side, names in enumerate(self.link_names):
            for step, name in enumerate(names):
                result[name] = frames[..., side, step + 1, :, :]
        return result

    def chain_transforms(self, arm_angles):
        """Actual link-to-torso transforms [...,2,9,4,4], root then 8 links."""
        values = np.asarray(arm_angles, dtype=float)
        if values.shape[-1] != 14 or not np.isfinite(values).all():
            raise ValueError("need fourteen finite measured arm angles in radians")
        out = np.empty((*values.shape[:-1], 2, 9, 4, 4))
        for index in np.ndindex(values.shape[:-1]):
            for side, chain in enumerate(self.chains):
                frame = np.eye(4)
                out[index + (side, 0)] = frame
                for step, (origin, axis) in enumerate(chain):
                    frame = frame @ origin
                    if axis is not None:
                        turn = np.eye(4)
                        turn[:3, :3] = rotation(axis, values[index + (side * 7 + step,)])
                        frame = frame @ turn
                    out[index + (side, step + 1)] = frame
        return out


class UrdfKinematics:
    """Named joint positions in radians to link-to-root rigid transforms.

    Unprovided moving joints are rejected unless explicitly listed as omitted
    roots. Omission removes their entire descendant branch, so no unmeasured
    finger configurations are invented. Fixed links remain attached.
    """

    def __init__(self, urdf_path, joint_names, *, omitted_joint_roots=()):
        root = ET.parse(urdf_path).getroot()
        self.joint_names = tuple(joint_names)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("named joint order must be unique")
        self.joint_indices = {name: i for i, name in enumerate(self.joint_names)}
        definitions = root.findall("joint")
        names = {j.attrib["name"] for j in definitions}
        omitted = set(omitted_joint_roots)
        if not set(self.joint_names).issubset(names) or not omitted.issubset(names):
            raise ValueError("controlled or omitted joint missing from URDF")
        if set(self.joint_names) & omitted:
            raise ValueError("a controlled joint cannot also be omitted")
        links = {link.attrib["name"] for link in root.findall("link")}
        children = {j.find("child").attrib["link"] for j in definitions}
        roots = links - children
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root")
        self.root_link = next(iter(roots))
        reachable = {self.root_link}
        excluded = set()
        pending = list(definitions)
        self.joints = []
        while pending:
            progress = False
            for joint in list(pending):
                parent = joint.find("parent").attrib["link"]
                child = joint.find("child").attrib["link"]
                name = joint.attrib["name"]
                if parent not in reachable and parent not in excluded:
                    continue
                pending.remove(joint)
                progress = True
                if name in omitted or parent in excluded:
                    if name in self.joint_indices:
                        raise ValueError("controlled joint lies inside an omitted branch")
                    excluded.add(child)
                    continue
                kind = joint.attrib["type"]
                if kind not in {"fixed", "revolute", "continuous"}:
                    raise ValueError(f"unsupported URDF joint type: {kind}")
                if kind != "fixed" and name not in self.joint_indices:
                    raise ValueError(f"moving joint has no supplied angle: {name}")
                origin = joint.find("origin")
                xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                frame = np.eye(4)
                frame[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
                frame[:3, 3] = xyz
                axis_node = joint.find("axis")
                axis = None if kind == "fixed" else np.fromstring(axis_node.attrib.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
                self.joints.append((name, parent, child, frame, axis))
                reachable.add(child)
            if not progress:
                raise ValueError("URDF links are disconnected or cyclic")
        self.link_names = tuple([self.root_link] + [item[2] for item in self.joints])
        self.child_joint = {item[2]: item[0] for item in self.joints}
        self.parent_link = {item[2]: item[1] for item in self.joints}
        self.omitted_joint_roots = tuple(sorted(omitted))

    def link_transforms(self, angles):
        values = np.asarray(angles, dtype=float)
        if values.shape[-1] != len(self.joint_names) or not np.isfinite(values).all():
            raise ValueError("need one finite radian value per named URDF joint")
        shape = values.shape[:-1]
        frames = {name: np.empty((*shape, 4, 4)) for name in self.link_names}
        for index in np.ndindex(shape):
            frames[self.root_link][index] = np.eye(4)
            for name, parent, child, origin, axis in self.joints:
                frame = frames[parent][index] @ origin
                if axis is not None:
                    turn = np.eye(4)
                    turn[:3, :3] = rotation(axis, values[index + (self.joint_indices[name],)])
                    frame = frame @ turn
                frames[child][index] = frame
        return frames

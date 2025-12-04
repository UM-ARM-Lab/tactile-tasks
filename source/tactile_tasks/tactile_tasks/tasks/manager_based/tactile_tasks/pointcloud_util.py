# pointcloud_util.py

import torch
import warp as wp
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import omni.usd
from isaaclab.utils.math import transform_points
from scipy.spatial.distance import pdist
import math
from pxr import Gf


def get_local_transform_to_root(prim, root_prim):
    """
    Get the local transform from a prim to the root prim.
    Returns (translation, rotation_quat) where rotation_quat is (w, x, y, z).
    """
    # Get the transform from prim to root
    # We need to walk up the hierarchy and accumulate transforms
    from pxr import UsdGeom
    
    # Get world transform of prim
    prim_xform = UsdGeom.Xformable(prim)
    if prim_xform:
        prim_world_transform = prim_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    else:
        prim_world_transform = Gf.Matrix4d(1.0)  # Identity
    
    # Get world transform of root
    root_xform = UsdGeom.Xformable(root_prim)
    if root_xform:
        root_world_transform = root_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    else:
        root_world_transform = Gf.Matrix4d(1.0)  # Identity
    
    # Transform from prim local to root local = root_world^-1 * prim_world
    root_to_world_inv = root_world_transform.GetInverse()
    prim_to_root = root_to_world_inv * prim_world_transform
    
    # Extract translation and rotation
    translation = prim_to_root.ExtractTranslation()
    rotation = prim_to_root.ExtractRotation()
    quat = rotation.GetQuat()
    
    # Return as numpy arrays: translation (3,), quaternion (w, x, y, z)
    trans_np = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
    quat_np = np.array([quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]], dtype=np.float32)
    
    return trans_np, quat_np


def transform_points_local_to_root(points, trans, quat):
    """
    Transform points from a local frame to the root frame.
    points: (N, 3) numpy array
    trans: (3,) translation vector
    quat: (4,) quaternion (w, x, y, z)
    Returns: (N, 3) transformed points
    """
    # Build rotation matrix from quaternion (w, x, y, z)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    
    # Transform: root_point = R * local_point + trans
    points_transformed = (R @ points.T).T + trans
    return points_transformed


def sample_cube_points(cube_prim, num_samples, rng=None):
    """
    Sample points uniformly on the surface of a UsdGeom.Cube.
    
    Args:
        cube_prim: Usd.Prim that is a UsdGeom.Cube
        num_samples: Number of points to sample
        rng: numpy RandomState (optional)
    
    Returns:
        numpy array of shape (num_samples, 3) with points in local coordinates
    """
    if rng is None:
        rng = np.random.RandomState(None)
    
    cube_geom = UsdGeom.Cube(cube_prim)
    size_attr = cube_geom.GetSizeAttr()
    
    if size_attr is None:
        # Try getting extent instead
        extent_attr = cube_geom.GetExtentAttr()
        if extent_attr is None:
            return np.zeros((num_samples, 3), dtype=np.float32)
        extent = extent_attr.Get()
        size = np.array(extent[1]) - np.array(extent[0])
    else:
        size_val = size_attr.Get()
        size = np.array([size_val, size_val, size_val], dtype=np.float32)
    
    # Cube has 6 faces, each with area = size[i] * size[j] for i != j
    # Face areas: xy (2), xz (2), yz (2)
    face_areas = [
        size[0] * size[1],  # +z face
        size[0] * size[1],  # -z face
        size[0] * size[2],  # +y face
        size[0] * size[2],  # -y face
        size[1] * size[2],  # +x face
        size[1] * size[2],  # -x face
    ]
    total_area = sum(face_areas)
    
    if total_area < 1e-10:
        return np.zeros((num_samples, 3), dtype=np.float32)
    
    # Sample points on each face proportionally to its area
    face_probs = np.array(face_areas) / total_area
    samples_per_face = rng.multinomial(num_samples, face_probs)
    
    points = []
    half_size = size / 2.0
    
    # Face 0: +z face (z = +half_size[2])
    for _ in range(samples_per_face[0]):
        x = rng.uniform(-half_size[0], half_size[0])
        y = rng.uniform(-half_size[1], half_size[1])
        z = half_size[2]
        points.append([x, y, z])
    
    # Face 1: -z face (z = -half_size[2])
    for _ in range(samples_per_face[1]):
        x = rng.uniform(-half_size[0], half_size[0])
        y = rng.uniform(-half_size[1], half_size[1])
        z = -half_size[2]
        points.append([x, y, z])
    
    # Face 2: +y face (y = +half_size[1])
    for _ in range(samples_per_face[2]):
        x = rng.uniform(-half_size[0], half_size[0])
        z = rng.uniform(-half_size[2], half_size[2])
        y = half_size[1]
        points.append([x, y, z])
    
    # Face 3: -y face (y = -half_size[1])
    for _ in range(samples_per_face[3]):
        x = rng.uniform(-half_size[0], half_size[0])
        z = rng.uniform(-half_size[2], half_size[2])
        y = -half_size[1]
        points.append([x, y, z])
    
    # Face 4: +x face (x = +half_size[0])
    for _ in range(samples_per_face[4]):
        y = rng.uniform(-half_size[1], half_size[1])
        z = rng.uniform(-half_size[2], half_size[2])
        x = half_size[0]
        points.append([x, y, z])
    
    # Face 5: -x face (x = -half_size[0])
    for _ in range(samples_per_face[5]):
        y = rng.uniform(-half_size[1], half_size[1])
        z = rng.uniform(-half_size[2], half_size[2])
        x = -half_size[0]
        points.append([x, y, z])
    
    return np.array(points, dtype=np.float32)


def sample_cylinder_points(cylinder_prim, num_samples, rng=None):
    """
    Sample points uniformly on the surface of a UsdGeom.Cylinder.
    
    Args:
        cylinder_prim: Usd.Prim that is a UsdGeom.Cylinder
        num_samples: Number of points to sample
        rng: numpy RandomState (optional)
    
    Returns:
        numpy array of shape (num_samples, 3) with points in local coordinates
    """
    if rng is None:
        rng = np.random.RandomState(None)
    
    cylinder_geom = UsdGeom.Cylinder(cylinder_prim)
    radius_attr = cylinder_geom.GetRadiusAttr()
    height_attr = cylinder_geom.GetHeightAttr()
    
    if radius_attr is None or height_attr is None:
        return np.zeros((num_samples, 3), dtype=np.float32)
    
    radius = radius_attr.Get()
    height = height_attr.Get()
    
    # Cylinder has:
    # - Curved surface area: 2 * pi * radius * height
    # - Two circular caps, each with area: pi * radius^2
    curved_area = 2.0 * math.pi * radius * height
    cap_area = math.pi * radius * radius
    total_area = curved_area + 2.0 * cap_area
    
    if total_area < 1e-10:
        return np.zeros((num_samples, 3), dtype=np.float32)
    
    # Sample proportionally to area
    curved_prob = curved_area / total_area
    cap_prob = cap_area / total_area
    
    # Sample number of points for each surface
    curved_samples = rng.binomial(num_samples, curved_prob)
    cap_samples = (num_samples - curved_samples) // 2
    cap_samples_2 = num_samples - curved_samples - cap_samples
    
    points = []
    half_height = height / 2.0
    
    # Curved surface: sample uniformly in z and angle
    for _ in range(curved_samples):
        z = rng.uniform(-half_height, half_height)
        theta = rng.uniform(0, 2 * math.pi)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        points.append([x, y, z])
    
    # Top cap (+z)
    for _ in range(cap_samples):
        # Uniform sampling on disk using rejection sampling
        while True:
            x = rng.uniform(-radius, radius)
            y = rng.uniform(-radius, radius)
            if x*x + y*y <= radius*radius:
                break
        z = half_height
        points.append([x, y, z])
    
    # Bottom cap (-z)
    for _ in range(cap_samples_2):
        # Uniform sampling on disk
        while True:
            x = rng.uniform(-radius, radius)
            y = rng.uniform(-radius, radius)
            if x*x + y*y <= radius*radius:
                break
        z = -half_height
        points.append([x, y, z])
    
    return np.array(points, dtype=np.float32)


class MeshSampler:
    """
    Handles sampling for both Rigid Objects (Screwdriver) and Articulations (Robot).
    """
    
    def __init__(self, device="cuda:0"):
        self.device = device
        self.static_points = None # For rigid objects (single env, deprecated)
        self.rigid_points_per_env = {}  # For rigid objects per env: {env_idx: tensor_of_points}
        self.link_points = {}     # For articulations: {body_index: tensor_of_points}
        self.stage = omni.usd.get_context().get_stage()

    def sample_rigid_object(self, prim_path, num_samples=1024, env_idx=None):
        """
        Sample rigid object (screwdriver).
        If env_idx is provided, stores per-environment point cloud.
        Otherwise, uses legacy single-point-cloud mode.
        """
        print(f"[MeshSampler] Sampling rigid object at: {prim_path} (env_idx={env_idx})")
        
        # Find all meshes under this prim (handling references/instances)
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[MeshSampler] ERROR: Prim {prim_path} is not valid")
            if env_idx is not None:
                self.rigid_points_per_env[env_idx] = torch.zeros((num_samples, 3), device=self.device)
            else:
                self.static_points = torch.zeros((num_samples, 3), device=self.device)
            return
        
        # Collect all mesh prims (following references and instances)
        mesh_prims = []
        cube_prims = []
        cylinder_prims = []
        
        for p in Usd.PrimRange(prim, predicate=Usd.TraverseInstanceProxies()):
            if p.IsA(UsdGeom.Mesh):
                mesh_prims.append(str(p.GetPath()))
            elif p.IsA(UsdGeom.Cube):
                cube_prims.append(str(p.GetPath()))
            elif p.IsA(UsdGeom.Cylinder):
                cylinder_prims.append(str(p.GetPath()))
        
        # If meshes are found, use existing mesh sampling logic (unchanged)
        if mesh_prims:
            # Continue with existing mesh sampling logic below
            pass
        elif cube_prims or cylinder_prims:
            # Fallback: sample from cubes/cylinders if no meshes found
            print(f"[MeshSampler] WARNING: No meshes found under {prim_path}, sampling from {len(cube_prims)} cube(s) and {len(cylinder_prims)} cylinder(s)")
            
            rng = np.random.RandomState(None)
            all_primitive_points = []
            primitive_areas = []
            
            # Sample from cubes
            primitive_transforms = {}  # Store transforms for primitives
            for cube_path in cube_prims:
                cube_prim = self.stage.GetPrimAtPath(cube_path)
                if cube_prim.IsValid():
                    cube_points = sample_cube_points(cube_prim, num_samples, rng)
                    
                    # Get local transform from cube prim to root prim
                    cube_trans, cube_quat = get_local_transform_to_root(cube_prim, prim)
                    primitive_transforms[cube_path] = (cube_trans, cube_quat)
                    
                    # Transform points from cube local to root local
                    cube_points_root = transform_points_local_to_root(cube_points, cube_trans, cube_quat)
                    all_primitive_points.append(cube_points_root)
                    
                    # Calculate area for weighting
                    cube_geom = UsdGeom.Cube(cube_prim)
                    size_attr = cube_geom.GetSizeAttr()
                    if size_attr is None:
                        extent_attr = cube_geom.GetExtentAttr()
                        if extent_attr is not None:
                            extent = extent_attr.Get()
                            size = np.array(extent[1]) - np.array(extent[0])
                        else:
                            size = np.ones(3, dtype=np.float32)
                    else:
                        size_val = size_attr.Get()
                        size = np.array([size_val, size_val, size_val], dtype=np.float32)
                    area = 2.0 * (size[0]*size[1] + size[0]*size[2] + size[1]*size[2])
                    primitive_areas.append(area)
            
            # Sample from cylinders
            for cylinder_path in cylinder_prims:
                cylinder_prim = self.stage.GetPrimAtPath(cylinder_path)
                if cylinder_prim.IsValid():
                    cylinder_points = sample_cylinder_points(cylinder_prim, num_samples, rng)
                    
                    # Get local transform from cylinder prim to root prim
                    cylinder_trans, cylinder_quat = get_local_transform_to_root(cylinder_prim, prim)
                    primitive_transforms[cylinder_path] = (cylinder_trans, cylinder_quat)
                    
                    # Transform points from cylinder local to root local
                    cylinder_points_root = transform_points_local_to_root(cylinder_points, cylinder_trans, cylinder_quat)
                    all_primitive_points.append(cylinder_points_root)
                    
                    # Calculate area for weighting
                    cylinder_geom = UsdGeom.Cylinder(cylinder_prim)
                    radius_attr = cylinder_geom.GetRadiusAttr()
                    height_attr = cylinder_geom.GetHeightAttr()
                    if radius_attr is not None and height_attr is not None:
                        radius = radius_attr.Get()
                        height = height_attr.Get()
                        area = 2.0 * math.pi * radius * height + 2.0 * math.pi * radius * radius
                        primitive_areas.append(area)
                    else:
                        primitive_areas.append(1.0)  # Default area
            
            if not all_primitive_points:
                print(f"[MeshSampler] ERROR: Failed to sample from primitives")
                if env_idx is not None:
                    self.rigid_points_per_env[env_idx] = torch.zeros((num_samples, 3), device=self.device)
                else:
                    self.static_points = torch.zeros((num_samples, 3), device=self.device)
                return
            
            # Combine points from all primitives with area-weighted sampling
            if len(all_primitive_points) == 1:
                # Single primitive, use all its points
                sampled_points = all_primitive_points[0]
            else:
                # Multiple primitives: sample proportionally to area
                primitive_areas = np.array(primitive_areas, dtype=np.float32)
                total_area = primitive_areas.sum()
                if total_area < 1e-10:
                    # Equal weighting if areas are invalid
                    primitive_probs = np.ones(len(all_primitive_points)) / len(all_primitive_points)
                else:
                    primitive_probs = primitive_areas / total_area
                
                # Distribute samples across primitives
                samples_per_primitive = rng.multinomial(num_samples, primitive_probs)
                sampled_points = []
                
                for i, prim_points in enumerate(all_primitive_points):
                    n_samples = samples_per_primitive[i]
                    if n_samples > 0:
                        if n_samples <= len(prim_points):
                            selected_indices = rng.choice(len(prim_points), size=n_samples, replace=False)
                        else:
                            selected_indices = rng.choice(len(prim_points), size=n_samples, replace=True)
                        sampled_points.extend(prim_points[selected_indices])
                
                sampled_points = np.array(sampled_points, dtype=np.float32)
            
            # Convert to tensor and store
            points_tensor = torch.from_numpy(sampled_points).to(self.device)
            
            if env_idx is not None:
                self.rigid_points_per_env[env_idx] = points_tensor
            else:
                self.static_points = points_tensor
            
            print(f"[MeshSampler] Sampled {len(sampled_points)} points from primitives (env_idx={env_idx})")
            return
        else:
            print(f"[MeshSampler] WARNING: No geometry prims (Mesh/Cube/Cylinder) found under {prim_path}, trying direct sampling")
            # Fallback to direct sampling (assumes mesh)
            if env_idx is not None:
                self.rigid_points_per_env[env_idx] = sample_mesh_points(prim_path, num_samples, self.device)
            else:
                self.static_points = sample_mesh_points(prim_path, num_samples, self.device)
            return
        
        print(f"[MeshSampler] Found {len(mesh_prims)} mesh(es) under {prim_path}")
        
        # Sample all meshes together with area-weighted sampling across ALL triangles
        # This ensures uniform distribution across the entire object, not per-mesh
        all_triangles = []
        mesh_transforms = {}  # Store local-to-root transforms for each mesh
        
        # Collect all triangles from all meshes with their areas
        for mesh_path in mesh_prims:
            mesh_prim = self.stage.GetPrimAtPath(mesh_path)
            if not mesh_prim.IsValid():
                continue
            
            # Get local transform from mesh prim to root prim
            mesh_trans, mesh_quat = get_local_transform_to_root(mesh_prim, prim)
            mesh_transforms[mesh_path] = (mesh_trans, mesh_quat)
                
            mesh_geom = UsdGeom.Mesh(mesh_prim)
            points_attr = mesh_geom.GetPointsAttr().Get()
            indices_attr = mesh_geom.GetFaceVertexIndicesAttr().Get()
            
            if points_attr is None:
                continue
            
            points_np = np.array(points_attr, dtype=np.float32)
            indices_np = np.array(indices_attr, dtype=int)
            num_faces = len(indices_np) // 3
            
            for i in range(num_faces):
                idx = i * 3
                v0 = points_np[indices_np[idx]]
                v1 = points_np[indices_np[idx + 1]]
                v2 = points_np[indices_np[idx + 2]]
                
                # Compute triangle area
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                
                all_triangles.append({
                    'mesh_path': mesh_path,
                    'tri_idx': i,
                    'v0': v0,
                    'v1': v1,
                    'v2': v2,
                    'area': area
                })
        
        if not all_triangles:
            print(f"[MeshSampler] ERROR: No triangles found in meshes")
            if env_idx is not None:
                self.rigid_points_per_env[env_idx] = torch.zeros((num_samples, 3), device=self.device)
            else:
                self.static_points = torch.zeros((num_samples, 3), device=self.device)
            return
        
        # Area-weighted sampling across ALL triangles
        triangle_areas = np.array([t['area'] for t in all_triangles], dtype=np.float32)
        total_area = triangle_areas.sum()
        
        if total_area < 1e-10:
            print(f"[MeshSampler] ERROR: Total mesh area is too small")
            if env_idx is not None:
                self.rigid_points_per_env[env_idx] = torch.zeros((num_samples, 3), device=self.device)
            else:
                self.static_points = torch.zeros((num_samples, 3), device=self.device)
            return
        
        triangle_probs = triangle_areas / total_area
        rng = np.random.RandomState(None)
        selected_indices = rng.choice(len(all_triangles), size=num_samples, p=triangle_probs)
        
        # Sample points uniformly on selected triangles
        sampled_points = []
        for idx in selected_indices:
            tri = all_triangles[idx]
            v0, v1, v2 = tri['v0'], tri['v1'], tri['v2']
            
            # Uniform sampling on triangle using barycentric coordinates
            r1, r2 = rng.rand(2)
            sqrt_r1 = np.sqrt(r1)
            u = 1.0 - sqrt_r1
            v = r2 * sqrt_r1
            w = 1.0 - u - v
            
            point = u * v0 + v * v1 + w * v2
            sampled_points.append(point)
        
        sampled_points = np.array(sampled_points, dtype=np.float32)
        
        # Transform points from mesh local frames to root local frame
        # Group points by mesh and transform them
        points_by_mesh = {}
        for idx, tri_idx in enumerate(selected_indices):
            tri = all_triangles[tri_idx]
            mesh_path = tri['mesh_path']
            if mesh_path not in points_by_mesh:
                points_by_mesh[mesh_path] = []
            points_by_mesh[mesh_path].append(sampled_points[idx])
        
        # Transform each mesh's points to root frame
        transformed_points = []
        for mesh_path, mesh_pts in points_by_mesh.items():
            if mesh_path in mesh_transforms:
                mesh_trans, mesh_quat = mesh_transforms[mesh_path]
                mesh_pts_array = np.array(mesh_pts, dtype=np.float32)
                # Transform from mesh local to root local
                root_frame_pts = transform_points_local_to_root(mesh_pts_array, mesh_trans, mesh_quat)
                transformed_points.extend(root_frame_pts)
            else:
                # No transform available, use points as-is
                transformed_points.extend(mesh_pts)
        
        sampled_points = np.array(transformed_points, dtype=np.float32)
        points_tensor = torch.from_numpy(sampled_points).to(self.device)
        
        # Check if all points are the same (clustered at one location)
        print(f"[MeshSampler] Checking screwdriver point distribution for env_idx={env_idx}, num_points={len(sampled_points)}")
        
        if len(sampled_points) > 1:
            # Round to 6 decimal places to check for near-duplicates
            pts_rounded = np.round(sampled_points, decimals=6)
            unique_pts = np.unique(pts_rounded, axis=0)
            num_unique = len(unique_pts)
            
            if num_unique == 1:
                print(f"[MeshSampler] WARNING: All {len(sampled_points)} screwdriver points are identical at location {unique_pts[0]} (env_idx={env_idx})")
            else:
                # Check bounding box size
                bounds_min = np.min(unique_pts, axis=0)
                bounds_max = np.max(unique_pts, axis=0)
                bounds_size = bounds_max - bounds_min
                max_dim = np.max(bounds_size)
                
                if max_dim < 0.001:  # Less than 1mm spread
                    print(f"[MeshSampler] WARNING: Screwdriver points are very clustered! Max dimension: {max_dim*1000:.3f}mm (env_idx={env_idx})")
                elif max_dim < 0.01:  # Less than 1cm spread
                    print(f"[MeshSampler] WARNING: Screwdriver points are clustered! Max dimension: {max_dim*1000:.3f}mm (env_idx={env_idx})")
                else:
                    print(f"[MeshSampler] Screwdriver points OK: {num_unique} unique points, spread: {max_dim*1000:.1f}mm (env_idx={env_idx})")
        elif len(sampled_points) == 1:
            print(f"[MeshSampler] Only 1 screwdriver point sampled (env_idx={env_idx})")
        else:
            print(f"[MeshSampler] ERROR: No screwdriver points sampled (env_idx={env_idx})")
        
        if env_idx is not None:
            self.rigid_points_per_env[env_idx] = points_tensor
        else:
            self.static_points = points_tensor

    def sample_articulation(self, robot_prim_path, robot_object, num_samples_per_env=512, hand_only=True):
        """
        Sampling for the Robot.
        1. Finds all visual meshes.
        2. Maps them to the correct Physics Body Index.
        3. Distributes samples across links (hand links only if hand_only=True).
        
        Args:
            robot_prim_path: Path to robot prim
            robot_object: Isaac Lab Articulation object
            num_samples_per_env: Number of points per environment (default: 512 for hand)
            hand_only: If True, only sample hand links (fingers, palm, etc.)
        """
        print(f"[MeshSampler] Sampling articulation at: {robot_prim_path}")
        print(f"[MeshSampler] Target: {num_samples_per_env} points per env ({'hand only' if hand_only else 'all links'})")
        self.link_points = {}
        
        # Get body names from the Isaac Lab Articulation view
        body_names = robot_object.data.body_names
        
        # Identify hand links by name patterns
        hand_link_indices = set()
        if hand_only:
            hand_keywords = ['finger', 'hand', 'palm', 'thumb', 'index', 'middle', 'ring', 'pinky', 
                           'hitosashi', 'naka', 'kusuri', 'oya', 'allegro']
            for i, body_name in enumerate(body_names):
                body_name_lower = body_name.lower()
                if any(keyword in body_name_lower for keyword in hand_keywords):
                    hand_link_indices.add(i)
            print(f"[MeshSampler] Identified {len(hand_link_indices)} hand links: {[body_names[i] for i in sorted(hand_link_indices)]}")
        
        # Helper: Find which physics body a visual mesh belongs to
        def find_parent_body_index(curr_prim):
            # Walk up the tree until we find a RigidBody or hit root
            checked_paths = []
            while curr_prim.GetPath() != "/":
                checked_paths.append(str(curr_prim.GetPath()))
                if curr_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    name = curr_prim.GetName()
                    # Try exact match first
                    if name in body_names:
                        return body_names.index(name)
                    # Try matching by path component (more robust)
                    prim_path_str = str(curr_prim.GetPath())
                    for i, body_name in enumerate(body_names):
                        if body_name in prim_path_str or prim_path_str.endswith(body_name):
                            return i
                curr_prim = curr_prim.GetParent()
            print(f"[MeshSampler] WARNING: Could not find body for mesh. Checked paths: {checked_paths}")
            return 0 # Default to base if fails

        # 1. Collect all visual meshes
        mesh_candidates = []
        prim = self.stage.GetPrimAtPath(robot_prim_path)
        
        if not prim.IsValid():
            print(f"[MeshSampler] ERROR: Prim {robot_prim_path} is not valid")
            return
        
        print(f"[MeshSampler] Body names: {body_names}")
        print(f"[MeshSampler] Traversing prim: {robot_prim_path}")
        
        # Use TraverseInstanceProxies to follow instances and references
        mesh_count = 0
        for p in Usd.PrimRange(prim, predicate=Usd.TraverseInstanceProxies()):
            # Check if this prim itself is a mesh
            if p.IsA(UsdGeom.Mesh):
                mesh_count += 1
                b_idx = find_parent_body_index(p)
                
                # Filter: only include hand links if hand_only=True
                if hand_only and b_idx not in hand_link_indices:
                    continue
                
                mesh_path = str(p.GetPath())
                print(f"[MeshSampler] Found mesh: {mesh_path}, body_idx: {b_idx}, body_name: {body_names[b_idx] if b_idx < len(body_names) else 'unknown'}")
                mesh_candidates.append({"path": mesh_path, "body_idx": b_idx})
        
        print(f"[MeshSampler] Total meshes found: {mesh_count}, filtered to {len(mesh_candidates)} hand meshes")

        if not mesh_candidates:
            print(f"[MeshSampler] WARNING: No meshes found under {robot_prim_path} ({'hand links only' if hand_only else 'all links'})")
            return

        # 2. Sample points per link with area-weighted distribution
        # Collect all triangles from all hand meshes for area-weighted sampling
        all_triangles = []
        
        for mesh_idx, m in enumerate(mesh_candidates):
            mesh_prim = self.stage.GetPrimAtPath(m['path'])
            if not mesh_prim.IsValid():
                continue
                
            mesh_geom = UsdGeom.Mesh(mesh_prim)
            points_attr = mesh_geom.GetPointsAttr().Get()
            indices_attr = mesh_geom.GetFaceVertexIndicesAttr().Get()
            
            if points_attr is None:
                continue
            
            points_np = np.array(points_attr, dtype=np.float32)
            indices_np = np.array(indices_attr, dtype=int)
            num_faces = len(indices_np) // 3
            
            for i in range(num_faces):
                idx = i * 3
                v0 = points_np[indices_np[idx]]
                v1 = points_np[indices_np[idx + 1]]
                v2 = points_np[indices_np[idx + 2]]
                
                # Compute triangle area
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                
                all_triangles.append({
                    'mesh_path': m['path'],
                    'body_idx': m['body_idx'],
                    'tri_idx': i,
                    'v0': v0,
                    'v1': v1,
                    'v2': v2,
                    'area': area
                })
        
        if not all_triangles:
            print(f"[MeshSampler] ERROR: No triangles found in hand meshes")
            return
        
        # Area-weighted sampling across ALL hand triangles to get exactly num_samples_per_env points
        triangle_areas = np.array([t['area'] for t in all_triangles], dtype=np.float32)
        total_area = triangle_areas.sum()
        
        if total_area < 1e-10:
            print(f"[MeshSampler] ERROR: Total hand mesh area is too small")
            return
        
        triangle_probs = triangle_areas / total_area
        rng = np.random.RandomState(None)
        selected_indices = rng.choice(len(all_triangles), size=num_samples_per_env, p=triangle_probs)
        
        # Sample points and group by body index
        points_per_link = {}
        
        for idx in selected_indices:
            tri = all_triangles[idx]
            v0, v1, v2 = tri['v0'], tri['v1'], tri['v2']
            b_idx = tri['body_idx']
            
            # Uniform sampling on triangle using barycentric coordinates
            r1, r2 = rng.rand(2)
            sqrt_r1 = np.sqrt(r1)
            u = 1.0 - sqrt_r1
            v = r2 * sqrt_r1
            w = 1.0 - u - v
            
            point = u * v0 + v * v1 + w * v2
            
            if b_idx not in points_per_link:
                points_per_link[b_idx] = []
            points_per_link[b_idx].append(point)
        
        # Convert to tensors and check for duplicates
        print(f"[MeshSampler] Converting {len(points_per_link)} links to tensors...")
        for b_idx, pts_list in points_per_link.items():
            pts_array = np.array(pts_list, dtype=np.float32)
            link_name = body_names[b_idx] if b_idx < len(body_names) else 'unknown'
            
            print(f"[MeshSampler] Processing Link {b_idx} ({link_name}): {len(pts_array)} points")
            
            # Check for duplicate points (within a small tolerance)
            if len(pts_array) == 0:
                print(f"[MeshSampler]   ERROR: No points in array!")
                continue
                
            # Round to 6 decimal places to check for near-duplicates
            pts_rounded = np.round(pts_array, decimals=6)
            unique_pts, counts = np.unique(pts_rounded, axis=0, return_counts=True)
            num_unique = len(unique_pts)
            num_duplicates = len(pts_array) - num_unique
            
            print(f"[MeshSampler]   {len(pts_array)} total points, {num_unique} unique points")
            
            if num_duplicates > 0:
                print(f"[MeshSampler]   WARNING: {num_duplicates} duplicate points detected!")
                print(f"[MeshSampler]   Duplicate distribution: {np.bincount(counts)[1:]}")  # Show how many points have duplicates
            
            # Check point spacing
            if num_unique > 1:
                try:
                    # Compute pairwise distances
                    distances = pdist(unique_pts)
                    min_dist = np.min(distances)
                    mean_dist = np.mean(distances)
                    max_dist = np.max(distances)
                    print(f"[MeshSampler]   Distance stats - min={min_dist:.6f}m, mean={mean_dist:.6f}m, max={max_dist:.6f}m")
                    
                    # Check bounds to see if points are spread out
                    bounds_min = np.min(unique_pts, axis=0)
                    bounds_max = np.max(unique_pts, axis=0)
                    bounds_size = bounds_max - bounds_min
                    print(f"[MeshSampler]   Bounds - min=[{bounds_min[0]:.6f}, {bounds_min[1]:.6f}, {bounds_min[2]:.6f}], max=[{bounds_max[0]:.6f}, {bounds_max[1]:.6f}, {bounds_max[2]:.6f}], size=[{bounds_size[0]:.6f}, {bounds_size[1]:.6f}, {bounds_size[2]:.6f}]")
                    
                    # Check if points are clustered (very small bounding box)
                    if np.any(bounds_size < 0.001):
                        print(f"[MeshSampler]   WARNING: Points are very clustered! One dimension is < 1mm")
                    # Check if points are clustered (very small bounding box)
                    if np.any(bounds_size < 0.001):
                        print(f"[MeshSampler]   WARNING: Points are very clustered! One dimension is < 1mm")
                    if np.all(bounds_size < 0.01):
                        print(f"[MeshSampler]   WARNING: Points are clustered in a very small volume (< 1cm)")
                except Exception as e:
                    print(f"[MeshSampler]   ERROR computing distances: {e}")
            elif num_unique == 1:
                print(f"[MeshSampler]   WARNING: Only 1 unique point! All points are duplicates!")
                print(f"[MeshSampler]   Point location: {unique_pts[0]}")
            else:
                print(f"[MeshSampler]   ERROR: No unique points!")
            
            # Shape: (1, N_link, 3) - Prepared for broadcasting
            self.link_points[b_idx] = torch.from_numpy(pts_array).unsqueeze(0).to(self.device)
        
        total_points = sum(pts.shape[1] for pts in self.link_points.values())
        print(f"[MeshSampler] Sampled exactly {total_points} points across {len(self.link_points)} hand links")

    def get_rigid_pcd(self, pos_w, quat_w, env_origins):
        """
        For Screwdriver.
        Uses per-environment point clouds if available, otherwise falls back to static_points.
        """
        num_envs = pos_w.shape[0]
        
        # Check if we have per-environment point clouds
        if self.rigid_points_per_env:
            # Use per-environment point clouds
            all_env_points = []
            for env_idx in range(num_envs):
                if env_idx in self.rigid_points_per_env:
                    local_pts = self.rigid_points_per_env[env_idx].unsqueeze(0)  # (1, P, 3)
                    env_pos = pos_w[env_idx:env_idx+1]  # (1, 3)
                    env_quat = quat_w[env_idx:env_idx+1]  # (1, 4)
                    transformed = transform_points(local_pts, env_pos, env_quat)  # (1, P, 3)
                    all_env_points.append(transformed)
                else:
                    # Fallback: use zeros if this env wasn't sampled
                    num_pts = next(iter(self.rigid_points_per_env.values())).shape[0]
                    all_env_points.append(torch.zeros((1, num_pts, 3), device=self.device))
            
            # Stack: (num_envs, P, 3)
            pts = torch.cat(all_env_points, dim=0)
        elif self.static_points is not None:
            # Legacy mode: use single point cloud for all environments
            # (N, P, 3) = transform( (1, P, 3), (N, 3), (N, 4) )
            pts = transform_points(self.static_points.unsqueeze(0), pos_w, quat_w)
        else:
            return torch.zeros((num_envs, 0, 3), device=self.device)
        
        # Apply Env Offset if env_origins is not zero (for observations)
        # If env_origins is zero (for visualization), this does nothing
        return pts - env_origins.unsqueeze(1)

    def get_articulation_pcd(self, body_states_w, env_origins):
        """For Robot"""
        all_env_points = []
        
        for b_idx, local_pts in self.link_points.items():
            # Link state: (num_envs, 3) pos, (num_envs, 4) quat
            link_pos = body_states_w[:, b_idx, 0:3]
            link_quat = body_states_w[:, b_idx, 3:7]
            
            transformed = transform_points(local_pts, link_pos, link_quat)
            all_env_points.append(transformed)
        
        if not all_env_points:
             return torch.zeros((body_states_w.shape[0], 0, 3), device=self.device)

        full_cloud = torch.cat(all_env_points, dim=1)
        return full_cloud - env_origins.unsqueeze(1)

# --- Lower level Warp Function (Keep this as is) ---
def sample_mesh_points(prim_path: str, num_samples: int, device: str = "cuda:0") -> torch.Tensor:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    if not prim.IsValid():
        return torch.zeros((num_samples, 3), device=device)

    mesh_geom = UsdGeom.Mesh(prim)
    points_attr = mesh_geom.GetPointsAttr().Get()
    indices_attr = mesh_geom.GetFaceVertexIndicesAttr().Get()
    
    if points_attr is None:
        return torch.zeros((num_samples, 3), device=device)

    points_np = np.array(points_attr, dtype=np.float32)
    indices_np = np.array(indices_attr, dtype=int)

    # Use numpy-based sampling instead of Warp (simpler and more reliable)
    # Sample points uniformly on triangles
    num_faces = len(indices_np) // 3
    if num_faces == 0:
        return torch.zeros((num_samples, 3), device=device)
    
    # Calculate triangle areas for weighted sampling
    triangle_areas = []
    for i in range(num_faces):
        idx = i * 3
        v0 = points_np[indices_np[idx]]
        v1 = points_np[indices_np[idx + 1]]
        v2 = points_np[indices_np[idx + 2]]
        # Compute triangle area using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        triangle_areas.append(area)
    
    triangle_areas = np.array(triangle_areas, dtype=np.float32)
    total_area = triangle_areas.sum()
    if total_area < 1e-10:
        return torch.zeros((num_samples, 3), device=device)
    
    # Weighted random selection of triangles
    # Use None as seed to get different random samples each time
    rng = np.random.RandomState(None)
    triangle_probs = triangle_areas / total_area
    selected_triangles = rng.choice(num_faces, size=num_samples, p=triangle_probs)
    
    # Sample points uniformly on selected triangles
    sampled_points = []
    for tri_idx in selected_triangles:
        idx = tri_idx * 3
        v0 = points_np[indices_np[idx]]
        v1 = points_np[indices_np[idx + 1]]
        v2 = points_np[indices_np[idx + 2]]
        
        # Uniform sampling on triangle using barycentric coordinates
        r1, r2 = rng.rand(2)
        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1.0 - u - v
        
        point = u * v0 + v * v1 + w * v2
        sampled_points.append(point)
    
    sampled_points = np.array(sampled_points, dtype=np.float32)
    return torch.from_numpy(sampled_points).to(device)
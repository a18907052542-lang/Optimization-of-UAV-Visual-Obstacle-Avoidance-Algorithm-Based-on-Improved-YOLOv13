"""
Real-time Obstacle Avoidance Decision Mechanism
Implements hazard level evaluation (Equation 8) and path planning (Equation 9)
from the paper: "Optimization of UAV Visual Obstacle Avoidance Algorithm Based on 
Improved YOLOv13 in Complex Scenarios"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
from collections import deque


@dataclass
class ObstacleInfo:
    """
    Information about a detected obstacle
    
    Attributes:
        id: Unique identifier for tracking
        bbox: Bounding box [x1, y1, x2, y2] in image coordinates
        class_id: Object class index
        confidence: Detection confidence score
        distance: Estimated distance to obstacle (meters)
        velocity: Estimated relative velocity [vx, vy, vz] (m/s)
        area: Area in image (pixels^2)
        position_3d: 3D position estimate [x, y, z] (meters)
    """
    id: int
    bbox: np.ndarray
    class_id: int
    confidence: float
    distance: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    area: float = 0.0
    position_3d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        """Calculate area from bbox if not provided"""
        if self.area == 0.0 and self.bbox is not None:
            self.area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class KalmanFilter:
    """
    Kalman Filter for obstacle trajectory prediction
    Used in post-processing module for trajectory prediction
    """
    
    def __init__(self, dim_x: int = 6, dim_z: int = 3):
        """
        Initialize Kalman Filter
        
        Args:
            dim_x: State dimension (position + velocity)
            dim_z: Measurement dimension (position only)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros((dim_x, 1))
        
        # State covariance matrix
        self.P = np.eye(dim_x) * 1000.0
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(dim_x)
        dt = 1.0 / 30.0  # Assume 30 FPS
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        # Measurement noise covariance
        self.R = np.eye(dim_z) * 0.1
        
        # Process noise covariance
        q = 0.1
        self.Q = np.eye(dim_x) * q
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """
        Predict next state
        
        Returns:
            Predicted state vector
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x.flatten()[:3]
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update state with measurement
        
        Args:
            z: Measurement vector [x, y, z]
            
        Returns:
            Updated state vector
        """
        z = z.reshape((self.dim_z, 1))
        
        if not self.initialized:
            self.x[:3] = z
            self.initialized = True
            return self.x.flatten()[:3]
        
        # Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.dim_x) - np.dot(K, self.H), self.P)
        
        return self.x.flatten()[:3]
    
    def get_velocity(self) -> np.ndarray:
        """Get estimated velocity"""
        return self.x.flatten()[3:6]


class TrajectoryPredictor:
    """
    Trajectory prediction using Kalman filtering
    Maintains tracking state for multiple obstacles
    """
    
    def __init__(self, max_age: int = 30):
        """
        Initialize trajectory predictor
        
        Args:
            max_age: Maximum frames to keep track without update
        """
        self.trackers: Dict[int, KalmanFilter] = {}
        self.ages: Dict[int, int] = {}
        self.max_age = max_age
    
    def update(self, obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """
        Update trackers with new detections
        
        Args:
            obstacles: List of detected obstacles
            
        Returns:
            Updated obstacles with predicted velocities
        """
        updated_ids = set()
        
        for obs in obstacles:
            if obs.id not in self.trackers:
                self.trackers[obs.id] = KalmanFilter()
                self.ages[obs.id] = 0
            
            # Update tracker
            self.trackers[obs.id].update(obs.position_3d)
            obs.velocity = self.trackers[obs.id].get_velocity()
            updated_ids.add(obs.id)
            self.ages[obs.id] = 0
        
        # Age out old trackers
        to_remove = []
        for track_id in self.trackers:
            if track_id not in updated_ids:
                self.ages[track_id] += 1
                if self.ages[track_id] > self.max_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.trackers[track_id]
            del self.ages[track_id]
        
        return obstacles
    
    def predict_future(self, obstacle_id: int, steps: int = 10) -> np.ndarray:
        """
        Predict future trajectory for an obstacle
        
        Args:
            obstacle_id: Obstacle ID to predict
            steps: Number of future steps to predict
            
        Returns:
            Array of predicted positions shape (steps, 3)
        """
        if obstacle_id not in self.trackers:
            return np.zeros((steps, 3))
        
        tracker = self.trackers[obstacle_id]
        predictions = []
        
        # Save current state
        x_saved = tracker.x.copy()
        P_saved = tracker.P.copy()
        
        for _ in range(steps):
            pred = tracker.predict()
            predictions.append(pred)
        
        # Restore state
        tracker.x = x_saved
        tracker.P = P_saved
        
        return np.array(predictions)


class HazardEvaluator:
    """
    Hazard level evaluation for obstacles
    Implements Equation (8) from the paper:
    
    φ(o_i) = α * e^(-d_i/R) + β * |v_rel| + γ * (A_i / A_max)
    
    where:
        o_i: i-th obstacle
        d_i: relative distance
        R: distance decay factor
        v_rel: relative velocity
        A_i: obstacle area in image
        A_max: normalization constant
        α, β, γ: weight coefficients
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        distance_decay: float = 10.0,
        max_area: float = 640 * 640 * 0.5,
        max_velocity: float = 30.0
    ):
        """
        Initialize hazard evaluator
        
        Args:
            alpha: Weight for distance component (default 0.5)
            beta: Weight for velocity component (default 0.3)
            gamma: Weight for area component (default 0.2)
            distance_decay: Distance decay factor R (meters)
            max_area: Maximum obstacle area for normalization A_max
            max_velocity: Maximum relative velocity for normalization (m/s)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.R = distance_decay
        self.A_max = max_area
        self.v_max = max_velocity
        
        # Ensure weights sum to 1
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
    
    def evaluate(self, obstacle: ObstacleInfo) -> float:
        """
        Evaluate hazard level for a single obstacle
        Implements Equation (8)
        
        Args:
            obstacle: Obstacle information
            
        Returns:
            Hazard level φ(o_i) in range [0, 1]
        """
        # Distance component: α * e^(-d_i/R)
        distance_term = self.alpha * math.exp(-obstacle.distance / self.R)
        
        # Velocity component: β * |v_rel| / v_max
        velocity_magnitude = np.linalg.norm(obstacle.velocity)
        velocity_term = self.beta * min(velocity_magnitude / self.v_max, 1.0)
        
        # Area component: γ * (A_i / A_max)
        area_term = self.gamma * min(obstacle.area / self.A_max, 1.0)
        
        # Total hazard level
        hazard = distance_term + velocity_term + area_term
        
        return min(max(hazard, 0.0), 1.0)
    
    def evaluate_batch(self, obstacles: List[ObstacleInfo]) -> List[Tuple[ObstacleInfo, float]]:
        """
        Evaluate hazard levels for multiple obstacles
        
        Args:
            obstacles: List of obstacles
            
        Returns:
            List of (obstacle, hazard_level) tuples sorted by hazard (descending)
        """
        results = [(obs, self.evaluate(obs)) for obs in obstacles]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_critical_obstacles(
        self, 
        obstacles: List[ObstacleInfo], 
        threshold: float = 0.5
    ) -> List[ObstacleInfo]:
        """
        Get obstacles with hazard level above threshold
        
        Args:
            obstacles: List of obstacles
            threshold: Hazard level threshold
            
        Returns:
            List of critical obstacles
        """
        evaluated = self.evaluate_batch(obstacles)
        return [obs for obs, hazard in evaluated if hazard >= threshold]


class OccupancyGrid:
    """
    Local occupancy grid map for path planning
    Maps detected obstacles into 3D space
    """
    
    def __init__(
        self,
        resolution: float = 0.5,
        x_range: Tuple[float, float] = (-20, 20),
        y_range: Tuple[float, float] = (-20, 20),
        z_range: Tuple[float, float] = (-10, 10)
    ):
        """
        Initialize occupancy grid
        
        Args:
            resolution: Grid cell size in meters
            x_range: X-axis range (forward/backward)
            y_range: Y-axis range (left/right)
            z_range: Z-axis range (up/down)
        """
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        # Calculate grid dimensions
        self.nx = int((x_range[1] - x_range[0]) / resolution)
        self.ny = int((y_range[1] - y_range[0]) / resolution)
        self.nz = int((z_range[1] - z_range[0]) / resolution)
        
        # Initialize grid (0 = free, 1 = occupied)
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        
        # Risk grid (continuous hazard values)
        self.risk_grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
    
    def world_to_grid(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices"""
        x = int((position[0] - self.x_range[0]) / self.resolution)
        y = int((position[1] - self.y_range[0]) / self.resolution)
        z = int((position[2] - self.z_range[0]) / self.resolution)
        
        x = max(0, min(x, self.nx - 1))
        y = max(0, min(y, self.ny - 1))
        z = max(0, min(z, self.nz - 1))
        
        return x, y, z
    
    def grid_to_world(self, ix: int, iy: int, iz: int) -> np.ndarray:
        """Convert grid indices to world coordinates"""
        x = self.x_range[0] + (ix + 0.5) * self.resolution
        y = self.y_range[0] + (iy + 0.5) * self.resolution
        z = self.z_range[0] + (iz + 0.5) * self.resolution
        return np.array([x, y, z])
    
    def update(
        self, 
        obstacles: List[ObstacleInfo], 
        hazard_evaluator: HazardEvaluator
    ):
        """
        Update occupancy grid with detected obstacles
        
        Args:
            obstacles: List of detected obstacles
            hazard_evaluator: Hazard evaluator for risk calculation
        """
        # Clear grid
        self.grid.fill(0)
        self.risk_grid.fill(0)
        
        for obs in obstacles:
            if obs.position_3d is None:
                continue
            
            # Get grid position
            ix, iy, iz = self.world_to_grid(obs.position_3d)
            
            # Calculate hazard level
            hazard = hazard_evaluator.evaluate(obs)
            
            # Mark occupied cells (with some inflation)
            inflation = max(1, int(math.sqrt(obs.area) / 100 / self.resolution))
            
            for dx in range(-inflation, inflation + 1):
                for dy in range(-inflation, inflation + 1):
                    for dz in range(-inflation // 2, inflation // 2 + 1):
                        nx = ix + dx
                        ny = iy + dy
                        nz = iz + dz
                        
                        if 0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz:
                            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                            if dist <= inflation:
                                self.grid[nx, ny, nz] = 1.0
                                # Risk decreases with distance from obstacle center
                                self.risk_grid[nx, ny, nz] = max(
                                    self.risk_grid[nx, ny, nz],
                                    hazard * math.exp(-dist / 2)
                                )
    
    def is_occupied(self, position: np.ndarray) -> bool:
        """Check if a position is occupied"""
        ix, iy, iz = self.world_to_grid(position)
        return self.grid[ix, iy, iz] > 0.5
    
    def get_risk(self, position: np.ndarray) -> float:
        """Get risk level at a position"""
        ix, iy, iz = self.world_to_grid(position)
        return self.risk_grid[ix, iy, iz]


@dataclass
class Path:
    """Represents a candidate path for obstacle avoidance"""
    waypoints: np.ndarray  # Shape (N, 3)
    cost: float = float('inf')
    collision_free: bool = True


class PathPlanner:
    """
    Path planning for obstacle avoidance
    Implements Equation (9) from the paper:
    
    J(p) = w1 * Σ[R(o_i) * 1(o_i ∈ p)] + w2 * ∫|κ(s)|²ds + w3 * |p_end - p_goal|²
    
    where:
        p: candidate path
        1(): indicator function
        κ(s): path curvature
        p_end: path endpoint
        p_goal: target position
        w1, w2, w3: weight parameters
    """
    
    def __init__(
        self,
        w1: float = 10.0,
        w2: float = 1.0,
        w3: float = 5.0,
        num_candidates: int = 20,
        path_length: float = 10.0,
        num_waypoints: int = 20
    ):
        """
        Initialize path planner
        
        Args:
            w1: Weight for obstacle collision cost
            w2: Weight for path smoothness (curvature) cost
            w3: Weight for goal distance cost
            num_candidates: Number of candidate paths to generate
            path_length: Length of candidate paths (meters)
            num_waypoints: Number of waypoints per path
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.num_candidates = num_candidates
        self.path_length = path_length
        self.num_waypoints = num_waypoints
    
    def compute_curvature(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Compute curvature along path
        
        Args:
            waypoints: Path waypoints shape (N, 3)
            
        Returns:
            Curvature values shape (N-2,)
        """
        if len(waypoints) < 3:
            return np.zeros(max(len(waypoints) - 2, 0))
        
        curvatures = []
        for i in range(1, len(waypoints) - 1):
            p1 = waypoints[i - 1]
            p2 = waypoints[i]
            p3 = waypoints[i + 1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product magnitude / product of lengths
            cross = np.cross(v1, v2)
            cross_mag = np.linalg.norm(cross)
            
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 1e-6 and len2 > 1e-6:
                # Curvature approximation
                curvature = 2 * cross_mag / (len1 * len2 * (len1 + len2))
            else:
                curvature = 0.0
            
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def compute_path_cost(
        self,
        path: Path,
        occupancy_grid: OccupancyGrid,
        goal: np.ndarray
    ) -> float:
        """
        Compute path cost using Equation (9)
        
        J(p) = w1 * Σ[R(o_i) * 1(o_i ∈ p)] + w2 * ∫|κ(s)|²ds + w3 * |p_end - p_goal|²
        
        Args:
            path: Candidate path
            occupancy_grid: Occupancy grid with risk values
            goal: Goal position
            
        Returns:
            Total path cost
        """
        waypoints = path.waypoints
        
        # Term 1: Collision/Risk cost - w1 * Σ[R(o_i) * 1(o_i ∈ p)]
        collision_cost = 0.0
        collision_free = True
        
        for point in waypoints:
            if occupancy_grid.is_occupied(point):
                collision_free = False
                collision_cost += 100.0  # High penalty for collision
            collision_cost += occupancy_grid.get_risk(point)
        
        path.collision_free = collision_free
        
        # Term 2: Smoothness cost - w2 * ∫|κ(s)|²ds
        curvatures = self.compute_curvature(waypoints)
        smoothness_cost = np.sum(curvatures ** 2)
        
        # Term 3: Goal distance cost - w3 * |p_end - p_goal|²
        endpoint = waypoints[-1]
        goal_distance = np.linalg.norm(endpoint - goal)
        goal_cost = goal_distance ** 2
        
        # Total cost (Equation 9)
        total_cost = (
            self.w1 * collision_cost +
            self.w2 * smoothness_cost +
            self.w3 * goal_cost
        )
        
        path.cost = total_cost
        return total_cost
    
    def generate_candidate_paths(
        self,
        start: np.ndarray,
        heading: np.ndarray,
        goal: np.ndarray
    ) -> List[Path]:
        """
        Generate candidate paths for evaluation
        
        Args:
            start: Starting position
            heading: Current heading direction (unit vector)
            goal: Goal position
            
        Returns:
            List of candidate paths
        """
        candidates = []
        
        # Normalize heading
        heading = heading / (np.linalg.norm(heading) + 1e-6)
        
        # Generate paths with different turning angles
        angles_horizontal = np.linspace(-np.pi/3, np.pi/3, self.num_candidates)
        angles_vertical = np.linspace(-np.pi/6, np.pi/6, 5)
        
        for h_angle in angles_horizontal:
            for v_angle in angles_vertical:
                waypoints = self._generate_smooth_path(
                    start, heading, h_angle, v_angle, 
                    self.path_length, self.num_waypoints
                )
                candidates.append(Path(waypoints=waypoints))
        
        # Add direct path to goal
        direct_waypoints = self._generate_direct_path(start, goal, self.num_waypoints)
        candidates.append(Path(waypoints=direct_waypoints))
        
        return candidates
    
    def _generate_smooth_path(
        self,
        start: np.ndarray,
        heading: np.ndarray,
        h_angle: float,
        v_angle: float,
        length: float,
        num_points: int
    ) -> np.ndarray:
        """Generate a smooth curved path"""
        waypoints = [start.copy()]
        
        # Rotate heading
        cos_h, sin_h = math.cos(h_angle), math.sin(h_angle)
        cos_v, sin_v = math.cos(v_angle), math.sin(v_angle)
        
        # Create rotation matrices
        # Horizontal rotation (yaw)
        rotated_heading = np.array([
            heading[0] * cos_h - heading[1] * sin_h,
            heading[0] * sin_h + heading[1] * cos_h,
            heading[2]
        ])
        
        # Vertical rotation (pitch)
        horizontal_mag = math.sqrt(rotated_heading[0]**2 + rotated_heading[1]**2)
        rotated_heading = np.array([
            rotated_heading[0] * cos_v,
            rotated_heading[1] * cos_v,
            horizontal_mag * sin_v + rotated_heading[2] * cos_v
        ])
        
        # Normalize
        rotated_heading = rotated_heading / (np.linalg.norm(rotated_heading) + 1e-6)
        
        # Generate waypoints with smooth transition
        step = length / (num_points - 1)
        current_heading = heading.copy()
        
        for i in range(1, num_points):
            # Gradually change heading
            t = i / (num_points - 1)
            current_heading = (1 - t) * heading + t * rotated_heading
            current_heading = current_heading / (np.linalg.norm(current_heading) + 1e-6)
            
            new_point = waypoints[-1] + step * current_heading
            waypoints.append(new_point)
        
        return np.array(waypoints)
    
    def _generate_direct_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """Generate direct path to goal"""
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = start + t * (goal - start)
            waypoints.append(point)
        return np.array(waypoints)
    
    def plan(
        self,
        start: np.ndarray,
        heading: np.ndarray,
        goal: np.ndarray,
        occupancy_grid: OccupancyGrid
    ) -> Path:
        """
        Plan optimal path using gradient descent optimization
        
        Args:
            start: Current position
            heading: Current heading direction
            goal: Target position
            occupancy_grid: Occupancy grid map
            
        Returns:
            Best path
        """
        # Generate candidate paths
        candidates = self.generate_candidate_paths(start, heading, goal)
        
        # Evaluate all candidates
        for path in candidates:
            self.compute_path_cost(path, occupancy_grid, goal)
        
        # Select best collision-free path
        collision_free_paths = [p for p in candidates if p.collision_free]
        
        if collision_free_paths:
            best_path = min(collision_free_paths, key=lambda p: p.cost)
        else:
            # If no collision-free path, select path with lowest cost
            best_path = min(candidates, key=lambda p: p.cost)
        
        return best_path
    
    def optimize_path(
        self,
        path: Path,
        occupancy_grid: OccupancyGrid,
        goal: np.ndarray,
        iterations: int = 50,
        learning_rate: float = 0.1
    ) -> Path:
        """
        Optimize path using gradient descent
        
        Args:
            path: Initial path
            occupancy_grid: Occupancy grid map
            goal: Target position
            iterations: Number of optimization iterations
            learning_rate: Step size for gradient descent
            
        Returns:
            Optimized path
        """
        waypoints = path.waypoints.copy()
        
        for _ in range(iterations):
            gradients = np.zeros_like(waypoints)
            
            # Compute gradients for interior points
            for i in range(1, len(waypoints) - 1):
                # Risk gradient (move away from obstacles)
                risk = occupancy_grid.get_risk(waypoints[i])
                if risk > 0:
                    # Numerical gradient of risk
                    eps = 0.1
                    for d in range(3):
                        pos_plus = waypoints[i].copy()
                        pos_plus[d] += eps
                        pos_minus = waypoints[i].copy()
                        pos_minus[d] -= eps
                        
                        risk_plus = occupancy_grid.get_risk(pos_plus)
                        risk_minus = occupancy_grid.get_risk(pos_minus)
                        
                        gradients[i, d] += self.w1 * (risk_plus - risk_minus) / (2 * eps)
                
                # Smoothness gradient (minimize curvature)
                if i > 0 and i < len(waypoints) - 1:
                    smoothness_grad = (
                        2 * waypoints[i] - waypoints[i-1] - waypoints[i+1]
                    )
                    gradients[i] += self.w2 * smoothness_grad
            
            # Update waypoints
            waypoints[1:-1] -= learning_rate * gradients[1:-1]
        
        optimized_path = Path(waypoints=waypoints)
        self.compute_path_cost(optimized_path, occupancy_grid, goal)
        
        return optimized_path


class ObstacleAvoidanceSystem:
    """
    Complete obstacle avoidance system integrating detection, hazard evaluation,
    and path planning
    """
    
    def __init__(
        self,
        hazard_alpha: float = 0.5,
        hazard_beta: float = 0.3,
        hazard_gamma: float = 0.2,
        grid_resolution: float = 0.5,
        critical_threshold: float = 0.6,
        safe_distance: float = 3.0
    ):
        """
        Initialize obstacle avoidance system
        
        Args:
            hazard_alpha: Weight for distance in hazard evaluation
            hazard_beta: Weight for velocity in hazard evaluation
            hazard_gamma: Weight for area in hazard evaluation
            grid_resolution: Resolution of occupancy grid (meters)
            critical_threshold: Threshold for critical obstacle classification
            safe_distance: Minimum safe distance (meters)
        """
        self.hazard_evaluator = HazardEvaluator(
            alpha=hazard_alpha,
            beta=hazard_beta,
            gamma=hazard_gamma
        )
        
        self.occupancy_grid = OccupancyGrid(resolution=grid_resolution)
        self.path_planner = PathPlanner()
        self.trajectory_predictor = TrajectoryPredictor()
        
        self.critical_threshold = critical_threshold
        self.safe_distance = safe_distance
        
        # State
        self.current_position = np.zeros(3)
        self.current_heading = np.array([1.0, 0.0, 0.0])
        self.goal_position = np.zeros(3)
        
        # History for smoothing
        self.command_history = deque(maxlen=5)
    
    def process_detections(
        self,
        detections: List[Dict],
        depth_map: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None
    ) -> List[ObstacleInfo]:
        """
        Process raw detections into obstacle information
        
        Args:
            detections: List of detection dicts with keys: bbox, class_id, confidence
            depth_map: Depth map for distance estimation (optional)
            camera_matrix: Camera intrinsic matrix (optional)
            
        Returns:
            List of processed ObstacleInfo objects
        """
        obstacles = []
        
        for i, det in enumerate(detections):
            bbox = np.array(det['bbox'])
            
            # Calculate area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Estimate distance (from depth map or heuristic)
            if depth_map is not None:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                cx = max(0, min(cx, depth_map.shape[1] - 1))
                cy = max(0, min(cy, depth_map.shape[0] - 1))
                distance = depth_map[cy, cx]
            else:
                # Heuristic: larger area = closer
                distance = 1000.0 / (math.sqrt(area) + 10)
            
            # Estimate 3D position
            if camera_matrix is not None and depth_map is not None:
                position_3d = self._project_to_3d(bbox, distance, camera_matrix)
            else:
                # Simplified 3D position estimate
                cx = (bbox[0] + bbox[2]) / 2 - 320  # Assume 640 width
                cy = (bbox[1] + bbox[3]) / 2 - 240  # Assume 480 height
                position_3d = np.array([
                    distance,
                    -cx * distance / 500,  # Simplified focal length
                    -cy * distance / 500
                ])
            
            obstacle = ObstacleInfo(
                id=det.get('id', i),
                bbox=bbox,
                class_id=det['class_id'],
                confidence=det['confidence'],
                distance=distance,
                area=area,
                position_3d=position_3d
            )
            
            obstacles.append(obstacle)
        
        # Update trajectory predictions
        obstacles = self.trajectory_predictor.update(obstacles)
        
        return obstacles
    
    def _project_to_3d(
        self,
        bbox: np.ndarray,
        depth: float,
        camera_matrix: np.ndarray
    ) -> np.ndarray:
        """Project 2D detection to 3D coordinates"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        px = camera_matrix[0, 2]
        py = camera_matrix[1, 2]
        
        x = (cx - px) * depth / fx
        y = (cy - py) * depth / fy
        z = depth
        
        return np.array([z, -x, -y])  # Convert to UAV coordinate system
    
    def update(
        self,
        detections: List[Dict],
        current_position: np.ndarray,
        current_heading: np.ndarray,
        goal_position: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Main update function - process detections and generate avoidance command
        
        Args:
            detections: Raw detection results
            current_position: Current UAV position
            current_heading: Current UAV heading
            goal_position: Target position
            depth_map: Optional depth map
            
        Returns:
            Dictionary with avoidance command and status information
        """
        self.current_position = current_position
        self.current_heading = current_heading
        self.goal_position = goal_position
        
        # Process detections
        obstacles = self.process_detections(detections, depth_map)
        
        # Evaluate hazard levels
        hazard_results = self.hazard_evaluator.evaluate_batch(obstacles)
        
        # Get critical obstacles
        critical_obstacles = [
            obs for obs, hazard in hazard_results 
            if hazard >= self.critical_threshold
        ]
        
        # Update occupancy grid
        self.occupancy_grid.update(obstacles, self.hazard_evaluator)
        
        # Plan path
        best_path = self.path_planner.plan(
            current_position,
            current_heading,
            goal_position,
            self.occupancy_grid
        )
        
        # Optimize path
        optimized_path = self.path_planner.optimize_path(
            best_path,
            self.occupancy_grid,
            goal_position
        )
        
        # Generate control command
        command = self._generate_command(optimized_path)
        
        # Smooth command
        self.command_history.append(command)
        smoothed_command = self._smooth_command()
        
        return {
            'command': smoothed_command,
            'path': optimized_path,
            'obstacles': obstacles,
            'hazard_levels': hazard_results,
            'critical_obstacles': critical_obstacles,
            'collision_free': optimized_path.collision_free,
            'path_cost': optimized_path.cost
        }
    
    def _generate_command(self, path: Path) -> np.ndarray:
        """Generate velocity command from path"""
        if len(path.waypoints) < 2:
            return np.zeros(3)
        
        # Target is first waypoint after current position
        target = path.waypoints[1]
        direction = target - self.current_position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return np.zeros(3)
        
        # Normalize and scale by distance
        max_speed = 5.0  # m/s
        speed = min(distance, max_speed)
        
        return direction / distance * speed
    
    def _smooth_command(self) -> np.ndarray:
        """Apply exponential smoothing to commands"""
        if not self.command_history:
            return np.zeros(3)
        
        # Exponential moving average
        alpha = 0.5
        smoothed = np.zeros(3)
        weight = 1.0
        total_weight = 0.0
        
        for cmd in reversed(self.command_history):
            smoothed += weight * cmd
            total_weight += weight
            weight *= (1 - alpha)
        
        return smoothed / total_weight
    
    def emergency_stop(self) -> Dict:
        """Generate emergency stop command"""
        return {
            'command': np.zeros(3),
            'emergency': True,
            'message': 'Emergency stop activated'
        }


# Test function
def test_obstacle_avoidance():
    """Test obstacle avoidance system"""
    print("Testing Obstacle Avoidance System...")
    
    # Create system
    system = ObstacleAvoidanceSystem(
        hazard_alpha=0.5,
        hazard_beta=0.3,
        hazard_gamma=0.2
    )
    
    # Create test detections
    detections = [
        {
            'bbox': np.array([100, 100, 200, 200]),
            'class_id': 0,
            'confidence': 0.9,
            'id': 0
        },
        {
            'bbox': np.array([400, 300, 500, 400]),
            'class_id': 1,
            'confidence': 0.85,
            'id': 1
        }
    ]
    
    # Test update
    result = system.update(
        detections=detections,
        current_position=np.array([0, 0, 5]),
        current_heading=np.array([1, 0, 0]),
        goal_position=np.array([20, 0, 5])
    )
    
    print(f"  Command: {result['command']}")
    print(f"  Path cost: {result['path_cost']:.4f}")
    print(f"  Collision free: {result['collision_free']}")
    print(f"  Critical obstacles: {len(result['critical_obstacles'])}")
    
    # Test hazard evaluator
    evaluator = HazardEvaluator()
    obstacle = ObstacleInfo(
        id=0,
        bbox=np.array([100, 100, 200, 200]),
        class_id=0,
        confidence=0.9,
        distance=5.0,
        velocity=np.array([2.0, 0, 0]),
        area=10000
    )
    hazard = evaluator.evaluate(obstacle)
    print(f"  Test hazard level: {hazard:.4f}")
    
    # Test Kalman filter
    kf = KalmanFilter()
    for i in range(10):
        measurement = np.array([i * 0.1, 0, 5])
        kf.update(measurement)
        predicted = kf.predict()
    print(f"  Kalman filter velocity: {kf.get_velocity()}")
    
    print("Obstacle Avoidance System test completed!")


if __name__ == "__main__":
    test_obstacle_avoidance()

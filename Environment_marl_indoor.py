from __future__ import division
import numpy as np
import time
import random
import math
import os

from arguments import get_args
args = get_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

class A2GChannels:
    """
    Air-to-Ground (A2G) channel model for UAV networks
    Uses probabilistic LoS model with Rician fading (LoS) and Rayleigh fading (NLoS)
    """

    def __init__(self):
        # Ground Base Station (GBS) position [x, y, z] in meters
        self.GBS_position = np.array([[12.5, 12.5, 0]])  # Ground level base station
        self.fc = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  # Carrier frequencies in GHz
        
        # A2G channel parameters
        self.eta_LoS = 1.0  # Additional path loss for LoS (dB)
        self.eta_NLoS = 20.0  # Additional path loss for NLoS (dB)
        self.c = 3e8  # Speed of light (m/s)
        
        # Rician fading parameters for LoS
        self.K_rician = 10.0  # Rician K-factor (dB) for LoS links
        
        # Environment parameters for LoS probability
        self.a = 9.61  # Environment-dependent parameter
        self.b = 0.16  # Environment-dependent parameter
        self.alpha = 0.0  # Environment-dependent parameter (urban: 0.0, suburban: 0.5)
        
    def get_los_probability(self, uav_position):
        """
        Calculate LoS probability using probabilistic model
        Args:
            uav_position: [x, y, z] position of UAV
        Returns:
            LoS probability
        """
        # Calculate horizontal and vertical distances
        gbs_pos = self.GBS_position[0]
        d_2d = math.sqrt((uav_position[0] - gbs_pos[0])**2 + (uav_position[1] - gbs_pos[1])**2)
        h_uav = uav_position[2]
        h_gbs = gbs_pos[2]
        
        # Elevation angle in radians
        if d_2d > 0:
            theta = math.atan((h_uav - h_gbs) / d_2d) * 180 / math.pi  # Convert to degrees
        else:
            theta = 90.0
        
        # LoS probability model
        p_los = 1.0 / (1.0 + self.a * np.exp(-self.b * (theta - self.a)))
        return p_los
    
    def get_path_loss(self, uav_position):
        """
        Calculate path loss using A2G probabilistic model
        Adjusted to match original BSchannels path loss for similar SINR levels
        Args:
            uav_position: [x, y, z] position of UAV
        Returns:
            path_loss: [n_GBS, n_RB] array of path loss in dB
        """
        path_loss = np.zeros((len(self.GBS_position), len(self.fc)))
        gbs_pos = self.GBS_position[0]
        
        # Calculate 3D distance
        d_3d = math.sqrt((uav_position[0] - gbs_pos[0])**2 + 
                        (uav_position[1] - gbs_pos[1])**2 + 
                        (uav_position[2] - gbs_pos[2])**2)
        
        # Use original BSchannels formula for compatibility
        # PL = 32.4 + 20*log10(fc) + 31.9*log10(d_3d)
        # This matches the original environment's path loss model
        for k in range(len(self.fc)):
            path_loss[0, k] = 32.4 + 20 * np.log10(self.fc[k]) + 31.9 * np.log10(d_3d)
        
        return path_loss

    def get_fast_fading(self, uav_position, is_los):
        """
        Generate fast fading component
        Args:
            uav_position: [x, y, z] position of UAV
            is_los: boolean, whether link is LoS
        Returns:
            fading_gain: fading gain in dB
        """
        if is_los:
            # Rician fading for LoS
            # Convert K from dB to linear
            K_linear = 10 ** (self.K_rician / 10)
            # Rician distribution: sqrt(K/(K+1)) * direct + sqrt(1/(K+1)) * scattered
            direct_component = np.sqrt(K_linear / (K_linear + 1))
            scattered_real = np.random.normal(0, 1) / np.sqrt(2)
            scattered_imag = np.random.normal(0, 1) / np.sqrt(2)
            scattered_component = np.sqrt(1 / (K_linear + 1)) * (scattered_real + 1j * scattered_imag)
            h_rician = direct_component + scattered_component
            fading_gain = 20 * np.log10(np.abs(h_rician))
        else:
            # Rayleigh fading for NLoS
            h_rayleigh = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)
            fading_gain = 20 * np.log10(np.abs(h_rayleigh))
        
        return fading_gain
    
    def get_shadowing(self, delta_distance, shadowing):
        """
        Update shadowing with spatial correlation
        Args:
            delta_distance: distance moved since last update
            shadowing: previous shadowing value
        Returns:
            updated shadowing value
        """
        decorrelation_distance = 25.0  # meters
        return (np.exp(-delta_distance / decorrelation_distance) * shadowing + 
                np.sqrt(1 - np.exp(-2 * delta_distance / decorrelation_distance)) * np.random.normal(0, 8.29))


class UAV:
    """
    UAV simulator: includes all information for a UAV
    Supports 3D coordinates and Gauss-Markov mobility model
    """

    def __init__(self, start_position, start_velocity=None, alpha=0.8, sigma_v=2.0):
        """
        Initialize UAV
        Args:
            start_position: [x, y, z] initial 3D position
            start_velocity: [vx, vy, vz] initial velocity (optional, random if None)
            alpha: memory level for Gauss-Markov model (0-1), higher = more memory
            sigma_v: standard deviation of velocity variation
        """
        self.position = np.array(start_position, dtype=np.float32)  # 3D position [x, y, z]
        
        # Initialize velocity for Gauss-Markov model
        if start_velocity is None:
            self.velocity = np.array([np.random.uniform(-5, 5), 
                                     np.random.uniform(-5, 5), 
                                     np.random.uniform(-2, 2)], dtype=np.float32)
        else:
            self.velocity = np.array(start_velocity, dtype=np.float32)
        
        # Gauss-Markov model parameters
        self.alpha = alpha  # Memory level (0 = random walk, 1 = constant velocity)
        self.sigma_v = sigma_v  # Standard deviation of velocity variation
        self.mean_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Mean velocity (can be set for directed motion)
        
        # Graph-related attributes
        self.neighbors = []
        self.destinations = []
        
    def step(self, dt):
        """
        Update UAV position using Gauss-Markov mobility model
        Args:
            dt: time step in seconds
        """
        # Update velocity using Gauss-Markov model
        # v(t+1) = alpha * v(t) + (1-alpha) * v_mean + sqrt(1-alpha^2) * w(t)
        w = np.random.normal(0, self.sigma_v, size=3)
        self.velocity = (self.alpha * self.velocity + 
                        (1 - self.alpha) * self.mean_velocity + 
                        np.sqrt(1 - self.alpha**2) * w)
        
        # Update position
        self.position = self.position + self.velocity * dt
        
        # Keep velocity within reasonable bounds for UAVs
        max_horizontal_velocity = 20.0  # m/s
        max_vertical_velocity = 5.0  # m/s
        self.velocity[0] = np.clip(self.velocity[0], -max_horizontal_velocity, max_horizontal_velocity)
        self.velocity[1] = np.clip(self.velocity[1], -max_horizontal_velocity, max_horizontal_velocity)
        self.velocity[2] = np.clip(self.velocity[2], -max_vertical_velocity, max_vertical_velocity)


# Keep Vehicle as alias for backward compatibility (deprecated)
Vehicle = UAV


class Environ:
    def __init__(self, n_veh, n_RB, beta=0.5, circuit_power=0.06, optimization_target='SE_EE',
                 area_size=25.0, height_min=1.5, height_max=1.5, comm_range=500.0,
                 semantic_A_max=1.0, semantic_beta=2.0):
        """
        Initialize environment for UAV network
        Args:
            n_veh: number of UAVs
            n_RB: number of resource blocks
            beta: Weight for SE in reward calculation when optimization_target='SE_EE' (0.0-1.0), EE weight = 1 - beta (default: 0.5)
            circuit_power: Circuit power in linear scale for EE calculation (default: 0.06)
            optimization_target: Optimization target - 'SE', 'EE', or 'SE_EE' (default: 'SE_EE')
            area_size: Size of the area in meters (default: 25m x 25m to match original environment)
            height_min: Minimum UAV height in meters (default: 50m)
            height_max: Maximum UAV height in meters (default: 200m)
            comm_range: Communication range threshold for graph construction in meters (default: 500m)
            semantic_A_max: Maximum semantic accuracy (mAP) (default: 1.0)
            semantic_beta: Compression ratio sensitivity parameter (default: 2.0)
            collision_penalty: Penalty for RB collision (default: -0.5)
            low_accuracy_penalty: Penalty for low accuracy (default: -0.3)
            accuracy_threshold: Minimum acceptable accuracy threshold (default: 0.5)
        """
        # Area dimensions (3D space for UAVs)
        self.width = area_size
        self.height = area_size
        self.depth = height_max - height_min
        self.height_min = height_min
        self.height_max = height_max
        
        # Communication range for graph topology
        self.comm_range = comm_range

        # Channel model: A2G instead of BSchannels
        self.A2GChannels = A2GChannels()
        self.vehicles = []  # List of UAVs (keeping name for compatibility)

        self.demand = []
        self.cellular_Shadowing = []
        self.delta_distance = []
        self.cellular_channels_abs = []
        self.los_status = []  # LoS/NLoS status for each UAV

        self.cellular_power_dB_List = [24, 21, 18, 15, 12, 9, 6, 3, 0]   # the power levels
        self.sig2_dB = -160
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.uavAntGain = 3  # Renamed from vehAntGain
        self.vehAntGain = 3  # Keep for backward compatibility
        self.uavNoiseFigure = 9
        self.vehNoiseFigure = 9  # Keep for backward compatibility

        self.n_RB = n_RB
        self.n_Veh = n_veh

        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/UAV position every 100 ms
        self.BW = [0.18, 0.18, 0.36, 0.36, 0.36, 0.72, 0.72, 0.72, 1.44, 1.44] # MHz
        self.channel_choice = np.zeros([self.n_RB])
        self.success = np.zeros([self.n_Veh])
        self.sig2 = [i * 1e6 * 10 ** (self.sig2_dB / 10) for i in self.BW]
        
        # Reward weighting parameters for SE and EE (configurable)
        self.optimization_target = optimization_target  # 'SE', 'EE', or 'SE_EE'
        self.beta = beta  # Weight for SE (0.0 to 1.0), EE weight = 1 - beta (only used when optimization_target='SE_EE')
        self.circuit_power = circuit_power  # Circuit power in linear scale

        # Semantic Communication parameters
        self.semantic_A_max = semantic_A_max  # Maximum semantic accuracy (mAP)
        self.semantic_beta = semantic_beta  # Compression ratio sensitivity parameter

    def add_new_vehicles(self, start_position, start_velocity=None):
        """
        Add a new UAV to the environment
        Args:
            start_position: [x, y, z] 3D position
            start_velocity: [vx, vy, vz] initial velocity (optional)
        """
        uav = UAV(start_position, start_velocity)
        self.vehicles.append(uav)

    def add_new_vehicles_by_number(self, n):
        """
        Add n UAVs with random 3D positions
        """
        for i in range(n):
            # Random 3D position
            start_position = [
                np.random.uniform(0, self.width),
                np.random.uniform(0, self.height),
                np.random.uniform(self.height_min, self.height_max)
            ]
            # Random initial velocity (optional, UAV will generate if None)
            self.add_new_vehicles(start_position, start_velocity=None)

        # Initialize channels
        self.cellular_Shadowing = np.random.normal(0, 8.29, len(self.vehicles))
        # Calculate initial delta_distance (movement since last update)
        self.delta_distance = np.zeros(len(self.vehicles))
        for i, uav in enumerate(self.vehicles):
            self.delta_distance[i] = np.linalg.norm(uav.velocity) * self.time_slow

    def renew_positions(self):
        """
        Update UAV positions using Gauss-Markov mobility model
        Maintains interface compatibility with existing RL agent
        """
        for i, uav in enumerate(self.vehicles):
            # Store previous position for delta_distance calculation
            prev_position = uav.position.copy()
            
            # Update position using Gauss-Markov model
            uav.step(self.time_slow)
            
            # Calculate distance moved
            self.delta_distance[i] = np.linalg.norm(uav.position - prev_position)
            
            # Boundary handling: wrap around or reflect (wrap around for now)
            if uav.position[0] < 0:
                uav.position[0] = self.width + uav.position[0]
            elif uav.position[0] > self.width:
                uav.position[0] = uav.position[0] - self.width
                
            if uav.position[1] < 0:
                uav.position[1] = self.height + uav.position[1]
            elif uav.position[1] > self.height:
                uav.position[1] = uav.position[1] - self.height
            
            # Keep altitude within bounds
            uav.position[2] = np.clip(uav.position[2], self.height_min, self.height_max)


    def renew_neighbor(self):
        """
        Determine the neighbors of each UAV based on communication range
        Maintains interface compatibility with existing RL agent
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        
        # Calculate pairwise 3D distances
        n_uavs = len(self.vehicles)
        positions = np.array([uav.position for uav in self.vehicles])
        Distance = np.zeros((n_uavs, n_uavs))
        
        for i in range(n_uavs):
            for j in range(n_uavs):
                if i != j:
                    Distance[i, j] = np.linalg.norm(positions[i] - positions[j])
                else:
                    Distance[i, j] = np.inf
        
        # Find neighbors within communication range
        for i in range(len(self.vehicles)):
            neighbors = np.where(Distance[i, :] <= self.comm_range)[0].tolist()
            self.vehicles[i].neighbors = neighbors
            self.vehicles[i].destinations = neighbors
    
    def get_adjacency_matrix(self, threshold=None):
        """
        Get adjacency matrix for graph structure based on distance thresholds
        Args:
            threshold: Distance threshold in meters (default: self.comm_range)
        Returns:
            adjacency_matrix: [n_uavs, n_uavs] binary adjacency matrix
            edge_features: Optional edge features (distances) if needed
        """
        if threshold is None:
            threshold = self.comm_range
        
        n_uavs = len(self.vehicles)
        adjacency_matrix = np.zeros((n_uavs, n_uavs), dtype=np.float32)
        positions = np.array([uav.position for uav in self.vehicles])
        
        # Calculate pairwise distances and build adjacency matrix
        for i in range(n_uavs):
            for j in range(n_uavs):
                if i != j:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance <= threshold:
                        adjacency_matrix[i, j] = 1.0
        
        return adjacency_matrix

    def renew_BS_channel(self):
        """
        Renew slow fading channel using A2G model
        Maintains interface compatibility with existing RL agent
        """
        self.cellular_pathloss = np.zeros((len(self.vehicles), self.n_RB))
        self.los_status = []  # Store LoS status for each UAV

        for i in range(len(self.vehicles)):
            # Update shadowing with spatial correlation
            self.cellular_Shadowing[i] = self.A2GChannels.get_shadowing(
                self.delta_distance[i], self.cellular_Shadowing[i])
            
            # Get path loss using A2G model
            pathloss = self.A2GChannels.get_path_loss(self.vehicles[i].position)[0]
            self.cellular_pathloss[i] = pathloss
            
            # Determine LoS status (probabilistic)
            p_los = self.A2GChannels.get_los_probability(self.vehicles[i].position)
            is_los = np.random.random() < p_los
            self.los_status.append(is_los)
        
        # Combine path loss and shadowing
        self.cellular_channels_abs = (self.cellular_pathloss + 
                                     np.repeat(self.cellular_Shadowing[:, np.newaxis], self.n_RB, axis=1))

    def renew_BS_channels_fastfading(self):
        """
        Renew fast fading channel using A2G model (Rician for LoS, Rayleigh for NLoS)
        Maintains interface compatibility with existing RL agent
        """
        cellular_channels_with_fastfading = self.cellular_channels_abs.copy()
        
        # Apply fast fading based on LoS status
        for i in range(len(self.vehicles)):
            is_los = self.los_status[i] if i < len(self.los_status) else True
            fading_gain = self.A2GChannels.get_fast_fading(self.vehicles[i].position, is_los)
            
            # Apply fading to all RBs for this UAV
            for k in range(self.n_RB):
                cellular_channels_with_fastfading[i, k] += fading_gain
        
        self.cellular_channels_with_fastfading = cellular_channels_with_fastfading

    def Compute_Performance_Reward_Failure(self, actions_all=None, IS_PPO=False):
        """
        Compute performance metrics for failure case (low SINR or collision)
        Args:
            actions_all: Optional, action array [n_veh, 2 or 3]
            IS_PPO: bool or int, whether using PPO/DDPG format
        Returns:
            (cellular_Rate, cellular_SINR, SE, EE, semantic_accuracy, semantic_EE, collisions)
        """
        # 如果没有提供actions，则使用默认的第一个RB和默认压缩比
        if actions_all is None:
            actions = np.zeros(len(self.vehicles), dtype=int)
            rho = np.ones(len(self.vehicles)) * 0.5  # Default compression ratio
        else:
            # 兼容处理：将整数也当作布尔值
            is_ppo_mode = bool(IS_PPO) if IS_PPO is not None else False
            actions = actions_all[:, 0].astype(int)  # the channel_selection_part (always integer)
            if actions_all.shape[1] >= 3:
                rho = np.clip(actions_all[:, 2], 0.0, 1.0)
            else:
                rho = np.ones(len(self.vehicles)) * 0.5  # Default compression ratio

        transmit_power = np.zeros(len(self.vehicles))
        # Use minimum power for failure case
        for i in range(len(self.vehicles)):
            transmit_power[i] = self.cellular_power_dB_List[0]
        
        # Compute basic metrics (similar to success case but with low power)
        channel_choice = np.zeros(self.n_RB)
        cellular_SINR = np.zeros(self.n_Veh)
        cellular_Signals = np.zeros([self.n_Veh, self.n_RB])
        
        for i in range(len(self.vehicles)):
            l = actions[i]
            channel_choice[l] += 1
            cellular_Signals[i, l] = 10 ** ((transmit_power[i] -
                                            self.cellular_channels_with_fastfading[i, l] + 
                                            self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

        # Compute SINR (assuming interference)
        for i in range(len(self.vehicles)):
            l = actions[i]
            # Low SINR for failure case
            interference = self.sig2[l] * 10  # High interference
            cellular_SINR[i] = cellular_Signals[i, l] / (interference + self.sig2[l])
        
        # Calculate semantic accuracy (will be low due to low SINR)
        semantic_accuracy = np.zeros(self.n_Veh)
        semantic_EE = np.zeros(self.n_Veh)
        transmission_power_linear = 10 ** (transmit_power / 10)

        for i in range(len(self.vehicles)):
            semantic_accuracy[i] = self.compute_semantic_accuracy(rho[i], cellular_SINR[i])
            total_power = transmission_power_linear[i] + self.circuit_power
            if total_power > 0:
                semantic_EE[i] = semantic_accuracy[i] / total_power
        
        # Keep old metrics for backward compatibility
        cellular_Rate = np.zeros(self.n_Veh)
        SE = np.zeros(self.n_Veh)
        EE = np.zeros(self.n_Veh)
        for i in range(len(self.vehicles)):
            l = actions[i]
            cellular_Rate[i] = np.log2(1 + cellular_SINR[i]) * self.BW[l]
            SE[i] = np.divide(cellular_Rate[i], self.BW[l])
            power_linear = transmission_power_linear[i]
            EE[i] = np.divide(cellular_Rate[i], power_linear + self.circuit_power)

        collisions = np.zeros(self.n_Veh)  # No collision info in failure case
        
        return (cellular_Rate, cellular_SINR, SE, EE, 
                semantic_accuracy, semantic_EE, collisions)

    def compute_semantic_accuracy(self, rho, sinr):
        """
        Compute semantic accuracy (mAP) based on compression ratio and SINR
        Formula: Accuracy = A_max * (1 - exp(-beta * rho)) * log(1 + SINR) / log(2)
        
        Args:
            rho: Compression ratio [0, 1]
            sinr: Signal-to-Interference-plus-Noise Ratio (linear scale)
        
        Returns:
            accuracy: Semantic accuracy (mAP) in [0, A_max]
        """
        # Ensure rho is in [0, 1]
        rho = np.clip(rho, 0.0, 1.0)
        
        # Compression ratio term: (1 - exp(-beta * rho))
        compression_term = 1.0 - np.exp(-self.semantic_beta * rho)
        
        # SINR term: log(1 + SINR) / log(2) to normalize to [0, 1] range approximately
        # For better scaling, we use log(1 + SINR) / log(1 + max_SINR)
        # Assuming max_SINR around 100 (20 dB), log(101) ≈ 4.6
        max_sinr = 100.0  # Maximum expected SINR (linear)
        sinr_term = np.log(1.0 + sinr) / np.log(1.0 + max_sinr)
        sinr_term = np.clip(sinr_term, 0.0, 1.0)
        
        # Combined accuracy
        accuracy = self.semantic_A_max * compression_term * sinr_term
        
        return accuracy

    def Compute_Performance_Reward_Train(self, actions_all, IS_PPO):
        """
        Compute performance metrics for training with Semantic Communication
        Args:
            actions_all: [n_veh, 3] array (updated for semantic communication)
                - actions_all[:, 0]: RB selection (integer indices)
                - actions_all[:, 1]: transmit power
                    - If IS_PPO=True (or DDPG): direct power value in dB
                    - If IS_PPO=False: power level index (0-8)
                - actions_all[:, 2]: semantic compression ratio rho [0, 1] (NEW)
            IS_PPO: bool or int
                - True (or 1): For PPO/DDPG (actions format: [RB_index, power_dB, rho])
                - False (or 0): For traditional methods (actions format: [RB_index, power_level_index, rho])
        """
        # 兼容处理：将整数1也当作True，保持向后兼容
        is_ppo_mode = bool(IS_PPO) if IS_PPO is not None else False
        
        # Handle 2D or 3D actions (backward compatibility)
        if actions_all.shape[1] == 2:
            # Old format: [RB, Power], set rho to default
            actions = actions_all[:, 0].astype(int)
            rho = np.ones(len(self.vehicles)) * 0.8  # Default compression ratio
            if is_ppo_mode:
                transmit_power = actions_all[:, 1]
            else:
                power_selection = actions_all[:, 1].astype(int)
                transmit_power = np.zeros(len(self.vehicles))
                for i in range(len(self.vehicles)):
                    transmit_power[i] = self.cellular_power_dB_List[power_selection[i]]
        else:
            # New format: [RB, Power, rho]
            actions = actions_all[:, 0].astype(int)
            rho = np.clip(actions_all[:, 2], 0.0, 1.0)  # Compression ratio [0, 1]
            if is_ppo_mode:
                transmit_power = actions_all[:, 1]
            else:
                power_selection = actions_all[:, 1].astype(int)
                transmit_power = np.zeros(len(self.vehicles))
                for i in range(len(self.vehicles)):
                    transmit_power[i] = self.cellular_power_dB_List[power_selection[i]]
            if is_ppo_mode:
                transmit_power = actions_all[:, 1]
            else:
                power_selection = actions_all[:, 1].astype(int)
            transmit_power = np.zeros(len(self.vehicles))

        # ------------ Compute signal and interference --------------------
        channel_choice = np.zeros(self.n_RB)
        cellular_SINR = np.zeros(self.n_Veh)
        cellular_Signals = np.zeros([self.n_Veh, self.n_RB])
        cellular_Interference = np.zeros([self.n_RB])  # Total interference per RB
        
        for i in range(len(self.vehicles)):
            l = actions[i]
            if not is_ppo_mode:
                # 传统模式：需要从power_selection索引转换为dB值
                transmit_power[i] = self.cellular_power_dB_List[power_selection[i]]
            
            # Count channel usage (for collision detection)
            channel_choice[actions[i]] += 1
            
            # Compute signal power (linear scale)
            signal_power_linear = 10 ** ((transmit_power[i] -
                                         self.cellular_channels_with_fastfading[i, l] + 
                                         self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
            cellular_Signals[i, l] = signal_power_linear
            
            # Add to interference on this RB
            cellular_Interference[l] += signal_power_linear
        
        # Detect collisions (multiple UAVs on same RB)
        collisions = np.zeros(self.n_Veh)
        for i in range(len(self.vehicles)):
            if channel_choice[actions[i]] > 1:
                collisions[i] = 1  # Collision detected
        
        # Compute SINR (considering interference from other UAVs on same RB)
        for i in range(len(self.vehicles)):
            l = actions[i]
            # Interference = total power on this RB - own signal + noise
            interference_total = cellular_Interference[l] - cellular_Signals[i, l] + self.sig2[l]
            cellular_SINR[i] = cellular_Signals[i, l] / interference_total
        
        # Success: no collision and SINR above threshold
        self.success = np.zeros([self.n_Veh])
        sinr_threshold_linear = 10 ** (3.16 / 10)  # 3.16 dB in linear scale
        for i in range(len(self.vehicles)):
            if collisions[i] == 0 and cellular_SINR[i] > sinr_threshold_linear:
                    self.success[i] = 1
        
        self.channel_choice = channel_choice
        
        # ------------ Compute Semantic Communication Metrics --------------------
        semantic_accuracy = np.zeros(self.n_Veh)
        semantic_EE = np.zeros(self.n_Veh)
        transmission_power_linear = np.zeros(self.n_Veh)
        
        for i in range(len(self.vehicles)):
            # Convert power from dB to linear scale
            transmission_power_linear[i] = 10 ** (transmit_power[i] / 10)
            
            # Compute semantic accuracy (mAP) based on compression ratio and SINR
            semantic_accuracy[i] = self.compute_semantic_accuracy(rho[i], cellular_SINR[i])
            
            # Semantic Energy Efficiency = Accuracy / (Transmission_Power + Circuit_Power)
            total_power = transmission_power_linear[i] + self.circuit_power
            if total_power > 0:
                semantic_EE[i] = semantic_accuracy[i] / total_power
            else:
                semantic_EE[i] = 0.0
        
        # No additional penalties - failures handled by success flag and SINR threshold
        semantic_EE_penalized = semantic_EE
        
        # Keep old metrics for backward compatibility
        cellular_Rate = np.zeros(self.n_Veh)
        SE = np.zeros(self.n_Veh)
        EE = np.zeros(self.n_Veh)
        for i in range(len(self.vehicles)):
            l = actions[i]
            cellular_Rate[i] = np.log2(1 + cellular_SINR[i]) * self.BW[l]
            SE[i] = np.divide(cellular_Rate[i], self.BW[l])
            power_linear = transmission_power_linear[i]
            EE[i] = np.divide(cellular_Rate[i], power_linear + self.circuit_power)

        return (cellular_Rate, cellular_SINR, SE, EE, 
                semantic_accuracy, semantic_EE_penalized, collisions)

    def Compute_Performance_Reward_Test_rand(self, actions_all, IS_PPO):
        if IS_PPO:
            actions_probability = actions_all[:self.n_RB]
            actions = np.random.choice(range(len(actions_probability)),
                                       p=actions_probability.ravel())
            transmit_power = actions_all[-1]
        else:
            actions = actions_all[:, 0]  # the channel_selection_part
            power_selection = actions_all[:, 1]  # power selection

        # ------------ Compute cellular rate --------------------
        channel_choice = np.zeros(self.n_RB)
        cellular_Rate = np.zeros(self.n_RB)
        SE = np.zeros(self.n_RB)
        cellular_Signals = np.zeros([self.n_Veh, self.n_RB])
        cellular_Interference = np.zeros(self.n_RB)  # cellular interference index RB
        for i in range(len(self.vehicles)):
            for l in range(self.n_RB):
                if IS_PPO:
                    cellular_Signals[i, l] = 10 ** ((transmit_power -
                                               self.cellular_channels_with_fastfading[
                                                   i, l] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                else:
                    cellular_Signals[i, l] = 10 ** ((self.cellular_power_dB_List[power_selection[i]] -
                                               self.cellular_channels_with_fastfading[
                                                   i, l] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                if IS_PPO:
                    if (l == actions):
                        channel_choice[actions] += 1
                        cellular_Interference[actions] += 10 ** ((transmit_power -
                                                                   self.cellular_channels_with_fastfading[
                                                                       i, actions]
                                                                   + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                else:
                    if (l == actions[i]):
                        channel_choice[actions[i]] += 1
                        cellular_Interference[actions[i]] += 10 ** ((self.cellular_power_dB_List[
                                                                       power_selection[i]] -
                                                                   self.cellular_channels_with_fastfading[
                                                                       i, actions[i]]
                                                                   + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

        self.success = np.zeros([self.n_Veh])
        for i in range(len(self.vehicles)):
            for l in range(self.n_RB):
                if (len((channel_choice[channel_choice!=0])) == self.n_Veh):
                    self.success[i] = 1
                elif (channel_choice[l] <= 1):
                    if IS_PPO:
                        if (actions == l):
                            self.success[i] = 1
                    else:
                        if (actions[i] == l):
                            self.success[i] = 1
        self.channel_choice = channel_choice
        self.cellular_Interference = cellular_Interference + self.sig2
        cellular_Rate_all = np.log2(1 + np.divide(cellular_Signals, self.sig2)) * self.BW

        for i in range(len(self.vehicles)):
            if IS_PPO:
                action_idx = actions
            else:
                action_idx = actions[i]
            if  (self.success[i] == 1) and (cellular_Rate_all[i, action_idx] > 0.18):
                cellular_Rate[i] = cellular_Rate_all[i, action_idx]
            else:
                cellular_Rate[i] = 0

        # Calculate SE (Spectral Efficiency) and EE (Energy Efficiency)
        EE = np.zeros(self.n_Veh)
        transmit_power_array = np.zeros(self.n_Veh)
        
        for i in range(len(self.vehicles)):
            # Get transmit power for EE calculation
            if IS_PPO:
                action_idx = actions
                transmit_power_array[i] = transmit_power
            else:
                action_idx = actions[i]
                transmit_power_array[i] = self.cellular_power_dB_List[power_selection[i]]
            
            # SE = Rate / BW
            if (self.success[i] == 1):
                SE[i] = np.divide(cellular_Rate[i], self.BW[action_idx])
                
                # EE = Rate / (Power + Circuit_Power)
                power_linear = 10 ** (transmit_power_array[i] / 10)
                EE[i] = np.divide(cellular_Rate[i], power_linear + self.circuit_power)
            else:
                SE[i] = 0
                EE[i] = 0

        return cellular_Rate, SE, EE

    def act_for_meta_training(self, actions, IS_PPO):
        """
        Execute action and compute reward for meta training with Semantic Communication
        Args:
            actions: [n_veh, 3] action array (updated for semantic communication)
                - actions[:, 0]: RB selection
                - actions[:, 1]: transmit power
                - actions[:, 2]: semantic compression ratio rho [0, 1]
            IS_PPO: bool or int, whether using PPO/DDPG format
        Returns:
            reward: Semantic Energy Efficiency (mAP/Energy) with penalties
        """
        action_temp = actions.copy()
        # 兼容处理：确保IS_PPO参数被正确处理
        is_ppo_mode = bool(IS_PPO) if IS_PPO is not None else False
        
        # Compute performance metrics (includes semantic communication metrics)
        results = self.Compute_Performance_Reward_Train(action_temp, is_ppo_mode)
        (cellular_Rate, cellular_SINR, SE, EE, 
         semantic_accuracy, semantic_EE_penalized, collisions) = results
        
        # Use Semantic Energy Efficiency as reward
        semantic_EE_sum = 0.0
        for i in range(len(self.success)):
            if self.success[i] == 1:
                semantic_EE_sum += semantic_EE_penalized[i]
        
        # Return sum (not average) for meta training
        reward = semantic_EE_sum
        return reward

    def act_for_training(self, actions, IS_PPO):
        """
        Execute action and compute reward for training with Semantic Communication
        Minimal adaptation from original code structure
        Args:
            actions: [n_veh, 3] action array
                - actions[:, 0]: RB selection (integer)
                - actions[:, 1]: transmit power (dB if IS_PPO=True, index if False)
                - actions[:, 2]: semantic compression ratio rho [0, 1]
            IS_PPO: bool or int
        Returns:
            reward: Semantic Energy Efficiency based reward
        """
        action_temp = actions.copy()
        is_ppo_mode = bool(IS_PPO) if IS_PPO is not None else False
        
        # Compute performance metrics (includes semantic communication metrics)
        results = self.Compute_Performance_Reward_Train(action_temp, is_ppo_mode)
        (cellular_Rate, cellular_SINR, SE, EE, 
         semantic_accuracy, semantic_EE, collisions) = results
        
        # Get failure case metrics (low SINR scenario)
        failure_results = self.Compute_Performance_Reward_Failure(action_temp, is_ppo_mode)
        (_, _, failure_SE, _, _, failure_semantic_EE, _) = failure_results
        
        # Use Semantic-EE as reward (similar to original EE)
        SE_sum = 0.0
        Semantic_EE_sum = 0.0
        
        # Training SINR threshold (same as original)
        # Using 2.5 dB instead of 3.3 dB to allow more successful transmissions
        # This is a safety margin: test threshold is 3.16 dB (linear: 2.07)
        training_sinr_threshold = 2.5  # dB (linear: 1.78)
        
        # Original logic: three cases
        for i in range(len(self.success)):
            if (self.success[i] == 1) and (cellular_SINR[i] > training_sinr_threshold):
                # Case 1: Success with good SINR - use normal SE/EE
                SE_sum += SE[i]
                Semantic_EE_sum += semantic_EE[i]
            elif (self.success[i] == 1):
                # Case 2: Success but SINR not high enough - use failure SE/EE (smaller values)
                SE_sum += failure_SE[i]
                Semantic_EE_sum += failure_semantic_EE[i]
            else:
                # Case 3: Failure (collision) - penalty
                # Original code breaks here, but we accumulate to evaluate all UAVs
                SE_sum += -1  # Failure penalty
                Semantic_EE_sum += -1  # Failure penalty
        
        # Calculate reward based on optimization target (same as original)
        if self.optimization_target == 'SE':
            reward = SE_sum / self.n_Veh
        elif self.optimization_target == 'EE':
            # Use Semantic-EE instead of traditional EE
            reward = Semantic_EE_sum / self.n_Veh
        elif self.optimization_target == 'SE_EE':
            # Weighted combination: beta * SE + (1-beta) * Semantic-EE
            reward = (self.beta * SE_sum + (1 - self.beta) * Semantic_EE_sum) / self.n_Veh
        else:
            # Default to SE_EE
            reward = (self.beta * SE_sum + (1 - self.beta) * Semantic_EE_sum) / self.n_Veh
        
        return reward

    def act_for_testing(self, actions, IS_PPO):

        action_temp = actions.copy()
        cellular_Rate, cellular_SINR, SE, EE = self.Compute_Performance_Reward_Train(action_temp, IS_PPO)
        for i in range(len(cellular_Rate)):
            if (self.success[i] == 1) and (cellular_SINR [i] > 3.16):
                SE[i] = SE[i]
                EE[i] = EE[i]
            else:
                SE[i] = 0
                EE[i] = 0

        return SE, EE, cellular_Rate

    def act_for_testing_marl(self, actions, IS_PPO):
        done = True
        action_temp = actions.copy()
        cellular_Rate, cellular_SINR, SE, EE = self.Compute_Performance_Reward_Train(action_temp, IS_PPO)
        for i in range(len(cellular_Rate)):
            if (self.success[i] == 1) and  (cellular_SINR [i] > 3.16):
                SE[i] = SE[i]
                EE[i] = EE[i]
            else:
                done = False
                break
        if not done:
            for i in range (len(SE)):
                SE[i] = 0
                EE[i] = 0
        return SE, EE, cellular_Rate

    def act_for_testing_rand(self, actions, IS_PPO):

        action_temp = actions.copy()
        cellular_Rate, SE, EE = self.Compute_Performance_Reward_Test_rand(action_temp, IS_PPO)

        return SE, EE, cellular_Rate

    def act_for_testing_sarl(self, actions, IS_PPO):

        action_temp = actions.copy()
        cellular_Rate, SE, EE = self.Compute_Performance_Reward_Test_rand(action_temp, IS_PPO)
        return SE, EE, cellular_Rate

    def new_random_game(self, n_Veh=0):
        # make a new game

        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(self.n_Veh)
        self.renew_neighbor()
        self.renew_BS_channel()
        self.renew_BS_channels_fastfading()



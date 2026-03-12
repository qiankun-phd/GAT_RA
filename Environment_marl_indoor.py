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

    def __init__(self, path_loss_model='A2G'):
        # Ground Base Station (GBS) position [x, y, z] in meters
        self.GBS_position = np.array([[50, 50, 0]])  # Ground level base station
        self.fc = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  # Carrier frequencies in GHz
        self.path_loss_model = path_loss_model  # 'A2G' or '3GPP_UMa'
        
        # A2G channel parameters (standard ITU/3GPP A2G values)
        self.eta_LoS = 0.0   # Additional path loss for LoS (dB)
        self.eta_NLoS = 7.0  # Additional path loss for NLoS (dB)
        self.c = 3e8  # Speed of light (m/s)
        
        # Rician fading parameters for LoS
        self.K_rician = 10.0  # Rician K-factor (dB) for LoS links
        
        # Environment parameters for LoS probability (psi_a, psi_b in paper)
        self.a = 9.61  # psi_a: environment-dependent
        self.b = 0.16  # psi_b: environment-dependent
        
    def get_los_probability(self, uav_position):
        """
        LoS probability P_LoS(θ) = 1/(1 + ψ_a exp(-ψ_b(θ - ψ_a))).
        Paper: θ_i = arcsin((z_i - H_BS)/d_i) with d_i = 3D distance (radians then convert to degrees).
        """
        gbs_pos = self.GBS_position[0]
        d_3d = math.sqrt((uav_position[0] - gbs_pos[0])**2 +
                        (uav_position[1] - gbs_pos[1])**2 +
                        (uav_position[2] - gbs_pos[2])**2)
        d_3d = max(d_3d, 1e-6)
        h_uav = uav_position[2]
        h_gbs = gbs_pos[2]
        
        # Paper: theta = arcsin((z_i - H_BS) / d_i), then degrees for sigmoid
        sin_theta = (h_uav - h_gbs) / d_3d
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        theta_rad = math.asin(sin_theta)
        theta = theta_rad * 180.0 / math.pi  # degrees
        
        p_los = 1.0 / (1.0 + self.a * np.exp(-self.b * (theta - self.a)))
        return p_los
    
    def get_path_loss(self, uav_position, is_los=True):
        """
        Path loss in dB.
        Models:
          - A2G (default): PL = 32.4 + 20*log10(fc) + 31.9*log10(d_3d) + eta
          - 3GPP_UMa: PL = 128.1 + 37.6*log10(d_3d/1000), d_3d in m
        """
        path_loss = np.zeros((len(self.GBS_position), len(self.fc)))
        gbs_pos = self.GBS_position[0]
        d_3d = math.sqrt((uav_position[0] - gbs_pos[0])**2 +
                        (uav_position[1] - gbs_pos[1])**2 +
                        (uav_position[2] - gbs_pos[2])**2)
        d_3d = max(d_3d, 0.1)

        if self.path_loss_model == '3GPP_UMa':
            # 3GPP Urban Macro: PL = 128.1 + 37.6*log10(d_3d/1000), d_3d in m
            pl_dB = 128.1 + 37.6 * np.log10(d_3d / 1000.0)
            path_loss[:] = pl_dB
        else:
            # A2G (ITU): PL = 32.4 + 20*log10(fc) + 31.9*log10(d_3d) + eta
            eta_extra = self.eta_LoS if is_los else self.eta_NLoS
            for k in range(len(self.fc)):
                path_loss[0, k] = 32.4 + 20 * np.log10(self.fc[k]) + 31.9 * np.log10(d_3d) + eta_extra
        return path_loss

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

    def __init__(self, start_position, start_velocity=None, alpha=0.8, sigma_v=0.3):
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
            self.velocity = np.array([np.random.uniform(-1, 1), 
                                     np.random.uniform(-1, 1), 
                                     np.random.uniform(-1, 1)], dtype=np.float32)
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
        # Gauss-Markov velocity update:
        # v(t+1) = alpha * v(t) + (1-alpha) * v_mean + sqrt(1-alpha^2) * w(t)
        w = np.random.normal(0, self.sigma_v, size=3).astype(np.float32)
        self.velocity = (self.alpha * self.velocity +
                         (1 - self.alpha) * self.mean_velocity +
                         np.sqrt(1 - self.alpha ** 2) * w)

        # Keep velocity within reasonable bounds for UAVs
        max_horizontal_velocity = 3.0   # m/s
        max_vertical_velocity   = 1.0   # m/s
        self.velocity[0] = np.clip(self.velocity[0], -max_horizontal_velocity, max_horizontal_velocity)
        self.velocity[1] = np.clip(self.velocity[1], -max_horizontal_velocity, max_horizontal_velocity)
        self.velocity[2] = np.clip(self.velocity[2], -max_vertical_velocity,   max_vertical_velocity)

        # Update position
        self.position = self.position + self.velocity * dt


# Keep Vehicle as alias for backward compatibility (deprecated)
Vehicle = UAV


class Environ:
    # SINR success threshold: 3.16 dB -> linear ~2.07
    SINR_THRESHOLD_DB = 3.16
    SINR_THRESHOLD_LINEAR = 10 ** (SINR_THRESHOLD_DB / 10)

    def __init__(self, n_veh, n_RB, beta=0.5, circuit_power=0.06, optimization_target='SE_EE',
                 area_size=100.0, height_min=20, height_max=25, comm_range=500.0,
                 task_sim_A_peak=0.7128, task_sim_xi=10.0, task_sim_zeta=0.2313,
                 task_sim_gamma0=0.0, task_sim_b=0.3249, sig2_dB=None, path_loss_offset_dB=0,
                 path_loss_model='A2G'):
        """
        Initialize environment for UAV network
        Args:
            n_veh: number of UAVs
            n_RB: number of resource blocks
            beta: (Deprecated, not used) Kept for backward compatibility
            circuit_power: Circuit power in linear scale for EE calculation (default: 0.06)
            optimization_target: Fixed to 'SEE' (only optimize Semantic-EE)
            area_size: Size of the area in meters (default: 25m x 25m to match original environment)
            height_min: Minimum UAV height in meters (default: 50m)
            height_max: Maximum UAV height in meters (default: 200m)
            comm_range: Communication range threshold for graph construction in meters (default: 500m)
            collision_penalty: Penalty for RB collision (default: -0.5)
            low_accuracy_penalty: Penalty for low accuracy (default: -0.3)
            accuracy_threshold: Minimum acceptable accuracy threshold (default: 0.5)
            sig2_dB: Noise power spectral density in dB (default: -160). Use about -60 to -65
                     so that SINR falls roughly in -10~20 dB with current path loss.
            path_loss_offset_dB: Offset subtracted from path loss (default: 0). Use 50~55 to raise
                     min SNR from ~-70 dB to >= -20 dB for area_size=500.
            path_loss_model: 'A2G' (default, ITU) or '3GPP_UMa' (PL=128.1+37.6*log10(d_3d/1000)).
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
        self.A2GChannels = A2GChannels(path_loss_model=path_loss_model)
        # GBS 置于区域中心，与 area_size 一致（如 500×500 则 (250,250,0)）
        self.A2GChannels.GBS_position = np.array([[area_size / 2.0, area_size / 2.0, 0.0]])
        self.vehicles = []  # List of UAVs (keeping name for compatibility)

        self.cellular_Shadowing = []
        self.delta_distance = []
        self.cellular_channels_abs = []

        self.cellular_power_dB_List = [27, 21, 18, 15, 12, 9, 6, 3, 0]   # the power levels
        self.sig2_dB = -160 if sig2_dB is None else float(sig2_dB)
        self.path_loss_offset_dB = float(path_loss_offset_dB)
        self.bsAntGain = 10
        self.bsNoiseFigure = 5
        self.uavAntGain = 3
        self.uavNoiseFigure = 9

        self.n_RB = n_RB
        self.n_Veh = n_veh

        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/UAV position every 100 ms
        BW_base = [0.18, 0.18, 0.36, 0.36, 0.36, 0.72, 0.72, 0.72, 1.44, 1.44]  # MHz, 10 RBs
        self.BW = [BW_base[i % len(BW_base)] for i in range(n_RB)]  # 支持 n_RB != 10
        self.channel_choice = np.zeros([self.n_RB])
        self.success = np.zeros([self.n_Veh])
        self.similarity_success = np.zeros([self.n_Veh])  # Track similarity threshold achievement
        self.sig2 = [i * 1e6 * 10 ** (self.sig2_dB / 10) for i in self.BW]
        
        # Initialize semantic communication metrics for state representation
        self.semantic_accuracy = np.zeros(self.n_Veh)
        self.semantic_EE = np.zeros(self.n_Veh)
        self.semantic_Rate = np.zeros(self.n_Veh)
        self.rho_current = np.zeros(self.n_Veh)  # Current compression ratio
        self.cellular_SINR = np.zeros(self.n_Veh)  # Store SINR for state
        
        # Optimization target: only Semantic-EE (SEE)
        self.optimization_target = 'SEE'  # Fixed to SEE (only optimize Semantic-EE)
        self.beta = beta  # Deprecated, kept for backward compatibility (not used in reward calculation)
        self.circuit_power = circuit_power  # Circuit power in linear scale

        # 图片语义熵 (semantic entropy of image)，计算 semantic rate 时乘以该因子
        self.semantic_entropy_image = 3.22
        # Task similarity model: Q = A_peak * (1 - e^{-xi*rho}) / (1 + e^{-zeta*(gamma-gamma0)}) + b
        # Fitted parameters (used in compute_semantic_accuracy)
        self.task_sim_A_peak = task_sim_A_peak
        self.task_sim_xi = task_sim_xi
        self.task_sim_zeta = task_sim_zeta
        self.task_sim_gamma0 = task_sim_gamma0
        self.task_sim_b = task_sim_b

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

    def renew_BS_channel(self):
        """
        Renew slow fading channel using A2G model.
        Path loss uses the expected value over LoS/NLoS:
          PL_expected = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS
        This avoids per-episode 19 dB jumps from random LoS/NLoS draws,
        keeping the slow-fading channel stationary for RL training.
        """
        self.cellular_pathloss = np.zeros((len(self.vehicles), self.n_RB))

        for i in range(len(self.vehicles)):
            # Update shadowing with spatial correlation
            self.cellular_Shadowing[i] = self.A2GChannels.get_shadowing(
                self.delta_distance[i], self.cellular_Shadowing[i])

            # Expected path loss
            if self.A2GChannels.path_loss_model == '3GPP_UMa':
                pathloss = self.A2GChannels.get_path_loss(self.vehicles[i].position)[0, 0]
            else:
                p_los = self.A2GChannels.get_los_probability(self.vehicles[i].position)
                pl_los  = self.A2GChannels.get_path_loss(self.vehicles[i].position, is_los=True)[0]
                pl_nlos = self.A2GChannels.get_path_loss(self.vehicles[i].position, is_los=False)[0]
                pathloss = p_los * pl_los + (1.0 - p_los) * pl_nlos
            self.cellular_pathloss[i] = pathloss - self.path_loss_offset_dB

        # Combine path loss and shadowing
        self.cellular_channels_abs = (self.cellular_pathloss +
                                      np.repeat(self.cellular_Shadowing[:, np.newaxis], self.n_RB, axis=1))

    def renew_BS_channels_fastfading(self):
        """
        Renew fast fading channel: same as backup, superimpose Rayleigh fast fading (dB) on slow channel.
        channel_with_fast = channel_abs - 20*log10(|h|), h ~ CN(0,1)/sqrt(2).
        """
        shape = self.cellular_channels_abs.shape
        h = (np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape)) / math.sqrt(2)
        self.cellular_channels_with_fastfading = self.cellular_channels_abs - 20 * np.log10(np.abs(h) + 1e-20)

    def Compute_Performance_Reward_Failure(self, actions_all=None, IS_PPO=False):
        """
        Compute performance metrics for failure case (low SINR or collision)
        Args:
            actions_all: Optional, action array [n_veh, 2 or 3]
            IS_PPO: bool or int, whether using PPO/DDPG format
        Returns:
            (cellular_Rate, cellular_SINR, SE, EE, semantic_accuracy, semantic_EE, collisions,
             semantic_Rate, semantic_SE)
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
        # Use maximum power for failure case (index 0 = max power, e.g. 27 dBm)
        max_power_dB = self.cellular_power_dB_List[0]
        for i in range(len(self.vehicles)):
            transmit_power[i] = max_power_dB
        
        # Compute basic metrics (similar to success case but with low power)
        channel_choice = np.zeros(self.n_RB)
        cellular_SINR = np.zeros(self.n_Veh)
        cellular_Signals = np.zeros([self.n_Veh, self.n_RB])
        
        for i in range(len(self.vehicles)):
            l = actions[i]
            channel_choice[l] += 1
            cellular_Signals[i, l] = 10 ** ((transmit_power[i] -
                                            self.cellular_channels_with_fastfading[i, l] + 
                                            self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

        # Compute SINR (assuming interference)
        for i in range(len(self.vehicles)):
            l = actions[i]
            # Low SINR for failure case
            interference = self.sig2[l] * 10  # High interference
            cellular_SINR[i] = cellular_Signals[i, l] / (interference + self.sig2[l])
        
        # Calculate semantic accuracy (will be low due to low SINR)
        semantic_accuracy = np.zeros(self.n_Veh)
        semantic_EE = np.zeros(self.n_Veh)
        semantic_Rate = np.zeros(self.n_Veh)  # Semantic Rate R^s
        semantic_SE = np.zeros(self.n_Veh)    # Semantic Spectral Efficiency
        transmission_power_linear = 10 ** (transmit_power / 10)

        for i in range(len(self.vehicles)):
            semantic_accuracy[i] = self.compute_semantic_accuracy(rho[i], cellular_SINR[i])
            
            # Semantic Rate: S = W * (I/K) * L * semantic_accuracy * H_image (图片语义熵 3.22)
            l = actions[i]
            W = self.BW[l]
            LK = rho[i]  # I/K = rho (linear model)
            H_img = self.semantic_entropy_image
            semantic_Rate[i] = W * LK * semantic_accuracy[i] * H_img
            semantic_SE[i] = LK * semantic_accuracy[i] * H_img
            
            # Semantic Energy Efficiency = Semantic Rate / (Transmission_Power + Circuit_Power)
            # SEE = S / (P_tx + P_circuit)
            total_power = transmission_power_linear[i] + self.circuit_power
            if total_power > 0 :
                semantic_EE[i] = semantic_Rate[i] / total_power
            else:
                semantic_EE[i] = 0.0
        
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
                semantic_accuracy, semantic_EE, collisions,
                semantic_Rate, semantic_SE)

    def compute_semantic_accuracy(self, rho, sinr):
        """
        Compute task similarity (semantic accuracy) Q from fitted model:
        Q = A_peak * (1 - exp(-xi*rho)) / (1 + exp(-zeta*(gamma_dB - gamma0))) + b
        Fitted parameters (SNR -20~60 dB): A_peak=0.7128, xi=10, zeta=0.2313, gamma0=0, b=0.3249

        Args:
            rho: Compression ratio [0, 1]
            sinr: SNR/SINR in linear scale (converted to dB internally)

        Returns:
            Q: Task similarity in [b, A_peak + b]
        """
        rho = np.clip(rho, 0.0, 1.0)
        # Convert linear SNR to dB for sigmoid (model parameters are fitted in dB domain)
        sinr_linear = float(sinr)
        gamma_dB = 10.0 * np.log10(max(sinr_linear, 1e-10))
        num = 1.0 - np.exp(-self.task_sim_xi * rho)
        den = 1.0 + np.exp(-self.task_sim_zeta * (gamma_dB - self.task_sim_gamma0))
        den = np.clip(den, 1e-8, None)
        Q = self.task_sim_A_peak * (num / den) + self.task_sim_b
        Q = np.clip(Q, self.task_sim_b, self.task_sim_A_peak + self.task_sim_b)
        return Q
    
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
                transmit_power = actions_all[:, 1].astype(np.float32)
                power_selection = None
            else:
                power_selection = actions_all[:, 1].astype(int)
                transmit_power = np.array([self.cellular_power_dB_List[int(p)] for p in power_selection], dtype=np.float32)
        else:
            # New format: [RB, Power, rho]
            actions = actions_all[:, 0].astype(int)
            rho = np.clip(actions_all[:, 2], 0.0, 1.0)  # Compression ratio [0, 1]
            if is_ppo_mode:
                transmit_power = actions_all[:, 1].astype(np.float32)
                power_selection = None
            else:
                power_selection = actions_all[:, 1].astype(int)
                transmit_power = np.array([self.cellular_power_dB_List[int(p)] for p in power_selection], dtype=np.float32)

        # ------------ Compute signal, interference and SINR (with collision penalty) --------------------
        channel_choice = np.zeros(self.n_RB)
        cellular_Interference = np.zeros(self.n_RB)
        cellular_SINR = np.zeros(self.n_Veh)
        cellular_Signals = np.zeros([self.n_Veh, self.n_RB])

        for i in range(len(self.vehicles)):
            l = actions[i]
            channel_choice[l] += 1

            # Compute signal power (linear scale)
            signal_power_linear = 10 ** ((transmit_power[i] -
                                         self.cellular_channels_with_fastfading[i, l] +
                                         self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
            cellular_Signals[i, l] = signal_power_linear
            cellular_Interference[l] += signal_power_linear

        # Collision: multiple UAVs on same RB
        collisions = np.zeros(self.n_Veh)
        for i in range(len(self.vehicles)):
            if channel_choice[actions[i]] > 1:
                collisions[i] = 1

        # SINR = signal / (interference from other UAVs on same RB + noise)
        for i in range(len(self.vehicles)):
            l = actions[i]
            interference_total = cellular_Interference[l] - cellular_Signals[i, l] + self.sig2[l]
            cellular_SINR[i] = cellular_Signals[i, l] / interference_total

        # Success: no collision AND SINR above threshold
        self.success = np.zeros([self.n_Veh])
        for i in range(len(self.vehicles)):
            if collisions[i] == 0 and cellular_SINR[i] > self.SINR_THRESHOLD_LINEAR:
                self.success[i] = 1
        
        self.channel_choice = channel_choice
        
        # ------------ Compute Semantic Communication Metrics --------------------
        semantic_accuracy = np.zeros(self.n_Veh)
        semantic_EE = np.zeros(self.n_Veh)
        semantic_Rate = np.zeros(self.n_Veh)  # Semantic Rate R^s
        semantic_SE = np.zeros(self.n_Veh)    # Semantic Spectral Efficiency
        transmission_power_linear = np.zeros(self.n_Veh)
        
        for i in range(len(self.vehicles)):
            # Convert power from dB to linear scale
            transmission_power_linear[i] = 10 ** (transmit_power[i] / 10)
            
            # Compute semantic accuracy (mAP) based on compression ratio and SINR
            semantic_accuracy[i] = self.compute_semantic_accuracy(rho[i], cellular_SINR[i])
            
            # Semantic Rate: R^s = W * (I/K) * L * semantic_accuracy * H_image (图片语义熵 3.22)
            l = actions[i]
            W = self.BW[l]  # Bandwidth in MHz
            IK = rho[i]  # I/K = rho (linear model)
            L_factor = 1.0
            H_img = self.semantic_entropy_image
            semantic_Rate[i] = W * IK * L_factor * semantic_accuracy[i] * H_img
            semantic_SE[i] = IK * L_factor * semantic_accuracy[i] * H_img
            total_power = transmission_power_linear[i] + self.circuit_power
            semantic_EE[i] = semantic_Rate[i] / total_power if total_power > 0 else 0.0
            
            # Store for state representation (state uses semantic_accuracy)
            self.semantic_accuracy[i] = semantic_accuracy[i]
            self.semantic_EE[i] = semantic_EE[i]
            self.semantic_Rate[i] = semantic_Rate[i]
            self.rho_current[i] = rho[i]
            self.cellular_SINR[i] = cellular_SINR[i]
        
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
                semantic_accuracy, semantic_EE, collisions,
                semantic_Rate, semantic_SE)

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
        cellular_Rate = np.zeros(self.n_Veh)
        SE = np.zeros(self.n_Veh)
        cellular_Signals = np.zeros([self.n_Veh, self.n_RB])
        cellular_Interference = np.zeros(self.n_RB)  # cellular interference index RB
        for i in range(len(self.vehicles)):
            for l in range(self.n_RB):
                if IS_PPO:
                    cellular_Signals[i, l] = 10 ** ((transmit_power -
                                               self.cellular_channels_with_fastfading[
                                                   i, l] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                else:
                    cellular_Signals[i, l] = 10 ** ((self.cellular_power_dB_List[power_selection[i]] -
                                               self.cellular_channels_with_fastfading[
                                                   i, l] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                if IS_PPO:
                    if (l == actions):
                        channel_choice[actions] += 1
                        cellular_Interference[actions] += 10 ** ((transmit_power -
                                                                   self.cellular_channels_with_fastfading[
                                                                       i, actions]
                                                                   + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                else:
                    if (l == actions[i]):
                        channel_choice[actions[i]] += 1
                        cellular_Interference[actions[i]] += 10 ** ((self.cellular_power_dB_List[
                                                                       power_selection[i]] -
                                                                   self.cellular_channels_with_fastfading[
                                                                       i, actions[i]]
                                                                   + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

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
        # No inter-UAV interference: use noise only
        self.cellular_Interference = self.sig2
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
            reward: Semantic Energy Efficiency average (per UAV) for meta training
        """
        action_temp = actions.copy()
        # 兼容处理：确保IS_PPO参数被正确处理
        is_ppo_mode = bool(IS_PPO) if IS_PPO is not None else False
        
        # Compute performance metrics (includes semantic communication metrics)
        results = self.Compute_Performance_Reward_Train(action_temp, is_ppo_mode)
        (cellular_Rate, cellular_SINR, SE, EE, 
         semantic_accuracy, semantic_EE, collisions,
         semantic_Rate, semantic_SE) = results
        
        # Get failure case metrics (low SINR scenario)
        failure_results = self.Compute_Performance_Reward_Failure(action_temp, is_ppo_mode)
        (_, _, _, _, _, failure_semantic_EE, _, _, _) = failure_results
        
        Semantic_EE_sum = 0.0
        
        # Initialize reward to avoid UnboundLocalError
        reward = 0.0
        
        # 使用 semantic_accuracy 门限
        training_accuracy_threshold = 0.5  # semantic_accuracy 门限
        
        for i in range(len(self.success)):
            if (self.success[i] == 1) and (semantic_accuracy[i] > training_accuracy_threshold):
                Semantic_EE_sum += semantic_EE[i]
            elif (self.success[i] == 1):
                Semantic_EE_sum += failure_semantic_EE[i]
            else:
                # Case 3: Failure (collision) - apply penalty
                Semantic_EE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
                reward = Semantic_EE_sum  # Set reward before break
                break
            
            reward = Semantic_EE_sum
        
        return reward / self.n_Veh

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
         semantic_accuracy, semantic_EE, collisions,
         semantic_Rate, semantic_SE) = results
        
        # Get failure case metrics (low SINR scenario)
        failure_results = self.Compute_Performance_Reward_Failure(action_temp, is_ppo_mode)
        (_, _, _, _, _, failure_semantic_EE, _, _, _) = failure_results
        
        # Use Semantic-EE as reward (only optimize SEE)
        Semantic_EE_sum = 0.0
        
        training_accuracy_threshold = 0.5  # semantic_accuracy 门限
        
        self.similarity_success = np.zeros([self.n_Veh])
        reward = 0.0
        
        for i in range(len(self.success)):
            if (self.success[i] == 1) and (semantic_accuracy[i] > training_accuracy_threshold):
                self.similarity_success[i] = 1
                Semantic_EE_sum += semantic_EE[i]
            elif (self.success[i] == 1):
                Semantic_EE_sum += failure_semantic_EE[i]
            else:
                # Case 3: Failure (collision) - apply penalty
                Semantic_EE_sum = (np.sum(self.success) - self.n_Veh) / self.n_Veh
                break

        reward = Semantic_EE_sum / self.n_Veh
        
        return reward

    def act_for_testing(self, actions, IS_PPO):
        action_temp = actions.copy()
        results = self.Compute_Performance_Reward_Train(action_temp, IS_PPO)
        cellular_Rate, cellular_SINR, SE, EE, _, _, _, _, _ = results
        for i in range(len(cellular_Rate)):
            if not (self.success[i] == 1 and cellular_SINR[i] > self.SINR_THRESHOLD_LINEAR):
                SE[i], EE[i] = 0.0, 0.0
        return SE, EE, cellular_Rate

    def act_for_testing_marl(self, actions, IS_PPO):
        action_temp = actions.copy()
        results = self.Compute_Performance_Reward_Train(action_temp, IS_PPO)
        cellular_Rate, cellular_SINR, SE, EE, _, _, _, _, _ = results
        done = all(self.success[i] == 1 and cellular_SINR[i] > self.SINR_THRESHOLD_LINEAR for i in range(len(cellular_Rate)))
        if not done:
            SE[:], EE[:] = 0.0, 0.0
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
        # self.renew_neighbor()  # 暂不计算邻居/邻接，省算力；启用 GAT 时再恢复
        self.renew_BS_channel()
        self.renew_BS_channels_fastfading()



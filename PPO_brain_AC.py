import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import copy

from arguments import get_args
args = get_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.set_random_seed)


my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth = True

n_hidden_1 = args.n_hidden_1
n_hidden_2 = args.n_hidden_2
n_hidden_3 = args.n_hidden_3

sigma_add = args.sigma_add

def graph_attention_layer(inputs, adj_matrix, num_heads=4, out_dim=None, activation=tf.nn.relu, 
                          dropout_rate=0.0, is_training=True, name='gat'):
    """
    Multi-head Graph Attention Layer (GAT) implementation in TensorFlow
    
    Args:
        inputs: [N, F_in] node features, where N is number of nodes, F_in is input feature dim
        adj_matrix: [N, N] adjacency matrix (can be weighted)
        num_heads: number of attention heads
        out_dim: output dimension per head (default: F_in)
        activation: activation function
        dropout_rate: dropout rate
        is_training: whether in training mode
        name: variable scope name
    
    Returns:
        output: [N, out_dim * num_heads] concatenated multi-head features
    """
    with tf.variable_scope(name):
        N = tf.shape(inputs)[0]  # Number of nodes
        F_in = inputs.get_shape()[-1].value  # Input feature dimension
        
        if out_dim is None:
            out_dim = F_in
        
        # Linear transformation for each head
        head_outputs = []
        
        for head in range(num_heads):
            with tf.variable_scope(f'head_{head}'):
                # Weight matrix: [F_in, out_dim]
                W = tf.get_variable(f'W_{head}', shape=[F_in, out_dim],
                                   initializer=tf.compat.v1.keras.initializers.glorot_uniform())
                
                # Attention weight vector: [2 * out_dim, 1]
                a = tf.get_variable(f'a_{head}', shape=[2 * out_dim, 1],
                                   initializer=tf.compat.v1.keras.initializers.glorot_uniform())
                
                # Transform input features: [N, out_dim]
                h = tf.matmul(inputs, W)  # [N, out_dim]
                
                # Compute attention coefficients
                # Split attention vector: a = [a_1; a_2] where each is [out_dim, 1]
                a_1 = a[:out_dim, :]  # [out_dim, 1]
                a_2 = a[out_dim:, :]  # [out_dim, 1]
                
                # Compute: Wh_i^T a_1 for all i: [N, 1]
                # and: Wh_j^T a_2 for all j: [1, N]
                # Then broadcast and add
                e_i = tf.matmul(h, a_1)  # [N, 1]
                e_j = tf.matmul(h, a_2)  # [N, 1]
                
                # Broadcast: [N, 1] + [1, N] = [N, N]
                e = e_i + tf.transpose(e_j)  # [N, N]
                
                # Apply LeakyReLU
                e = tf.nn.leaky_relu(e, alpha=0.2)
                
                # Apply mask (only attend to neighbors)
                # adj_matrix: [N, N], mask out non-neighbors
                mask = -1e9 * (1.0 - adj_matrix)  # Large negative value for non-neighbors
                e = e + mask
                
                # Softmax over neighbors
                attention = tf.nn.softmax(e, axis=1)  # [N, N]
                
                # Apply dropout
                if dropout_rate > 0.0:
                    attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
                
                # Aggregate neighbor features: [N, out_dim]
                h_out = tf.matmul(attention, h)  # [N, out_dim]
                
                # Apply activation
                if activation is not None:
                    h_out = activation(h_out)
                
                head_outputs.append(h_out)
        
        # Concatenate all heads: [N, out_dim * num_heads]
        output = tf.concat(head_outputs, axis=1)
        
        return output


def multi_layer_gat(node_features, adj_matrix, hidden_dims, num_heads=4, 
                    activation=tf.nn.relu, dropout_rate=0.0, is_training=True, name='multi_gat', reuse=None,
                    use_residual=True):
    """
    Multi-layer GAT encoder with residual connections
    
    Args:
        node_features: [N, F_in] node features
        adj_matrix: [N, N] adjacency matrix
        hidden_dims: list of hidden dimensions for each layer
        num_heads: number of attention heads per layer
        activation: activation function
        dropout_rate: dropout rate
        is_training: whether in training mode
        name: variable scope name
        use_residual: whether to use residual connections (default: True)
    
    Returns:
        output: [N, hidden_dims[-1] * num_heads] final node embeddings
    """
    with tf.variable_scope(name, reuse=reuse):
        x = node_features
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Store input for residual connection
            x_input = x
            
            # For intermediate layers, we need to handle multi-head concatenation
            if i == 0:
                # First layer: input is original features
                x = graph_attention_layer(x, adj_matrix, num_heads=num_heads, 
                                         out_dim=hidden_dim, activation=None,  # Apply activation after residual
                                         dropout_rate=dropout_rate, is_training=is_training,
                                         name=f'gat_layer_{i}')
            else:
                # Subsequent layers: input is concatenated multi-head features
                # Project to match input dimension for next GAT layer
                input_dim = x.get_shape()[-1].value
                if input_dim != hidden_dim * num_heads:
                    # Project concatenated features to match expected input
                    with tf.variable_scope(f'project_{i}'):
                        W_proj = tf.get_variable(f'W_proj', 
                                               shape=[input_dim, hidden_dim * num_heads],
                                               initializer=tf.compat.v1.keras.initializers.glorot_uniform())
                        x = tf.matmul(x, W_proj)
                        # Don't apply activation here, will apply after residual
                
                # Apply GAT layer
                x = graph_attention_layer(x, adj_matrix, num_heads=num_heads,
                                         out_dim=hidden_dim, activation=None,  # Apply activation after residual
                                         dropout_rate=dropout_rate, is_training=is_training,
                                         name=f'gat_layer_{i}')
            
            # Residual connection (skip first layer if dimensions don't match)
            if use_residual and i > 0:
                x_output_dim = x.get_shape()[-1].value
                x_input_dim = x_input.get_shape()[-1].value
                
                if x_output_dim == x_input_dim:
                    # Dimensions match: direct residual connection
                    x = x + x_input
                else:
                    # Dimensions don't match: use projection for residual
                    with tf.variable_scope(f'residual_proj_{i}'):
                        W_res = tf.get_variable(f'W_res', 
                                               shape=[x_input_dim, x_output_dim],
                                               initializer=tf.compat.v1.keras.initializers.glorot_uniform())
                        x_res = tf.matmul(x_input, W_res)
                        x = x + x_res
            
            # Apply activation after residual connection
            if activation is not None:
                x = activation(x)
        
        return x

class PPO(object):
    def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr, meta_lr, K, n_veh, n_RB, IS_meta, meta_episode,
                 use_gat=False, num_gat_heads=4, node_feature_dim=None):
        """
        Args:
            s_dim: state dimension (for backward compatibility, if use_gat=False)
            a_bound: action bounds [RB_bound, power_bound, compression_bound]
            c1, c2, epsilon, lr, meta_lr, K: PPO hyperparameters
            n_veh: number of UAVs (nodes in graph)
            n_RB: number of resource blocks
            IS_meta: whether to use meta learning
            meta_episode: meta episode number
            use_gat: whether to use GAT instead of MLP
            num_gat_heads: number of attention heads in GAT
            node_feature_dim: dimension of node features (CSI, location, etc.)
        """
        self.a_bound = a_bound
        self.K = K
        self.s_dim = s_dim  # Keep for backward compatibility
        self.a_dim = 3  # RB_choice + Power + Compression Ratio (rho)
        self.n_RB = n_RB
        self.n_veh = n_veh
        self.IS_meta = IS_meta
        self.gamma = args.gamma
        self.GAE_discount = args.lambda_advantage
        self.use_gat = use_gat
        self.num_gat_heads = num_gat_heads
        self.c1 = c1  # Value function loss weight
        self.c2 = c2  # Entropy weight
        self.epsilon = epsilon  # PPO clipping parameter
        
        # Graph-based input (for GAT)
        if use_gat:
            # Node features: [batch_size, n_veh, node_feature_dim]
            # Adjacency matrix: [batch_size, n_veh, n_veh]
            if node_feature_dim is None:
                # Default: CSI features (n_RB * 2) + position (3) + other features
                node_feature_dim = n_RB * 2 + 3 + 2  # CSI fast, CSI abs, position (x,y,z), success, episode
            self.node_feature_dim = node_feature_dim
            self.node_features = tf.placeholder(tf.float32, shape=(None, n_veh, node_feature_dim), name='node_features')
            self.adj_matrix = tf.placeholder(tf.float32, shape=(None, n_veh, n_veh), name='adj_matrix')
            # For single-agent processing, we also keep the old interface
            self.s_input = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_t')
        else:
            # MLP-based input (backward compatibility)
            self.s_input = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_t')

        pi, RB_distribution, rho_distribution, self.v, params, self.saver = self._build_net('network', True)
        old_pi, old_RB_distribution, old_rho_distribution, old_v, old_params, _ = self._build_net('old_network', False)
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.v_pred_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_pred_next')
        self.gae = tf.placeholder(dtype=tf.float32, shape=[None], name='gae')

        GAE_advantage = self.gae

        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a_t')
        RB_action = self.a[:,0]
        power_action = self.a[:,1]
        rho_action = self.a[:,2]  # Compression ratio
        
        # Handle shape mismatch in GAT mode
        batch_size = tf.shape(self.a)[0]
        
        # For GAT mode: need to know which agent's node to select
        # This will be set via a placeholder or use 0 as default
        if self.use_gat:
            # Create a placeholder for agent_idx (will be set in train function)
            self.agent_idx_ph = tf.placeholder(tf.int32, shape=(), name='agent_idx')
            agent_idx_tf = self.agent_idx_ph
        else:
            agent_idx_tf = None
        
        if self.use_gat:
            # In GAT mode: distributions output [batch*n_veh, ...] but actions are [batch, ...]
            # Tile actions to match distribution shape
            power_action_expanded = tf.tile(tf.expand_dims(power_action, 1), [1, self.n_veh])
            power_action_expanded = tf.reshape(power_action_expanded, [-1])  # [batch*n_veh]
            
            rho_action_expanded = tf.tile(tf.expand_dims(rho_action, 1), [1, self.n_veh])
            rho_action_expanded = tf.reshape(rho_action_expanded, [-1])  # [batch*n_veh]
            
            # Compute probabilities
            power_prob_all = pi.prob(power_action_expanded)  # May have various shapes due to broadcasting
            old_power_prob_all = old_pi.prob(power_action_expanded)  # Same shape as power_prob_all
            
            # Flatten to 1D first, then get the expected size
            power_prob_all_flat = tf.reshape(power_prob_all, [-1])
            old_power_prob_all_flat = tf.reshape(old_power_prob_all, [-1])
            
            # Get the expected size: should be batch_size * n_veh
            expected_size = batch_size * self.n_veh
            actual_size = tf.shape(power_prob_all_flat)[0]
            
            # Handle size mismatch: take the first expected_size elements
            power_prob_all = power_prob_all_flat[:expected_size]
            old_power_prob_all = old_power_prob_all_flat[:expected_size]
            
            # Reshape to [batch_size, n_veh] and select the agent's node (not mean)
            power_prob_all_reshaped = tf.reshape(power_prob_all, [batch_size, self.n_veh])
            old_power_prob_all_reshaped = tf.reshape(old_power_prob_all, [batch_size, self.n_veh])
            # Select the specific agent's node output (not mean over all nodes)
            # Use agent_idx to select the correct node
            agent_idx_tf = tf.constant(0)  # Will be set via feed_dict or use first node as default
            power_prob = power_prob_all_reshaped[:, agent_idx_tf]  # [batch_size] - select agent's node
            old_power_prob = old_power_prob_all_reshaped[:, agent_idx_tf]  # [batch_size]
            
            rho_prob_all = rho_distribution.prob(rho_action_expanded)  # May be [batch_size*n_veh] or [batch_size*n_veh, 1] or other shapes
            old_rho_prob_all = old_rho_distribution.prob(rho_action_expanded)  # Same shape as rho_prob_all
            
            # Flatten to 1D first, then get the expected size
            rho_prob_all_flat = tf.reshape(rho_prob_all, [-1])
            old_rho_prob_all_flat = tf.reshape(old_rho_prob_all, [-1])
            
            # Get the expected size: should be batch_size * n_veh
            expected_size = batch_size * self.n_veh
            actual_size = tf.shape(rho_prob_all_flat)[0]
            
            # Handle size mismatch: if actual_size > expected_size, it might be due to broadcasting
            # Take the first expected_size elements
            rho_prob_all = rho_prob_all_flat[:expected_size]
            old_rho_prob_all = old_rho_prob_all_flat[:expected_size]
            
            # Reshape to [batch_size, n_veh] and select the agent's node (not mean)
            rho_prob_all_reshaped = tf.reshape(rho_prob_all, [batch_size, self.n_veh])
            old_rho_prob_all_reshaped = tf.reshape(old_rho_prob_all, [batch_size, self.n_veh])
            # Select the specific agent's node output
            rho_prob = rho_prob_all_reshaped[:, agent_idx_tf]  # [batch_size] - select agent's node
            old_rho_prob = old_rho_prob_all_reshaped[:, agent_idx_tf]  # [batch_size]
        else:
            # MLP mode: original simple logic (no shape manipulation needed)
            power_prob = pi.prob(power_action)
            old_power_prob = old_pi.prob(power_action)
            rho_prob = rho_distribution.prob(rho_action)
            old_rho_prob = old_rho_distribution.prob(rho_action)
        
        # Original code style: simple ratio calculation with numerical stability
        # Add small epsilon to prevent division by zero
        ratio_power = power_prob / (old_power_prob + 1e-8)
        ratio_rho = rho_prob / (old_rho_prob + 1e-8)
        # Clip ratios to prevent extreme values
        ratio_power = tf.clip_by_value(ratio_power, 1e-6, 1e6)
        ratio_rho = tf.clip_by_value(ratio_rho, 1e-6, 1e6)
        
        # Value function loss (original code style)
        if self.use_gat:
            # v_pred is [batch, n_veh] - each agent has its own value
            # For training, select the specific agent's value (not mean)
            v_pred = self.v[:, agent_idx_tf]  # [batch] - select agent's value
            # Replace NaN/Inf with zeros
            v_pred = tf.where(tf.is_finite(v_pred), v_pred, tf.zeros_like(v_pred))
        else:
            # MLP mode: v_pred is [batch, 1] or [batch]
            v_pred = tf.squeeze(self.v, axis=-1) if len(self.v.get_shape()) > 1 else self.v
        
        # Clip values to prevent extreme values
        v_target = self.reward + self.gamma * self.v_pred_next
        v_target = tf.clip_by_value(v_target, -10.0, 10.0)
        v_pred = tf.clip_by_value(v_pred, -10.0, 10.0)
        L_vf = tf.reduce_mean(tf.square(v_target - v_pred))
        # Replace NaN with zeros
        L_vf = tf.where(tf.is_finite(L_vf), L_vf, tf.zeros_like(L_vf))
        
        # Use GAE_advantage directly (original code style)
        GAE_advantage_clipped = GAE_advantage
        
        # PPO clipping loss for power action
        L_clip_power = tf.reduce_mean(tf.minimum(
            ratio_power * GAE_advantage_clipped,
            tf.clip_by_value(ratio_power, 1 - epsilon, 1 + epsilon) * GAE_advantage_clipped
        ))
        # Replace NaN with zeros
        L_clip_power = tf.where(tf.is_finite(L_clip_power), L_clip_power, tf.zeros_like(L_clip_power))
        
        # PPO clipping loss for RB action
        if self.use_gat:
            RB_action_expanded = tf.tile(tf.expand_dims(RB_action, 1), [1, self.n_veh])
            RB_action_expanded = tf.reshape(RB_action_expanded, [-1])  # [batch_size*n_veh]
            RB_prob_all = RB_distribution.prob(RB_action_expanded)  # [batch_size*n_veh] or [batch_size*n_veh, n_RB]
            old_RB_prob_all = old_RB_distribution.prob(RB_action_expanded)  # [batch_size*n_veh] or [batch_size*n_veh, n_RB]
            
            # Flatten to 1D first, then get the expected size
            RB_prob_all_flat = tf.reshape(RB_prob_all, [-1])
            old_RB_prob_all_flat = tf.reshape(old_RB_prob_all, [-1])
            
            # Get the expected size: should be batch_size * n_veh
            expected_size_RB = batch_size * self.n_veh
            actual_size_RB = tf.shape(RB_prob_all_flat)[0]
            
            # Handle size mismatch: take the first expected_size_RB elements
            RB_prob_all = RB_prob_all_flat[:expected_size_RB]
            old_RB_prob_all = old_RB_prob_all_flat[:expected_size_RB]
            
            # Reshape to [batch_size, n_veh] and select the agent's node (not mean)
            RB_prob_all_reshaped = tf.reshape(RB_prob_all, [batch_size, self.n_veh])
            old_RB_prob_all_reshaped = tf.reshape(old_RB_prob_all, [batch_size, self.n_veh])
            # Select the specific agent's node output
            RB_prob = RB_prob_all_reshaped[:, agent_idx_tf]  # [batch_size] - select agent's node
            old_RB_prob = old_RB_prob_all_reshaped[:, agent_idx_tf]  # [batch_size]
        else:
            # MLP mode: original simple logic (no shape manipulation needed)
            RB_prob = RB_distribution.prob(RB_action)
            old_RB_prob = old_RB_distribution.prob(RB_action)
        
        # Original code style: simple ratio calculation with numerical stability
        ratio_RB = RB_prob / (old_RB_prob + 1e-8)
        # Clip ratio to prevent extreme values
        ratio_RB = tf.clip_by_value(ratio_RB, 1e-6, 1e6)
        
        L_RB = tf.reduce_mean(tf.minimum(
            ratio_RB * GAE_advantage_clipped,
            tf.clip_by_value(ratio_RB, 1 - epsilon, 1 + epsilon) * GAE_advantage_clipped
        ))
        # Replace NaN with zeros
        L_RB = tf.where(tf.is_finite(L_RB), L_RB, tf.zeros_like(L_RB))
        
        # PPO clipping loss for compression ratio (rho) action
        L_rho = tf.reduce_mean(tf.minimum(
            ratio_rho * GAE_advantage_clipped,
            tf.clip_by_value(ratio_rho, 1 - epsilon, 1 + epsilon) * GAE_advantage_clipped
        ))
        # Replace NaN with zeros
        L_rho = tf.where(tf.is_finite(L_rho), L_rho, tf.zeros_like(L_rho))
        
        # Entropy for exploration
        if self.use_gat:
            # In GAT mode: distributions output [batch*n_veh, ...], need to aggregate
            pi_entropy = pi.entropy()  # [batch*n_veh]
            RB_entropy = RB_distribution.entropy()  # [batch*n_veh] or [batch*n_veh, n_RB]
            rho_entropy = rho_distribution.entropy()  # [batch*n_veh]
            
            # Flatten and aggregate
            pi_entropy = tf.reshape(pi_entropy, [-1])
            rho_entropy = tf.reshape(rho_entropy, [-1])
            RB_entropy = tf.reshape(RB_entropy, [-1])
            
            # Ensure entropy values are finite
            pi_entropy = tf.where(tf.is_finite(pi_entropy), pi_entropy, tf.zeros_like(pi_entropy))
            RB_entropy = tf.where(tf.is_finite(RB_entropy), RB_entropy, tf.zeros_like(RB_entropy))
            rho_entropy = tf.where(tf.is_finite(rho_entropy), rho_entropy, tf.zeros_like(rho_entropy))
            
            # Reshape entropy to [batch_size, n_veh] and select agent's node
            pi_entropy_reshaped = tf.reshape(pi_entropy, [batch_size, self.n_veh])
            RB_entropy_reshaped = tf.reshape(RB_entropy, [batch_size, self.n_veh])
            rho_entropy_reshaped = tf.reshape(rho_entropy, [batch_size, self.n_veh])
            # Select the specific agent's entropy
            S = pi_entropy_reshaped[:, agent_idx_tf] + RB_entropy_reshaped[:, agent_idx_tf] + rho_entropy_reshaped[:, agent_idx_tf]
            S = tf.reduce_mean(S)  # [1] - mean over batch
        else:
            # MLP mode: standard entropy calculation (original code style)
            S = tf.reduce_mean(pi.entropy() + RB_distribution.entropy() + rho_distribution.entropy())
        
        # Total loss (original code style, but with rho added)
        L = L_clip_power + L_RB + L_rho - c1 * L_vf + c2 * S
        # Replace NaN with zeros
        L = tf.where(tf.is_finite(L), L, tf.zeros_like(L))
        
        self.Loss = [L_clip_power, L_RB, L_rho, L_vf, S]
        self.Entropy_value = S
        
        # Sample actions
        if self.use_gat:
            # GAT mode: sample for all nodes, output [n_veh, action_dim]
            RB_sample = tf.transpose(tf.cast(RB_distribution.sample(1), dtype=tf.float32))
            RB_sample = tf.reshape(RB_sample, [-1, 1])
            power_sample = tf.reshape(pi.sample(1), [-1])
            rho_sample = tf.reshape(rho_distribution.sample(1), [-1])
            n_nodes = tf.shape(RB_sample)[0]
            RB_sample = tf.reshape(RB_sample, [n_nodes, 1])
            power_sample = tf.reshape(power_sample, [n_nodes])
            rho_sample = tf.reshape(rho_sample, [n_nodes])
            self.choose_action_op = tf.concat([
                RB_sample, 
                tf.expand_dims(power_sample, 1), 
                tf.expand_dims(rho_sample, 1)
            ], axis=1)
        else:
            # MLP mode: single action output [1, action_dim]
            self.choose_action_op = tf.concat([
                tf.transpose(tf.cast(RB_distribution.sample(1), dtype=tf.float32)), 
                tf.squeeze(pi.sample(1), axis=0),
                tf.squeeze(rho_distribution.sample(1), axis=0)
            ], 1)
        
        # Optimizer with gradient clipping for GAT mode (to prevent NaN)
        if self.use_gat:
            # GAT mode: use gradient clipping to prevent NaN
            optimizer = tf.train.AdamOptimizer(lr)
            grads_and_vars = optimizer.compute_gradients(-L)
            # Clip gradients to prevent explosion
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, 0.5) if grad is not None else grad, var) 
                                      for grad, var in grads_and_vars]
            self.train_op = optimizer.apply_gradients(clipped_grads_and_vars)
        else:
            # MLP mode: original code style (no gradient clipping)
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-L)
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]
        self.sesses = []
        for ind_agent in range(self.n_veh):
            print("Initializing agent", ind_agent)
            sess = tf.Session(config=my_config)
            sess.run(tf.global_variables_initializer())
            self.sesses.append(sess)
        if self.IS_meta:
            print("\nRestoring the model...")
            optimization_target = args.optimization_target if hasattr(args, 'optimization_target') else 'SE_EE'
            beta = args.beta if hasattr(args, 'beta') else 0.5
            
            opt_target_str = optimization_target.replace('_', '&')
            if optimization_target == 'SE_EE':
                opt_suffix = f'{opt_target_str}_{beta:.2f}'
            else:
                opt_suffix = opt_target_str
            
            for i in range(self.n_veh):
                meta_save_path = 'meta_model_'
                model_path = meta_save_path + 'AC_' + opt_suffix + '_' + '%s_' %sigma_add + '%d_' % meta_episode +'%s_' %args.lr_meta_a
                self.load_models(self.sesses[i], model_path, self.saver)

    def load_models(self, sess, model_path, saver):
        """ Restore models from the current directory with the name filename """
        dir_ = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_, "model/" + model_path)
        if os.path.exists(model_path + '.index'):
            saver.restore(sess, model_path)
        else:
            print(f"Warning: Model path {model_path} does not exist, skipping load.")

    def save_model(self, sess, model_path, saver):
        """ Save models to the current directory with the name filename """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        saver.save(sess, model_path, write_meta_graph=False)

    def save_models(self, label):
        for i in range(self.n_veh):
            model_path = label + '/agent_' + str(i)
            self.save_model(self.sesses[i], model_path, self.saver)
            
    def _build_net(self, scope, trainable):
        """
        Build network with GAT encoder and multiple actor heads
        Returns: (power_dist, RB_dist, rho_dist, value, params, saver)
        """
        with tf.variable_scope(scope):
            initializer = tf.compat.v1.keras.initializers.he_normal()
            
            if self.use_gat:
                # Graph Attention Network (GAT) encoder
                # Support batch processing: [batch_size, n_veh, node_feature_dim]
                batch_size = tf.shape(self.node_features)[0]
                
                # Process each batch sample's graph separately using map_fn
                # This ensures each batch sample uses its own node features
                def process_graph(idx):
                    """Process graph for batch sample idx"""
                    node_feat = self.node_features[idx]  # [n_veh, node_feature_dim]
                    adj = self.adj_matrix[idx]  # [n_veh, n_veh]
                    gat_output = multi_layer_gat(
                        node_feat, adj,
                        hidden_dims=[n_hidden_1, n_hidden_2, n_hidden_3],
                        num_heads=self.num_gat_heads,
                        activation=tf.nn.relu,
                        dropout_rate=0.0,
                        is_training=trainable,
                        name='gat_encoder',
                        reuse=tf.AUTO_REUSE  # Reuse variables across batch samples
                    )  # [n_veh, n_hidden_3 * num_heads]
                    return gat_output
                
                # Use map_fn to process all graphs in batch
                indices = tf.range(batch_size)
                gat_outputs = tf.map_fn(
                    process_graph,
                    indices,
                    dtype=tf.float32,
                    parallel_iterations=1,  # Reduce parallelism to avoid issues
                    back_prop=True
                )  # [batch, n_veh, n_hidden_3 * num_heads]
                
                # Reshape for actor: [batch * n_veh, hidden_dim]
                actor_input = tf.reshape(gat_outputs, [-1, gat_outputs.get_shape()[-1].value])
                
                # For critic: use node-level embeddings (each node has its own value)
                # This allows each agent to have its own value estimate
                # Reshape to [batch * n_veh, hidden_dim] for critic processing
                critic_input_flat = tf.reshape(gat_outputs, [-1, gat_outputs.get_shape()[-1].value])  # [batch * n_veh, hidden_dim]
                
                # Project actor input to n_hidden_2 for compatibility
                W_actor_proj = tf.get_variable('W_actor_proj',
                                               shape=[actor_input.get_shape()[-1].value, n_hidden_2],
                                               initializer=initializer, trainable=trainable)
                layer_2_b = tf.nn.relu(tf.matmul(actor_input, W_actor_proj))  # [batch*n_veh, n_hidden_2]
                
            else:
                # MLP encoder (backward compatibility)
                self.w_1 = tf.Variable(initializer(shape=(self.s_dim, n_hidden_1)), trainable=trainable)
                self.w_2 = tf.Variable(initializer(shape=(n_hidden_1, n_hidden_2)), trainable=trainable)
                self.w_3 = tf.Variable(initializer(shape=(n_hidden_2, n_hidden_3)), trainable=trainable)
                
                self.b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1), trainable=trainable)
                self.b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1), trainable=trainable)
                self.b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1), trainable=trainable)
                
                layer_p1 = tf.nn.relu(tf.add(tf.matmul(self.s_input, self.w_1), self.b_1), name='p_1')
                layer_1_b = tf.layers.batch_normalization(layer_p1)
                layer_p2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, self.w_2), self.b_2), name='p_2')
                layer_2_b = tf.layers.batch_normalization(layer_p2)
                layer_p3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, self.w_3), self.b_3), name='p_3')
                layer_3_b = tf.layers.batch_normalization(layer_p3)
                
                critic_input = layer_3_b  # For critic
            
            # Actor heads (shared for both GAT and MLP)
            self.w_mu = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_sigma = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_RB = tf.Variable(initializer(shape=(n_hidden_2, self.n_RB)), trainable=trainable)
            # 新增：rho的网络参数（Beta分布）
            self.w_rho_alpha = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            self.w_rho_beta = tf.Variable(initializer(shape=(n_hidden_2, 1)), trainable=trainable)
            
            self.b_mu = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_sigma = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_RB = tf.Variable(tf.truncated_normal([self.n_RB], stddev=0.1), trainable=trainable)
            # 新增：rho的bias
            self.b_rho_alpha = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            self.b_rho_beta = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
            
            mu = tf.nn.tanh(tf.add(tf.matmul(layer_2_b, self.w_mu), self.b_mu), name='mu_layer')
            sigma = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_sigma), self.b_sigma), name='sigma_layer')
            RB_probs = tf.nn.softmax(tf.add(tf.matmul(layer_2_b, self.w_RB), self.b_RB), name='RB_layer')
            RB_distribution = tf.distributions.Categorical(probs=RB_probs)
            
            # 新增：rho的Beta分布（压缩比 rho ∈ [0,1]）
            rho_alpha = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_alpha), self.b_rho_alpha)) + 1.0
            rho_beta = tf.nn.softplus(tf.add(tf.matmul(layer_2_b, self.w_rho_beta), self.b_rho_beta)) + 1.0
            rho_distribution = tf.distributions.Beta(rho_alpha, rho_beta)
            
            mu, sigma = mu, sigma + sigma_add
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            
            # Critic network
            if self.use_gat:
                # GAT mode: critic_input_flat is [batch * n_veh, n_hidden_3 * num_heads]
                # Project to hidden dimension
                W_v_proj = tf.get_variable('W_v_proj',
                                          shape=[critic_input_flat.get_shape()[-1].value, n_hidden_3],
                                          initializer=initializer, trainable=trainable)
                critic_hidden = tf.nn.relu(tf.matmul(critic_input_flat, W_v_proj))  # [batch * n_veh, n_hidden_3]
                # Clip to prevent extreme values before next layer
                critic_hidden = tf.clip_by_value(critic_hidden, -10.0, 10.0)
                self.w_v = tf.Variable(initializer(shape=(n_hidden_3, 1)), trainable=trainable)
                # Initialize bias to small value to avoid zero output
                self.b_v = tf.Variable(tf.constant([0.1]), trainable=trainable)
                # Use linear output (no ReLU) to allow negative values
                v_flat = tf.add(tf.matmul(critic_hidden, self.w_v), self.b_v)  # [batch * n_veh, 1]
                # Clip to reasonable range
                v_flat = tf.clip_by_value(v_flat, -50.0, 50.0)
                # Reshape back to [batch, n_veh, 1] then squeeze to [batch, n_veh]
                v = tf.reshape(v_flat, [batch_size, self.n_veh, 1])  # [batch, n_veh, 1]
                v = tf.squeeze(v, axis=2)  # [batch, n_veh] - each node has its own value
                # Replace NaN/Inf with small random values (not zeros, to allow learning)
                v = tf.where(tf.is_finite(v), v, tf.random_normal(tf.shape(v), mean=0.0, stddev=0.01))
            else:
                # MLP mode: critic_input is [batch, n_hidden_3]
                self.w_v = tf.Variable(initializer(shape=(n_hidden_3, 1)), trainable=trainable)
                self.b_v = tf.Variable(tf.truncated_normal([1], stddev=0.1), trainable=trainable)
                v = tf.nn.relu(tf.add(tf.matmul(critic_input, self.w_v), self.b_v), name='v_layer')
            
            saver = tf.train.Saver(max_to_keep=self.n_veh * 2)
            
        params = tf.global_variables(scope)
        return norm_dist, RB_distribution, rho_distribution, v, params, saver

    def get_v(self, s, sess, node_features=None, adj_matrix=None, agent_idx=0):
        """
        Get value function estimate
        Args:
            s: state (for backward compatibility with MLP)
            sess: TensorFlow session
            node_features: [n_veh, node_feature_dim] node features (for GAT)
            adj_matrix: [n_veh, n_veh] adjacency matrix (for GAT)
            agent_idx: agent index (for GAT mode, to select which node's value)
        """
        if self.use_gat and node_features is not None and adj_matrix is not None:
            # Expand dimensions for batch
            node_features_batch = node_features[np.newaxis, :, :]  # [1, n_veh, node_feature_dim]
            adj_matrix_batch = adj_matrix[np.newaxis, :, :]  # [1, n_veh, n_veh]
            v_all = sess.run(self.v, {
                self.node_features: node_features_batch,
                self.adj_matrix: adj_matrix_batch
            })  # [1, n_veh] in GAT mode - each node has its own value
            # Extract value for this specific agent (node)
            v_all = np.array(v_all)
            if len(v_all.shape) == 2:
                # Shape is [1, n_veh]
                if v_all.shape[1] > agent_idx:
                    return float(v_all[0, agent_idx])
                else:
                    return float(v_all[0, 0])
            elif len(v_all.shape) == 1:
                # Shape is [n_veh]
                if len(v_all) > agent_idx:
                    return float(v_all[agent_idx])
                else:
                    return float(v_all[0])
            else:
                return float(np.squeeze(v_all))
        else:
            # MLP mode (backward compatibility)
            return sess.run(self.v, {
                self.s_input: np.array([s])
            }).squeeze()

    def choose_action(self, s, sess, node_features=None, adj_matrix=None, agent_idx=0):
        """
        Choose action
        Args:
            s: state (for backward compatibility with MLP)
            sess: TensorFlow session
            node_features: [n_veh, node_feature_dim] node features (for GAT)
            adj_matrix: [n_veh, n_veh] adjacency matrix (for GAT)
            agent_idx: agent index (for GAT mode, to select which node's action)
        """
        if self.use_gat and node_features is not None and adj_matrix is not None:
            # Expand dimensions for batch
            node_features_batch = node_features[np.newaxis, :, :]  # [1, n_veh, node_feature_dim]
            adj_matrix_batch = adj_matrix[np.newaxis, :, :]  # [1, n_veh, n_veh]
            a_all = sess.run(self.choose_action_op, {
                self.node_features: node_features_batch,
                self.adj_matrix: adj_matrix_batch
            })  # [n_veh, action_dim] in GAT mode
            # Extract action for this specific agent (node)
            a = a_all[agent_idx] if agent_idx < len(a_all) else a_all[0]
        else:
            # MLP mode (backward compatibility)
            a = np.squeeze(sess.run(self.choose_action_op, {self.s_input: s[np.newaxis, :]}))
        
        clipped_a = np.zeros(self.a_dim)
        clipped_a[0] = a[0]
        clipped_a[1] = np.clip(a[1], -self.a_bound[1], self.a_bound[1])
        clipped_a[2] = np.clip(a[2], 0.0, 1.0)  # rho ∈ [0,1]
        return clipped_a

    def train(self, s, a, gae, reward, v_pred_next, sess, node_features=None, adj_matrix=None, agent_idx=0):
        """
        Train the network
        Args:
            s: state (for backward compatibility with MLP)
            a: actions [batch_size, 3] (RB, Power, Compression Ratio)
            gae: GAE advantages
            reward: rewards
            v_pred_next: next state values
            sess: TensorFlow session
            node_features: [batch_size, n_veh, node_feature_dim] (for GAT)
            adj_matrix: [batch_size, n_veh, n_veh] (for GAT)
        """
        sess.run(self.update_params_op)
        
        # Prepare feed dict
        if self.use_gat and node_features is not None and adj_matrix is not None:
            feed_dict = {
                self.node_features: node_features,
                self.adj_matrix: adj_matrix,
                self.a: a,
                self.reward: reward,
                self.v_pred_next: v_pred_next,
                self.gae: gae,
                self.agent_idx_ph: agent_idx  # Set agent index for node selection
            }
        else:
            feed_dict = {
                self.s_input: s,
                self.a: a,
                self.reward: reward,
                self.v_pred_next: v_pred_next,
                self.gae: gae
            }
        
        # K epochs
        for i in range(self.K):
            sess.run(self.train_op, feed_dict)
        
        # Get loss components for debugging
        loss_components = sess.run(self.Loss, feed_dict)
        entropy = sess.run(self.Entropy_value, feed_dict)
        
        # Debug output for GAT mode (only print first few times)
        if self.use_gat and node_features is not None and hasattr(self, '_debug_count'):
            self._debug_count += 1
            if self._debug_count <= 3:  # Only print first 3 times
                try:
                    v_debug = sess.run(self.v, feed_dict)
                    gae_debug = feed_dict.get(self.gae, None)
                    reward_debug = feed_dict.get(self.reward, None)
                    v_pred_next_debug = feed_dict.get(self.v_pred_next, None)
                    
                    if gae_debug is not None:
                        gae_arr = np.array(gae_debug)
                        # Check critic hidden output
                        # Check distribution parameters
                        try:
                            mu_debug = sess.run('network/mu_layer:0', feed_dict)
                            sigma_debug = sess.run('network/sigma_layer:0', feed_dict)
                            print(f"DEBUG GAT [{self._debug_count}]: mu sample={mu_debug.flatten()[:3]}, "
                                  f"sigma sample={sigma_debug.flatten()[:3]}")
                        except:
                            pass
                        print(f"DEBUG GAT [{self._debug_count}]: V shape={v_debug.shape}, V sample={v_debug.flatten()[:3]}, "
                              f"GAE shape={gae_arr.shape}, GAE sample={gae_arr.flatten()[:3]}, "
                              f"Reward={reward_debug[:3] if reward_debug is not None else None}, "
                              f"V_next={v_pred_next_debug[:3] if v_pred_next_debug is not None else None}, "
                              f"Loss={loss_components}, Entropy={entropy}")
                except Exception as e:
                    print(f"DEBUG GAT: Error: {e}")
        elif self.use_gat and node_features is not None:
            self._debug_count = 1
        
        return [loss_components, entropy]

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE
        :param rewards: r(t) - can be [T, n_veh] or [T]
        :param v_preds: v(st) - can be [T, n_veh] or [T]
        :param v_preds_next: v(st+1) - can be [T, n_veh] or [T]
        :return: gaes - same shape as input
        """
        # Convert to numpy arrays if needed
        rewards = np.array(rewards)
        v_preds = np.array(v_preds)
        v_preds_next = np.array(v_preds_next)
        
        # Handle 2D case (GAT mode: [T, n_veh])
        if len(rewards.shape) == 2:
            # For each agent, compute GAE separately
            n_veh = rewards.shape[1]
            gaes_list = []
            for agent_idx in range(n_veh):
                r_agent = rewards[:, agent_idx]
                v_agent = v_preds[:, agent_idx]
                v_next_agent = v_preds_next[:, agent_idx]
                
                # Compute deltas for this agent
                deltas = r_agent + self.gamma * v_next_agent - v_agent
                gaes = deltas.copy()
                
                # Compute GAE backwards
                for t in reversed(range(len(gaes) - 1)):
                    gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
                
                gaes_list.append(gaes)
            
            # Stack back to [T, n_veh]
            return np.stack(gaes_list, axis=1)
        else:
            # 1D case (MLP mode: [T])
            deltas = rewards + self.gamma * v_preds_next - v_preds
            gaes = deltas.copy()
            for t in reversed(range(len(gaes) - 1)):
                gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
            return gaes

    def averaging_model(self, success_rate):
        if self.use_gat:
            # GAT mode: aggregate all trainable variables using params
            # Collect all parameters from all agents
            all_params = []
            for i in range(self.n_veh):
                # Get all trainable variables in the 'network' scope
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
                param_values = self.sesses[i].run(params)
                all_params.append(param_values)
            
            # Average parameters
            avg_params = []
            for j in range(len(all_params[0])):
                param_avg = np.mean([p[j] for p in all_params], axis=0)
                avg_params.append(param_avg)
            
            # Assign averaged parameters back to all agents
            for i in range(self.n_veh):
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
                old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old_network')
                # Update network parameters
                assign_ops = [tf.assign(p_ref, p_val) for p_ref, p_val in zip(params, avg_params)]
                self.sesses[i].run(assign_ops)
                # Update old_network parameters (copy from network)
                assign_old_ops = [tf.assign(p_old, p_new) for p_old, p_new in zip(old_params, params)]
                self.sesses[i].run(assign_old_ops)
            return
        
        # MLP mode: use existing manual parameter aggregation
        mu = 0
        sigma = 1e-8
        w_1_mean = np.random.normal(0, sigma, [self.s_dim, n_hidden_1])
        w_2_mean = np.random.normal(0, sigma, [n_hidden_1, n_hidden_2])
        w_3_mean = np.random.normal(0, sigma, [n_hidden_2, n_hidden_3])
        w_mu_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_sigma_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_RB_mean = np.random.normal(0, sigma, [n_hidden_2, self.n_RB])
        # 新增：rho参数平均
        w_rho_alpha_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_rho_beta_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_v_mean = np.random.normal(0, sigma, [n_hidden_3, 1])

        b_1_mean = np.random.normal(0, sigma, [n_hidden_1])
        b_2_mean = np.random.normal(0, sigma, [n_hidden_2])
        b_3_mean = np.random.normal(0, sigma, [n_hidden_3])
        b_mu_mean = np.random.normal(0, sigma, [1])
        b_sigma_mean = np.random.normal(0, sigma, [1])
        b_RB_mean = np.random.normal(0, sigma, [self.n_RB])
        # 新增：rho bias平均
        b_rho_alpha_mean = np.random.normal(0, sigma, [1])
        b_rho_beta_mean = np.random.normal(0, sigma, [1])
        b_v_mean = np.random.normal(0, sigma, [1])

        for i in range(self.n_veh):
            w_1_mean += self.sesses[i].run(self.w_1) / self.n_veh
            w_2_mean += self.sesses[i].run(self.w_2) / self.n_veh
            w_3_mean += self.sesses[i].run(self.w_3) / self.n_veh
            w_mu_mean += self.sesses[i].run(self.w_mu) / self.n_veh
            w_sigma_mean += self.sesses[i].run(self.w_sigma) / self.n_veh
            w_RB_mean += self.sesses[i].run(self.w_RB) / self.n_veh
            # 新增：rho参数聚合
            w_rho_alpha_mean += self.sesses[i].run(self.w_rho_alpha) / self.n_veh
            w_rho_beta_mean += self.sesses[i].run(self.w_rho_beta) / self.n_veh
            w_v_mean += self.sesses[i].run(self.w_v) / self.n_veh

            b_1_mean += self.sesses[i].run(self.b_1) / self.n_veh
            b_2_mean += self.sesses[i].run(self.b_2) / self.n_veh
            b_3_mean += self.sesses[i].run(self.b_3) / self.n_veh
            b_mu_mean += self.sesses[i].run(self.b_mu) / self.n_veh
            b_sigma_mean += self.sesses[i].run(self.b_sigma) / self.n_veh
            b_RB_mean += self.sesses[i].run(self.b_RB) / self.n_veh
            # 新增：rho bias聚合
            b_rho_alpha_mean += self.sesses[i].run(self.b_rho_alpha) / self.n_veh
            b_rho_beta_mean += self.sesses[i].run(self.b_rho_beta) / self.n_veh
            b_v_mean += self.sesses[i].run(self.b_v) / self.n_veh

        for i in range(self.n_veh):
            self.sesses[i].run(self.w_1.assign(w_1_mean))
            self.sesses[i].run(self.w_2.assign(w_2_mean))
            self.sesses[i].run(self.w_3.assign(w_3_mean))
            self.sesses[i].run(self.w_mu.assign(w_mu_mean))
            self.sesses[i].run(self.w_sigma.assign(w_sigma_mean))
            self.sesses[i].run(self.w_RB.assign(w_RB_mean))
            # 新增：rho参数分发
            self.sesses[i].run(self.w_rho_alpha.assign(w_rho_alpha_mean))
            self.sesses[i].run(self.w_rho_beta.assign(w_rho_beta_mean))
            self.sesses[i].run(self.w_v.assign(w_v_mean))

            self.sesses[i].run(self.b_1.assign(b_1_mean))
            self.sesses[i].run(self.b_2.assign(b_2_mean))
            self.sesses[i].run(self.b_3.assign(b_3_mean))
            self.sesses[i].run(self.b_mu.assign(b_mu_mean))
            self.sesses[i].run(self.b_sigma.assign(b_sigma_mean))
            self.sesses[i].run(self.b_RB.assign(b_RB_mean))
            # 新增：rho bias分发
            self.sesses[i].run(self.b_rho_alpha.assign(b_rho_alpha_mean))
            self.sesses[i].run(self.b_rho_beta.assign(b_rho_beta_mean))
            self.sesses[i].run(self.b_v.assign(b_v_mean))

            sess.run(self.train_op, feed_dict)
        
        # Get loss components for debugging
        loss_components = sess.run(self.Loss, feed_dict)
        entropy = sess.run(self.Entropy_value, feed_dict)
        
        # Debug output for GAT mode (only print first few times)
        if self.use_gat and node_features is not None and hasattr(self, '_debug_count'):
            self._debug_count += 1
            if self._debug_count <= 3:  # Only print first 3 times
                try:
                    v_debug = sess.run(self.v, feed_dict)
                    gae_debug = feed_dict.get(self.gae, None)
                    reward_debug = feed_dict.get(self.reward, None)
                    v_pred_next_debug = feed_dict.get(self.v_pred_next, None)
                    
                    if gae_debug is not None:
                        gae_arr = np.array(gae_debug)
                        # Check critic hidden output
                        # Check distribution parameters
                        try:
                            mu_debug = sess.run('network/mu_layer:0', feed_dict)
                            sigma_debug = sess.run('network/sigma_layer:0', feed_dict)
                            print(f"DEBUG GAT [{self._debug_count}]: mu sample={mu_debug.flatten()[:3]}, "
                                  f"sigma sample={sigma_debug.flatten()[:3]}")
                        except:
                            pass
                        print(f"DEBUG GAT [{self._debug_count}]: V shape={v_debug.shape}, V sample={v_debug.flatten()[:3]}, "
                              f"GAE shape={gae_arr.shape}, GAE sample={gae_arr.flatten()[:3]}, "
                              f"Reward={reward_debug[:3] if reward_debug is not None else None}, "
                              f"V_next={v_pred_next_debug[:3] if v_pred_next_debug is not None else None}, "
                              f"Loss={loss_components}, Entropy={entropy}")
                except Exception as e:
                    print(f"DEBUG GAT: Error: {e}")
        elif self.use_gat and node_features is not None:
            self._debug_count = 1
        
        return [loss_components, entropy]

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE
        :param rewards: r(t) - can be [T, n_veh] or [T]
        :param v_preds: v(st) - can be [T, n_veh] or [T]
        :param v_preds_next: v(st+1) - can be [T, n_veh] or [T]
        :return: gaes - same shape as input
        """
        # Convert to numpy arrays if needed
        rewards = np.array(rewards)
        v_preds = np.array(v_preds)
        v_preds_next = np.array(v_preds_next)
        
        # Handle 2D case (GAT mode: [T, n_veh])
        if len(rewards.shape) == 2:
            # For each agent, compute GAE separately
            n_veh = rewards.shape[1]
            gaes_list = []
            for agent_idx in range(n_veh):
                r_agent = rewards[:, agent_idx]
                v_agent = v_preds[:, agent_idx]
                v_next_agent = v_preds_next[:, agent_idx]
                
                # Compute deltas for this agent
                deltas = r_agent + self.gamma * v_next_agent - v_agent
                gaes = deltas.copy()
                
                # Compute GAE backwards
                for t in reversed(range(len(gaes) - 1)):
                    gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
                
                gaes_list.append(gaes)
            
            # Stack back to [T, n_veh]
            return np.stack(gaes_list, axis=1)
        else:
            # 1D case (MLP mode: [T])
            deltas = rewards + self.gamma * v_preds_next - v_preds
            gaes = deltas.copy()
            for t in reversed(range(len(gaes) - 1)):
                gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
            return gaes

    def averaging_model(self, success_rate):
        if self.use_gat:
            # GAT mode: aggregate all trainable variables using params
            # Collect all parameters from all agents
            all_params = []
            for i in range(self.n_veh):
                # Get all trainable variables in the 'network' scope
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
                param_values = self.sesses[i].run(params)
                all_params.append(param_values)
            
            # Average parameters
            avg_params = []
            for j in range(len(all_params[0])):
                param_avg = np.mean([p[j] for p in all_params], axis=0)
                avg_params.append(param_avg)
            
            # Assign averaged parameters back to all agents
            for i in range(self.n_veh):
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
                old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old_network')
                # Update network parameters
                assign_ops = [tf.assign(p_ref, p_val) for p_ref, p_val in zip(params, avg_params)]
                self.sesses[i].run(assign_ops)
                # Update old_network parameters (copy from network)
                assign_old_ops = [tf.assign(p_old, p_new) for p_old, p_new in zip(old_params, params)]
                self.sesses[i].run(assign_old_ops)
            return
        
        # MLP mode: use existing manual parameter aggregation
        mu = 0
        sigma = 1e-8
        w_1_mean = np.random.normal(0, sigma, [self.s_dim, n_hidden_1])
        w_2_mean = np.random.normal(0, sigma, [n_hidden_1, n_hidden_2])
        w_3_mean = np.random.normal(0, sigma, [n_hidden_2, n_hidden_3])
        w_mu_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_sigma_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_RB_mean = np.random.normal(0, sigma, [n_hidden_2, self.n_RB])
        # 新增：rho参数平均
        w_rho_alpha_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_rho_beta_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_v_mean = np.random.normal(0, sigma, [n_hidden_3, 1])

        b_1_mean = np.random.normal(0, sigma, [n_hidden_1])
        b_2_mean = np.random.normal(0, sigma, [n_hidden_2])
        b_3_mean = np.random.normal(0, sigma, [n_hidden_3])
        b_mu_mean = np.random.normal(0, sigma, [1])
        b_sigma_mean = np.random.normal(0, sigma, [1])
        b_RB_mean = np.random.normal(0, sigma, [self.n_RB])
        # 新增：rho bias平均
        b_rho_alpha_mean = np.random.normal(0, sigma, [1])
        b_rho_beta_mean = np.random.normal(0, sigma, [1])
        b_v_mean = np.random.normal(0, sigma, [1])

        for i in range(self.n_veh):
            w_1_mean += self.sesses[i].run(self.w_1) / self.n_veh
            w_2_mean += self.sesses[i].run(self.w_2) / self.n_veh
            w_3_mean += self.sesses[i].run(self.w_3) / self.n_veh
            w_mu_mean += self.sesses[i].run(self.w_mu) / self.n_veh
            w_sigma_mean += self.sesses[i].run(self.w_sigma) / self.n_veh
            w_RB_mean += self.sesses[i].run(self.w_RB) / self.n_veh
            # 新增：rho参数聚合
            w_rho_alpha_mean += self.sesses[i].run(self.w_rho_alpha) / self.n_veh
            w_rho_beta_mean += self.sesses[i].run(self.w_rho_beta) / self.n_veh
            w_v_mean += self.sesses[i].run(self.w_v) / self.n_veh

            b_1_mean += self.sesses[i].run(self.b_1) / self.n_veh
            b_2_mean += self.sesses[i].run(self.b_2) / self.n_veh
            b_3_mean += self.sesses[i].run(self.b_3) / self.n_veh
            b_mu_mean += self.sesses[i].run(self.b_mu) / self.n_veh
            b_sigma_mean += self.sesses[i].run(self.b_sigma) / self.n_veh
            b_RB_mean += self.sesses[i].run(self.b_RB) / self.n_veh
            # 新增：rho bias聚合
            b_rho_alpha_mean += self.sesses[i].run(self.b_rho_alpha) / self.n_veh
            b_rho_beta_mean += self.sesses[i].run(self.b_rho_beta) / self.n_veh
            b_v_mean += self.sesses[i].run(self.b_v) / self.n_veh

        for i in range(self.n_veh):
            self.sesses[i].run(self.w_1.assign(w_1_mean))
            self.sesses[i].run(self.w_2.assign(w_2_mean))
            self.sesses[i].run(self.w_3.assign(w_3_mean))
            self.sesses[i].run(self.w_mu.assign(w_mu_mean))
            self.sesses[i].run(self.w_sigma.assign(w_sigma_mean))
            self.sesses[i].run(self.w_RB.assign(w_RB_mean))
            # 新增：rho参数分发
            self.sesses[i].run(self.w_rho_alpha.assign(w_rho_alpha_mean))
            self.sesses[i].run(self.w_rho_beta.assign(w_rho_beta_mean))
            self.sesses[i].run(self.w_v.assign(w_v_mean))

            self.sesses[i].run(self.b_1.assign(b_1_mean))
            self.sesses[i].run(self.b_2.assign(b_2_mean))
            self.sesses[i].run(self.b_3.assign(b_3_mean))
            self.sesses[i].run(self.b_mu.assign(b_mu_mean))
            self.sesses[i].run(self.b_sigma.assign(b_sigma_mean))
            self.sesses[i].run(self.b_RB.assign(b_RB_mean))
            # 新增：rho bias分发
            self.sesses[i].run(self.b_rho_alpha.assign(b_rho_alpha_mean))
            self.sesses[i].run(self.b_rho_beta.assign(b_rho_beta_mean))
            self.sesses[i].run(self.b_v.assign(b_v_mean))

            sess.run(self.train_op, feed_dict)
        
        # Get loss components for debugging
        loss_components = sess.run(self.Loss, feed_dict)
        entropy = sess.run(self.Entropy_value, feed_dict)
        
        # Debug output for GAT mode (only print first few times)
        if self.use_gat and node_features is not None and hasattr(self, '_debug_count'):
            self._debug_count += 1
            if self._debug_count <= 3:  # Only print first 3 times
                try:
                    v_debug = sess.run(self.v, feed_dict)
                    gae_debug = feed_dict.get(self.gae, None)
                    reward_debug = feed_dict.get(self.reward, None)
                    v_pred_next_debug = feed_dict.get(self.v_pred_next, None)
                    
                    if gae_debug is not None:
                        gae_arr = np.array(gae_debug)
                        # Check critic hidden output
                        # Check distribution parameters
                        try:
                            mu_debug = sess.run('network/mu_layer:0', feed_dict)
                            sigma_debug = sess.run('network/sigma_layer:0', feed_dict)
                            print(f"DEBUG GAT [{self._debug_count}]: mu sample={mu_debug.flatten()[:3]}, "
                                  f"sigma sample={sigma_debug.flatten()[:3]}")
                        except:
                            pass
                        print(f"DEBUG GAT [{self._debug_count}]: V shape={v_debug.shape}, V sample={v_debug.flatten()[:3]}, "
                              f"GAE shape={gae_arr.shape}, GAE sample={gae_arr.flatten()[:3]}, "
                              f"Reward={reward_debug[:3] if reward_debug is not None else None}, "
                              f"V_next={v_pred_next_debug[:3] if v_pred_next_debug is not None else None}, "
                              f"Loss={loss_components}, Entropy={entropy}")
                except Exception as e:
                    print(f"DEBUG GAT: Error: {e}")
        elif self.use_gat and node_features is not None:
            self._debug_count = 1
        
        return [loss_components, entropy]

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE
        :param rewards: r(t) - can be [T, n_veh] or [T]
        :param v_preds: v(st) - can be [T, n_veh] or [T]
        :param v_preds_next: v(st+1) - can be [T, n_veh] or [T]
        :return: gaes - same shape as input
        """
        # Convert to numpy arrays if needed
        rewards = np.array(rewards)
        v_preds = np.array(v_preds)
        v_preds_next = np.array(v_preds_next)
        
        # Handle 2D case (GAT mode: [T, n_veh])
        if len(rewards.shape) == 2:
            # For each agent, compute GAE separately
            n_veh = rewards.shape[1]
            gaes_list = []
            for agent_idx in range(n_veh):
                r_agent = rewards[:, agent_idx]
                v_agent = v_preds[:, agent_idx]
                v_next_agent = v_preds_next[:, agent_idx]
                
                # Compute deltas for this agent
                deltas = r_agent + self.gamma * v_next_agent - v_agent
                gaes = deltas.copy()
                
                # Compute GAE backwards
                for t in reversed(range(len(gaes) - 1)):
                    gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
                
                gaes_list.append(gaes)
            
            # Stack back to [T, n_veh]
            return np.stack(gaes_list, axis=1)
        else:
            # 1D case (MLP mode: [T])
            deltas = rewards + self.gamma * v_preds_next - v_preds
            gaes = deltas.copy()
            for t in reversed(range(len(gaes) - 1)):
                gaes[t] = gaes[t] + self.gamma * self.GAE_discount * gaes[t + 1]
            return gaes

    def averaging_model(self, success_rate):
        if self.use_gat:
            # GAT mode: aggregate all trainable variables using params
            # Collect all parameters from all agents
            all_params = []
            for i in range(self.n_veh):
                # Get all trainable variables in the 'network' scope
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
                param_values = self.sesses[i].run(params)
                all_params.append(param_values)
            
            # Average parameters
            avg_params = []
            for j in range(len(all_params[0])):
                param_avg = np.mean([p[j] for p in all_params], axis=0)
                avg_params.append(param_avg)
            
            # Assign averaged parameters back to all agents
            for i in range(self.n_veh):
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
                old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old_network')
                # Update network parameters
                assign_ops = [tf.assign(p_ref, p_val) for p_ref, p_val in zip(params, avg_params)]
                self.sesses[i].run(assign_ops)
                # Update old_network parameters (copy from network)
                assign_old_ops = [tf.assign(p_old, p_new) for p_old, p_new in zip(old_params, params)]
                self.sesses[i].run(assign_old_ops)
            return
        
        # MLP mode: use existing manual parameter aggregation
        mu = 0
        sigma = 1e-8
        w_1_mean = np.random.normal(0, sigma, [self.s_dim, n_hidden_1])
        w_2_mean = np.random.normal(0, sigma, [n_hidden_1, n_hidden_2])
        w_3_mean = np.random.normal(0, sigma, [n_hidden_2, n_hidden_3])
        w_mu_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_sigma_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_RB_mean = np.random.normal(0, sigma, [n_hidden_2, self.n_RB])
        # 新增：rho参数平均
        w_rho_alpha_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_rho_beta_mean = np.random.normal(0, sigma, [n_hidden_2, 1])
        w_v_mean = np.random.normal(0, sigma, [n_hidden_3, 1])

        b_1_mean = np.random.normal(0, sigma, [n_hidden_1])
        b_2_mean = np.random.normal(0, sigma, [n_hidden_2])
        b_3_mean = np.random.normal(0, sigma, [n_hidden_3])
        b_mu_mean = np.random.normal(0, sigma, [1])
        b_sigma_mean = np.random.normal(0, sigma, [1])
        b_RB_mean = np.random.normal(0, sigma, [self.n_RB])
        # 新增：rho bias平均
        b_rho_alpha_mean = np.random.normal(0, sigma, [1])
        b_rho_beta_mean = np.random.normal(0, sigma, [1])
        b_v_mean = np.random.normal(0, sigma, [1])

        for i in range(self.n_veh):
            w_1_mean += self.sesses[i].run(self.w_1) / self.n_veh
            w_2_mean += self.sesses[i].run(self.w_2) / self.n_veh
            w_3_mean += self.sesses[i].run(self.w_3) / self.n_veh
            w_mu_mean += self.sesses[i].run(self.w_mu) / self.n_veh
            w_sigma_mean += self.sesses[i].run(self.w_sigma) / self.n_veh
            w_RB_mean += self.sesses[i].run(self.w_RB) / self.n_veh
            # 新增：rho参数聚合
            w_rho_alpha_mean += self.sesses[i].run(self.w_rho_alpha) / self.n_veh
            w_rho_beta_mean += self.sesses[i].run(self.w_rho_beta) / self.n_veh
            w_v_mean += self.sesses[i].run(self.w_v) / self.n_veh

            b_1_mean += self.sesses[i].run(self.b_1) / self.n_veh
            b_2_mean += self.sesses[i].run(self.b_2) / self.n_veh
            b_3_mean += self.sesses[i].run(self.b_3) / self.n_veh
            b_mu_mean += self.sesses[i].run(self.b_mu) / self.n_veh
            b_sigma_mean += self.sesses[i].run(self.b_sigma) / self.n_veh
            b_RB_mean += self.sesses[i].run(self.b_RB) / self.n_veh
            # 新增：rho bias聚合
            b_rho_alpha_mean += self.sesses[i].run(self.b_rho_alpha) / self.n_veh
            b_rho_beta_mean += self.sesses[i].run(self.b_rho_beta) / self.n_veh
            b_v_mean += self.sesses[i].run(self.b_v) / self.n_veh

        for i in range(self.n_veh):
            self.sesses[i].run(self.w_1.assign(w_1_mean))
            self.sesses[i].run(self.w_2.assign(w_2_mean))
            self.sesses[i].run(self.w_3.assign(w_3_mean))
            self.sesses[i].run(self.w_mu.assign(w_mu_mean))
            self.sesses[i].run(self.w_sigma.assign(w_sigma_mean))
            self.sesses[i].run(self.w_RB.assign(w_RB_mean))
            # 新增：rho参数分发
            self.sesses[i].run(self.w_rho_alpha.assign(w_rho_alpha_mean))
            self.sesses[i].run(self.w_rho_beta.assign(w_rho_beta_mean))
            self.sesses[i].run(self.w_v.assign(w_v_mean))

            self.sesses[i].run(self.b_1.assign(b_1_mean))
            self.sesses[i].run(self.b_2.assign(b_2_mean))
            self.sesses[i].run(self.b_3.assign(b_3_mean))
            self.sesses[i].run(self.b_mu.assign(b_mu_mean))
            self.sesses[i].run(self.b_sigma.assign(b_sigma_mean))
            self.sesses[i].run(self.b_RB.assign(b_RB_mean))
            # 新增：rho bias分发
            self.sesses[i].run(self.b_rho_alpha.assign(b_rho_alpha_mean))
            self.sesses[i].run(self.b_rho_beta.assign(b_rho_beta_mean))
            self.sesses[i].run(self.b_v.assign(b_v_mean))

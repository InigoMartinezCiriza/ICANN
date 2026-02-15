import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
from modifiedGRU import ModifiedGRU
from sparse_constraint import SparseConstraint

# --------------------------
# Actor (Policy) Network
# --------------------------
@tf.keras.utils.register_keras_serializable()
class ActorModel(Model):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 prob_connection=1.0,
                 layer_type='GRU_standard',
                 alpha = 0.1,
                 **kwargs):
        """
        Builds an Actor network: Dense -> Hidden layers -> Dense.

        Args:
        -----------
        input_size: int
            Dimensionality of the observations (states).
        hidden_size: int
            Number of units for the hidden layers.
        output_size: int
            Number of actions.
        num_layers: int
            Number of hidden layers.
        prob_connection: float
            Connection probability for weights in hidden layers.
        layer_type: str
            Type of the hidden layers, can be 'GRU_standard', 'GRU_modified' or 'Dense' units.
        alpha: float
            A factor representing the discretized time step over the time constant.
        """

        super(ActorModel, self).__init__(**kwargs)
        self.num_layers      = num_layers
        self.layer_type      = layer_type
        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.output_size     = output_size
        self.prob_connection = prob_connection
        self.alpha           = alpha

        # --- Helper function to create constraint instance ---
        def _create_new_constraint_if_sparse(prob):
            return SparseConstraint(prob) if prob < 1.0 else None

        # 1. Initial Dense Layer
        self.input_fc = layers.Dense(input_size, activation='relu', name='actor_input_dense')

        # 2. Hidden Layers
        self.hidden_layers = []
        for i in range(num_layers):
            layer_name_base = f'actor_{layer_type.lower().replace("_","")}_{i}'

            # --- Create constraint instances specifically for this hidden layer ---
            kernel_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)
            recurrent_sparse_constraint_hidden = None

            if 'GRU' in layer_type:
                recurrent_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)

            if layer_type == 'GRU_modified':
                self.hidden_layers.append(ModifiedGRU(hidden_size,
                                                      return_sequences=True, return_state=True,
                                                      kernel_constraint=kernel_sparse_constraint_hidden,
                                                      recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                      name=layer_name_base,
                                                      alpha = alpha))
            elif layer_type == 'GRU_standard':
                self.hidden_layers.append(layers.GRU(hidden_size,
                                                     activation='tanh', recurrent_activation='sigmoid',
                                                     return_sequences=True, return_state=True,
                                                     kernel_constraint=kernel_sparse_constraint_hidden,
                                                     recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                     name=layer_name_base))
            elif layer_type == 'Dense':
                self.hidden_layers.append(layers.Dense(hidden_size, activation='relu',
                                                       kernel_constraint=kernel_sparse_constraint_hidden,
                                                       name=layer_name_base))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # 3. Final Dense Layer
        self.fc_out = layers.Dense(output_size, activation='softmax', name='actor_output_dense')

    def call(self, inputs, hidden_states=None, training=None):
        """
        Forward pass through Dense -> Hidden -> Dense.

        Args:
        -----------
        inputs: tf.Tensor
            Input tensor of shape (batch_size, time_steps, input_size).
        hidden_states: list
            List of initial hidden states for each hidden layer.
        training: bool
            TensorFlow flag for training mode.
        
        Returns:
        -----------
        probs: tf.Tensor)
            Action probabilities (batch_size, output_size).
        new_states: list
            List of final hidden states from each hidden layer.
        """

        # Process input through initial Dense layer
        processed_input = self.input_fc(inputs, training=training)
        output = processed_input
        new_states = []
        initial_states_for_call = [None] * self.num_layers

        # Prepare initial states for GRU layers if provided
        if 'GRU' in self.layer_type and hidden_states is not None:
             initial_states_for_call = hidden_states

        # Pass through hidden layers
        for i, hidden_layer in enumerate(self.hidden_layers):
            current_initial_state = initial_states_for_call[i]
            if 'GRU' in self.layer_type:
                output, state = hidden_layer(output, initial_state=current_initial_state, training=training)
                new_states.append(state)
            elif self.layer_type == 'Dense':
                 output = hidden_layer(output, training=training)
            else:
                 raise TypeError(f"Layer type {self.layer_type} processing not handled in call.")

        # Final output layer to get action probabilities
        probs = self.fc_out(output, training=training)
        return probs, new_states
    
    def get_config(self):
        """
        Returns the configuration of the Actor model.

        Returns:
        -----------
            config: dict
                Configuration dictionary.
        """

        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "num_layers": self.num_layers,
            "prob_connection": self.prob_connection,
            "layer_type": self.layer_type,
            "alpha": self.alpha
        })
        return config
    
    def get_hidden_dense(self, inputs):
        """
        Obtains the hidden representation from the Actor model when using Dense layers.

        Args:
        -----------
        inputs: tf.Tensor
            Input tensor of shape (batch_size, time_steps, input_size).

        Returns:
        -----------
        output: tf.Tensor
            Hidden representation tensor after passing through the network.
        """
        # Process input through initial Dense layer
        processed_input = self.input_fc(inputs, training=False)
        output = processed_input

        # Process through all hidden_layers (all are Dense when layer_type=='Dense')
        for hidden_layer in self.hidden_layers:
            output = hidden_layer(output, training=False)
        return output

# --------------------------
# Critic (Value) Network
# --------------------------
@tf.keras.utils.register_keras_serializable()
class CriticModel(Model):
    def __init__(self,
                 actor_hidden_size,
                 act_size,
                 hidden_size,
                 num_layers=1,
                 prob_connection=1.0,
                 layer_type='GRU_standard',
                 alpha = 0.1,
                 **kwargs):
        """
        Builds a Critic network: Dense -> Hidden layers -> Dense.

        Args:
        -----------
        actor_hidden_size: int
            Dimensionality of the actor's hidden representation.
        act_size: int
            Number of actions.
        hidden_size: int
            Number of units for the hidden layers.
        num_layers: int
            Number of hidden layers.
        prob_connection: float
            Connection probability for weights in hidden layers.
        layer_type: str
            Type of the hidden layers, can be 'GRU_standard', 'GRU_modified' or 'Dense' units.
        alpha: float
            A factor representing the discretized time step over the time constant.
        """
        super(CriticModel, self).__init__(**kwargs)
        self.num_layers        = num_layers
        self.layer_type        = layer_type
        self.input_size        = actor_hidden_size + act_size
        self.actor_hidden_size = actor_hidden_size
        self.act_size          = act_size
        self.hidden_size       = hidden_size
        self.prob_connection   = prob_connection
        self.alpha             = alpha

        # --- Helper function to create constraint instance ---
        def _create_new_constraint_if_sparse(prob):
            return SparseConstraint(prob) if prob < 1.0 else None

        # 1. Initial Dense Layer
        self.input_fc = layers.Dense(self.input_size, activation='relu', name='critic_input_dense')

        # 2. Hidden Layers
        self.hidden_layers = []
        for i in range(num_layers):
            layer_name_base = f'critic_{layer_type.lower().replace("_","")}_{i}'

            # --- Create constraint instances specifically for this hidden layer ---
            kernel_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)
            recurrent_sparse_constraint_hidden = None
            if 'GRU' in layer_type:
                 recurrent_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)

            if layer_type == 'GRU_modified':
                 self.hidden_layers.append(ModifiedGRU(hidden_size,
                                                      return_sequences=True, return_state=True,
                                                      kernel_constraint=kernel_sparse_constraint_hidden,
                                                      recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                      name=layer_name_base,
                                                      alpha = alpha))
            elif layer_type == 'GRU_standard':
                self.hidden_layers.append(layers.GRU(hidden_size,
                                                     activation='tanh', recurrent_activation='sigmoid',
                                                     return_sequences=True, return_state=True,
                                                     kernel_constraint=kernel_sparse_constraint_hidden,
                                                     recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                     name=layer_name_base))
            elif layer_type == 'Dense':
                self.hidden_layers.append(layers.Dense(hidden_size, activation='relu',
                                                       kernel_constraint=kernel_sparse_constraint_hidden,
                                                       name=layer_name_base))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # 3. Final Dense Layer
        self.fc_out = layers.Dense(1, name='critic_output_dense')

    def call(self, inputs, hidden_states=None, training=None):
        """
        Forward pass through Dense -> Hidden -> Dense.

        Args:
        -----------
        inputs: tf.Tensor
            Input tensor of shape (batch, time_steps, input_size).
        hidden_states: list
            List of initial hidden states for each hidden layer.
        training: bool
            TensorFlow flag for training mode.

        Returns:
        -----------
        value: tf.Tensor
            Estimated state value (batch, 1).
        new_states: list
            List of final hidden states from each hidden layer.
        """

        # Process input through initial Dense layer
        processed_input = self.input_fc(inputs, training=training)
        output = processed_input
        new_states = []
        initial_states_for_call = [None] * self.num_layers

        # Prepare initial states for GRU layers if provided
        if 'GRU' in self.layer_type and hidden_states is not None:
             initial_states_for_call = hidden_states

        # Pass through hidden layers
        for i, hidden_layer in enumerate(self.hidden_layers):
            current_initial_state = initial_states_for_call[i]
            if 'GRU' in self.layer_type:
                output, state = hidden_layer(output, initial_state=current_initial_state, training=training)
                new_states.append(state)
            elif self.layer_type == 'Dense':
                 output = hidden_layer(output, training=training)
            else:
                 raise TypeError(f"Layer type {self.layer_type} processing not handled in call.")

        # Final output layer to get state value
        value = self.fc_out(output, training=training)
        return value, new_states
    
    def get_config(self):
        """
        Returns the configuration of the Critic model.

        Returns:
        -----------
            config: dict
                Configuration dictionary.
        """

        config = super().get_config()
        config.update({
            "actor_hidden_size": self.actor_hidden_size,
            "act_size": self.act_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "prob_connection": self.prob_connection,
            "layer_type": self.layer_type,
            "alpha": self.alpha
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Keras will pass everything that was in get_config() here
        return cls(
            config.pop("actor_hidden_size"),
            config.pop("act_size"),
            **config
        )

# --------------------------
# Actor-Critic Agent
# --------------------------
class ActorCriticAgent:
    def __init__(self,
                 obs_size,
                 act_size,
                 actor_hidden_size=128,
                 critic_hidden_size=128,
                 actor_layers=1,
                 critic_layers=1,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 noise_std=0.0,
                 actor_prob_connection=1.0,
                 critic_prob_connection=1.0,
                 layer_type='GRU_standard',
                 alpha = 0.1,
                 ):
        """
        Builds an Actor-Critic model.

        Args:
        -----------
        obs_size: int
            Dimensionality of the observations (states).
        act_size: int
            Number of actions.
        actor_hidden_size: int
            Number of units for the actor's hidden layers.
        critic_hidden_size: int
            Number of units for the critic's hidden layers.
        actor_layers: int
            Number of hidden layers in the actor.
        critic_layers: int
            Number of hidden layers in the critic.
        actor_lr: float
            Learning rate for the actor optimizer.
        critic_lr: float
            Learning rate for the critic optimizer.
        noise_std: float
            Standard deviation of Gaussian noise added to states for exploration.
        actor_prob_connection: float
            Connection probability for weights in actor hidden layers.
        critic_prob_connection: float
            Connection probability for weights in critic hidden layers. 
        layer_type: str
            Type of the hidden layers, can be 'GRU_standard', 'GRU_modified' or 'Dense' units.
        alpha: float
            A factor representing the discretized time step over the time constant.
        """

        # Build Actor model
        self.actor = ActorModel(
            input_size=obs_size, hidden_size=actor_hidden_size, output_size=act_size,
            num_layers=actor_layers, prob_connection=actor_prob_connection, layer_type=layer_type, alpha=alpha
        )

        # Determine the size of the actor's hidden representation for the critic input
        actor_hid_for_critic = actor_hidden_size

        # Build Critic model
        self.critic = CriticModel(
            actor_hidden_size=actor_hid_for_critic, act_size=act_size, hidden_size=critic_hidden_size, num_layers=critic_layers,
            prob_connection=critic_prob_connection, layer_type=layer_type, alpha=alpha
        )

        # Optimizers and other parameters
        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)
        self.noise_std = noise_std
        self.act_size = act_size

    def add_noise(self, state):
        """
        Adds Gaussian noise to the state for exploration.

        Args:
        -----------
        state: np.array
            The current state.
        
        Returns:
        -----------
        noisy_state: np.array
            The state with added Gaussian noise.
        
        """

        if self.noise_std != 0.0:
            noise = np.random.normal(0, self.noise_std, size=state.shape).astype(state.dtype)
            return state + noise
        return state

    def select_action(self, state, actor_hidden_states=None, training=True):
        """
        Selects an action based on the current state and the actor's policy.

        Args:
        -----------
        state: np.array
            The current state.
        actor_hidden_states: list
            List of hidden states for the actor's hidden layers.
        training: bool
            TensorFlow flag for training mode.
        
        Returns:
        -----------
        action_idx: int
            The selected action index.
        log_prob: tf.Tensor
            The log probability of the selected action.
        new_actor_hidden_states: list
            List of updated hidden states from the actor's hidden layers.
        
        """

        # Process the state to get the correct dimensions
        state_noisy = self.add_noise(state)
        state_tensor = tf.convert_to_tensor(state_noisy, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, axis=0)
        state_tensor = tf.expand_dims(state_tensor, axis=1)

        # Forward pass through the actor to get action probabilities
        probs, new_actor_hidden_states = self.actor(
            state_tensor, hidden_states=actor_hidden_states, training=training
        )

        # Sample action from the probability distribution and correct dimensions
        probs_t = probs[:, 0, :]
        action = tf.random.categorical(tf.math.log(probs_t + 1e-10), num_samples=1)
        action = tf.squeeze(action, axis=-1)
        action_idx = tf.cast(action[0], dtype=tf.int32)
        action_one_hot = tf.one_hot(action_idx, depth=self.act_size)
        action_one_hot = tf.expand_dims(action_one_hot, axis=0)

        # Compute log probability of the selected action
        log_prob = tf.math.log(tf.reduce_sum(probs_t * action_one_hot, axis=1) + 1e-10)
        return int(action.numpy()[0]), log_prob[0], new_actor_hidden_states

    def evaluate_critic_step(self, r_t, prev_a, critic_hidden_states=None, training=False):
        """
        Evaluates a step in the critic network.

        Args:
        -----------
        r_t: np.array
            Hidden representation from the actor of shape (actor_hidden_size,).
        prev_a: np.array
            Previous action taken as one-hot vector of shape (act_size,).
        critic_hidden_states: list
            List of hidden states for the critic's hidden layers.
        training: bool
            TensorFlow flag for training mode.

        Returns:
        -----------
        scalar_value: tf.Tensor
            Estimated state value as a scalar tensor.
        new_critic_hidden_states: list
            List of updated hidden states from the critic's hidden layers.
        """

        # Ensure r_t is provided
        if r_t is None:
            raise ValueError("For Dense Actor, use evaluate_critic_step with r_t calculated via get_hidden_dense.")

        # Prepare input for the critic  (1, 1, actor_hidden_size + act_size)
        inp = np.concatenate([r_t, prev_a], axis=0)[None, None, :]
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)

        # Forward pass through the critic
        value, new_critic_hidden_states = self.critic(inp, hidden_states=critic_hidden_states, training=training)
        scalar_value = value[0, 0, 0]

        return scalar_value, new_critic_hidden_states
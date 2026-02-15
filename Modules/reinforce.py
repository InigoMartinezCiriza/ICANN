import tensorflow as tf
from helper_functions import discount_rewards
from helper_functions import pad_hidden_states
from helper_functions import collect_other_measurements
import numpy as np

# --------------------------
# Training using REINFORCE with baseline and regularization
# --------------------------
def train_agent(env, agent, num_episodes=500, gamma=0.99, print_interval=10, l2_actor=1e-4, l2_critic=1e-4):
    """
    Trains the Actor-Critic agent using the REINFORCE algorithm with baseline and L2 regularization.
    Provides detailed recording of rewards, losses, and firing rates.
    Processes one episode at a time for RNN updates.

    Args:
    -----------
    env: Gymnasium.Env
        The potentially wrapped environment instance.
    agent: tf.keras.models.Model
        The ActorCriticAgent instance. Assumes agent.actor and agent.critic
        have a 'layer_type' attribute.
    num_episodes: int
        Number of episodes to train for.
    gamma: float
        Discount factor. Takes values between 0 and 1.
    print_interval: int
        Interval for printing progress.
    l2_actor: float
        L2 regularization strength for the actor.
    l2_critic: float
        L2 regularization strength for the critic.

    Returns:
    -----------
    total_rewards_history: list
        Total reward per episode.
    actor_loss_history: list
        Actor loss per episode.
    critic_loss_history: list
        Critic loss per episode.
    actor_firing_rates: list of lists or None
        Hidden states (firing rates) per layer for the actor
    critic_firing_rates: list of lists or None
        Hidden states (firing rates) per layer for the critic
    other_measurements_history: list
        Other measurements of the environment per episode.
    """

    # Initialize recording containers
    total_rewards_history = []
    actor_loss_history = []
    critic_loss_history = []
    other_measurements_history = []

    # Determine if networks are recurrent based on the stored layer_type
    actor_is_recurrent = hasattr(agent.actor, 'layer_type') and 'GRU' in agent.actor.layer_type
    critic_is_recurrent = hasattr(agent.critic, 'layer_type') and 'GRU' in agent.critic.layer_type

    # L2 regularization parameters
    lambda_actor = l2_actor
    lambda_critic = l2_critic

    # Determine the action dimension size once
    act_size = env.action_space.n

    # Containers to store hidden states across all episodes
    actor_states_all = [] if actor_is_recurrent else None
    critic_states_all = [] if critic_is_recurrent else None

    # Initialize max steps for padding
    max_steps_actor = 0
    max_steps_critic = 0

    # --- Main training loop ---
    for episode in range(1, num_episodes + 1):

        # Episode data collection
        state, info = env.reset()
        done = False
        states = []
        actions = []
        rewards = []
        actor_hidden_states_ep = []
        critic_hidden_states_ep = []
        
        # Store hidden states
        current_actor_hidden = None
        current_critic_hidden = None

        # --- Episode interaction loop ---
        while not done:

            # Agent selects action
            action, _, actor_hidden_states_after_select = agent.select_action(
                state, actor_hidden_states=current_actor_hidden, training=True
            )
            
            # Store the latest actor hidden state if recurrent
            if actor_is_recurrent:
                current_actor_hidden = actor_hidden_states_after_select
                actor_hidden_states_ep.append(current_actor_hidden[0].numpy().flatten())

            # Environment steps
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store data for this step
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Hidden state depends on the actor nature
            if actor_is_recurrent:
                 r_t = actor_hidden_states_ep[-1]
            else:
                 state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                 state_tensor = tf.expand_dims(state_tensor, axis=0)
                 state_tensor = tf.expand_dims(state_tensor, axis=1)
                 # Get hidden state activation for dense network
                 hidden_dense = agent.actor.get_hidden_dense(state_tensor)
                 r_t = hidden_dense[0, 0, :].numpy()

            # Evaluate critic to get next hidden state
            prev_a = np.zeros(act_size, dtype=np.float32)
            prev_a[ actions[-1] ] = 1.0

            _, critic_hidden_state_after_eval = agent.evaluate_critic_step(
                r_t,
                np.eye(act_size, dtype=np.float32)[actions[-1]],
                critic_hidden_states=current_critic_hidden,
                training=False
                )
            
            # Hidden state update for critic
            if critic_is_recurrent:
                critic_hidden_states_ep.append(critic_hidden_state_after_eval[0].numpy().flatten())
            
            # Update current state
            state = next_state

        # --- Networks update ---
        returns = discount_rewards(rewards, gamma)
        
        # Convert collected data to tensors with appropriate shapes (batch_size = 1, sequence_length, feature_dim)
        states_sequence_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        states_sequence_tensor = tf.expand_dims(states_sequence_tensor, axis=0)

        # Actions tensor
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        actions_tensor = tf.expand_dims(actions_tensor, axis=0)

        # Returns tensor
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        returns_tensor = tf.expand_dims(returns_tensor, axis=0)

        # ---  Critic input ---

        # Calculate dense activations of the actor in advance, if layer_type=='Dense'
        dense_feats = None
        if not actor_is_recurrent:
             states_np = np.stack(states, axis=0)
             states_tensor = tf.convert_to_tensor(states_np, dtype=tf.float32)
             states_tensor = tf.expand_dims(states_tensor, axis=0)
             dense_hidden_seq = agent.actor.get_hidden_dense(states_tensor)
             dense_feats = dense_hidden_seq[0].numpy()

        # Critic input depends on the actor nature
        critic_inputs = []
        for t in range(len(actions)):
                if actor_is_recurrent:
                    feat = actor_hidden_states_ep[t]
                else:
                    feat = dense_feats[t]
                # Current action one-hot encoding
                curr_a = np.eye(act_size, dtype=np.float32)[actions[t]]
                critic_inputs.append( np.concatenate([feat, curr_a]) )

        # Convert to tensor with shape (1, sequence_length, feature_dim)
        critic_inputs = tf.convert_to_tensor([critic_inputs], dtype=tf.float32)

        # --- Actor Update ---
        with tf.GradientTape() as tape_actor:

            # Pass the whole sequence to the actor and create one-hot encoding for actions taken
            all_probs, _ = agent.actor(states_sequence_tensor, hidden_states=None, training=True)
            actions_one_hot = tf.one_hot(actions_tensor, depth=act_size)

            # Select probabilities of the actions actually taken and calculate log probabilities
            probs_taken_actions = tf.reduce_sum(all_probs * actions_one_hot, axis=-1)
            log_probs = tf.math.log(probs_taken_actions + 1e-10)

            # Get values from Critic and calculates advantage
            all_values, _ = agent.critic(critic_inputs, hidden_states=None, training=True)
            values = tf.squeeze(all_values, axis=-1)
            advantage = returns_tensor - values

            # Calculate actor loss and add L2 regularization
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))
            l2_reg_actor = tf.add_n([tf.nn.l2_loss(v) for v in agent.actor.trainable_weights if 'kernel' in v.name or 'recurrent_kernel' in v.name])
            actor_loss += lambda_actor * l2_reg_actor

        # Compute and apply gradients for Actor
        actor_grads = tape_actor.gradient(actor_loss, agent.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, clip_norm=1.0)
        agent.actor_optimizer.apply_gradients(zip(actor_grads, agent.actor.trainable_variables))

        # --- Critic Update ---
        with tf.GradientTape() as tape_critic:

            # Pass the whole sequence to the Critic and calculate values
            all_values, _ = agent.critic(critic_inputs, hidden_states=None, training=True)
            values = tf.squeeze(all_values, axis=-1)

            # Calculate critic loss and add L2 regularization
            critic_loss = tf.reduce_mean(tf.square(returns_tensor - values))
            l2_reg_critic = tf.add_n([tf.nn.l2_loss(v) for v in agent.critic.trainable_weights if 'kernel' in v.name or 'recurrent_kernel' in v.name])
            critic_loss += lambda_critic * l2_reg_critic

        # Compute and apply gradients for Critic
        critic_grads = tape_critic.gradient(critic_loss, agent.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, clip_norm=1.0)
        agent.critic_optimizer.apply_gradients(zip(critic_grads, agent.critic.trainable_variables))

        # --- Recording ---
        # Hidden states recording
        if actor_is_recurrent:
            actor_hidden_states_ep = np.array(actor_hidden_states_ep).T  # shape: units x steps
            actor_states_all.append(actor_hidden_states_ep)
            max_steps_actor = max(max_steps_actor, actor_hidden_states_ep.shape[1])

        if critic_is_recurrent:
            critic_hidden_states_ep = np.array(critic_hidden_states_ep).T
            critic_states_all.append(critic_hidden_states_ep)
            max_steps_critic = max(max_steps_critic, critic_hidden_states_ep.shape[1])

        # Other measurements recording
        last_reward = rewards[-1]
        measurement = collect_other_measurements(env, done, last_reward)
        other_measurements_history.append(measurement)

        # Reward and loss recording
        total_rewards_history.append(sum(rewards))
        actor_loss_history.append(actor_loss.numpy())
        critic_loss_history.append(critic_loss.numpy())

        # Print progress
        if episode % print_interval == 0:
            print(f"Episode {episode}\tTotal Reward: {sum(rewards):.2f}\t"
                  f"Actor Loss: {actor_loss.numpy():.4f}\tCritic Loss: {critic_loss.numpy():.4f}\t")
        
        # --- Post-training processing of hidden states ---
        actor_states_tensor = None
        if actor_is_recurrent:
            units_actor = actor_states_all[0].shape[0]
            actor_states_tensor = np.array([pad_hidden_states(s, max_steps_actor, units_actor) for s in actor_states_all])
            actor_states_tensor = np.transpose(actor_states_tensor, (1, 2, 0))

        critic_states_tensor = None
        if critic_is_recurrent:
            units_critic = critic_states_all[0].shape[0]
            critic_states_tensor = np.array([pad_hidden_states(s, max_steps_critic, units_critic) for s in critic_states_all])
            critic_states_tensor = np.transpose(critic_states_tensor, (1, 2, 0))

    return total_rewards_history, actor_loss_history, critic_loss_history, \
           actor_states_tensor, critic_states_tensor, other_measurements_history
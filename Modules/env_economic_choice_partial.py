import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Action mapping
ACT_FIXATE = 0
ACT_CHOOSE_LEFT = 1
ACT_CHOOSE_RIGHT = 2

# Observation mapping
OBS_FIX_CUE = 0
OBS_JUICE_POS = 1
OBS_N_LEFT = 2
OBS_N_RIGHT = 3
OBS_DIM = 4

# Epoch names
EPOCH_FIXATION = 'Fixation'
EPOCH_OFFER = 'Offer'
EPOCH_DECISION = 'Decision'
EPOCH_END = 'End'


class EconomicChoiceEnv(gym.Env):
    """
    Gymnasium environment for the Economic Choice Task with Fixation, Offer,
    and Decision epochs.
    - Agent must fixate during Fixation and Offer.
    - During Decision, agent chooses left or right, depending on the offer.
    - Choosing left/right during Decision immediately ends the trial and 
      delivers the corresponding reward.
    - Fixating during Decision gives a small penalty per step.
    - Timeout during Decision results in abortion of the trial.
    - Observation: [Fixation cue, Juice position, Number of drops on left, Number of drops on right]
    """

    def __init__(self,
                 dt=10,
                 A_to_B_ratio=2.2,
                 reward_B=100,
                 abort_penalty=-0.1,
                 input_noise_sigma=0.0,
                 reward_fixation=0.01,
                 reward_go_fixation=-0.01,
                 duration_params=[1500, 1000, 2000, 2000]
                 ):
        """
        Initializes the environment.

        Args:
        -----------
        dt: int
            Simulation time step in ms.
        A_to_B_ratio: float
            Relative value A vs B.
        reward_B: float
            Base reward for one drop of juice B.
        abort_penalty: float
            Reward (negative) for fixation breaks or timeouts.
        input_noise_sigma: float
            Standard deviation of noise on numerical inputs.
        reward_fixation: float
            Reward per step for correct fixation.
        reward_go_fixation: float
            Reward (negative) per step for fixating during Decision.
        duration_params: list
            Number of ms each duration lasts. Takes the shape
            [Fixation, Offer_min, Offer_max, Decision_timeout].
        """

        super().__init__()

        # --- Task parameters ---
        self.dt = dt
        self.dt_sec = dt / 1000.0
        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = float(reward_B)
        self.R_A = float(A_to_B_ratio * self.R_B)
        self.R_ABORTED = float(abort_penalty)
        self.sigma = input_noise_sigma

        # Apply noise scaling based on dt
        self.noise_scale = 1.0 / np.sqrt(self.dt_sec) if self.dt_sec > 0 else 1.0

        # Store reward parameters
        self.R_fix_step = float(reward_fixation)
        self.R_go_fix_step = float(reward_go_fixation)

        # --- Action and Observation Spaces ---
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.1, high=2.1, shape=(OBS_DIM,), dtype=np.float32)

        # --- Timing ---
        if len(duration_params) != 4:
            raise ValueError("duration_params must have 4 elements: [Fixation, Offer_min, Offer_max, Decision_timeout]")
        self._durations_ms = {
            'Fixation':    duration_params[0],
            'Offer_min':   duration_params[1],
            'Offer_max':   duration_params[2],
            'Decision_timeout':  duration_params[3],
        }

        # Convert durations to steps
        self.t_fixation_steps = self._ms_to_steps(self._durations_ms['Fixation'])
        self.t_choice_timeout_steps = self._ms_to_steps(self._durations_ms['Decision_timeout'])

        # --- Trial setup ---
        # Possible juice configurations and offer pairs (nB, nA)
        self.juice_types = [('A', 'B'), ('B', 'A')]
        self.offer_sets = [(0, 1), (1, 2), (1, 1), (2, 1), (3, 1),
                           (4, 1), (6, 1), (10, 1), (2, 0)]
        self.rng = np.random.default_rng()

        # --- State variables ---
        self.current_step = 0
        self.trial_juice_LR = None              # ('A', 'B') or ('B', 'A')
        self.trial_offer_BA = None              # (nB, nA) chosen for trial
        self.trial_nL = 0                       # n drops Left
        self.trial_nR = 0                       # n drops Right
        self.trial_rL = 0.0                     # reward value Left
        self.trial_rR = 0.0                     # reward value Right
        self.epochs = {}                        # Stores epoch start/end steps
        self.current_epoch_name = EPOCH_END     # Current epoch
        self.t_go_signal_step = -1              # Step when GO signal appears
        self.t_choice_made_step = -1            # Step when L/R choice made in GO
        self.chosen_action = -1                 # Action chosen (0, 1 or 2)

    def _ms_to_steps(self, ms):
        """
        Converts milliseconds to simulation steps.
        Ensures at least 1 step for any non-zero duration
        
        Args:
        -----------
        ms: int
            Duration in milliseconds.
        
        Returns:
        -----------
        int
            Equivalent number of steps.
        """
        return max(1, int(np.round(ms / self.dt))) if ms > 0 else 0

    def _calculate_epochs(self, delay_ms):
        """
        Calculates epoch boundaries in steps for the current trial.

        Args:
        -----------
        delay_ms: int
            Random delay duration between Offer and Decision epochs in ms.
        """

        # Epoch boundaries in steps
        t_fix_end = self.t_fixation_steps
        t_delay_steps = self._ms_to_steps(delay_ms)
        t_go_signal = t_fix_end + t_delay_steps
        t_choice_end = t_go_signal + self.t_choice_timeout_steps
        # Max trial time is effectively the end of the choice window + buffer
        t_max = t_choice_end + self._ms_to_steps(100)

        # Store epoch boundaries
        self.epochs = {
            EPOCH_FIXATION:   (0, t_fix_end),
            EPOCH_OFFER:      (t_fix_end, t_go_signal),
            EPOCH_DECISION:   (t_go_signal, t_choice_end),
            EPOCH_END:        (t_max, t_max + 1),
            'tmax_steps': t_max
        }
        self.t_go_signal_step = t_go_signal

    def _get_current_epoch(self, step):
        """
        Determines the current epoch name based on the step count.
        Assumes trial hasn't already ended by choice.
           
        Args:
        -----------
        step: int
           Current time step in the trial.

        Returns:
        -----------
        str
            Current epoch name.
        """

        # Check fixed epochs based on step count progression
        if self.epochs[EPOCH_FIXATION][0] <= step < self.epochs[EPOCH_FIXATION][1]:
            return EPOCH_FIXATION
        elif self.epochs[EPOCH_OFFER][0] <= step < self.epochs[EPOCH_OFFER][1]:
            return EPOCH_OFFER
        elif self.epochs[EPOCH_DECISION][0] <= step < self.epochs[EPOCH_DECISION][1]:
            # If we are within the Decision window and no choice has been made yet
            if self.t_choice_made_step == -1:
                return EPOCH_DECISION
            else:
                 return EPOCH_END
        else:
            return EPOCH_END

    def _select_trial_conditions(self):
        """
        Sets up juice/offer conditions for a new trial.
        """
        
        # Randomly select juice positions and offer amounts
        self.trial_juice_LR = random.choice(self.juice_types)
        nB, nA = random.choice(self.offer_sets)
        self.trial_offer_BA = (nB, nA)
        
        # Assign left/right offers and rewards based on juice positions
        juiceL, juiceR = self.trial_juice_LR
        if juiceL == 'A':
            self.trial_nL, self.trial_nR = nA, nB
            self.trial_rL, self.trial_rR = nA * self.R_A, nB * self.R_B
        else:
            self.trial_nL, self.trial_nR = nB, nA
            self.trial_rL, self.trial_rR = nB * self.R_B, nA * self.R_A

        # Random delay for the Offer epoch
        delay_ms = self.rng.uniform(self._durations_ms['delay_min'],
                                    self._durations_ms['delay_max'])
        self._calculate_epochs(delay_ms)

    def _get_observation(self, current_epoch):
        """
        Constructs the 4D observation vector based on the current epoch.
        
        Args:
        -----------
        current_epoch: str
            Current epoch name.

        Returns:
        -----------
        np.ndarray
            Observation vector of shape (4,).
        """

        # Intialize observation
        obs = np.zeros(OBS_DIM, dtype=np.float32)       

        # Fixation cue, active during Fixation and Offer epochs
        if current_epoch in [EPOCH_FIXATION, EPOCH_OFFER]:
            obs[OBS_FIX_CUE] = 1.0

        # Juice position, Number of juice drops on left, Number of juice drops on right
        # Active during Offer but not Decision so the agent must remember the offer
        if current_epoch == EPOCH_OFFER:
            juiceL, juiceR = self.trial_juice_LR
            obs[OBS_JUICE_POS] = -1.0 if juiceL == 'A' else 1.0

            # Scale down the number of drops for numerical stability and add noise
            scaling_factor = 10.0
            scaled_nL = self.trial_nL / scaling_factor
            scaled_nR = self.trial_nR / scaling_factor
            if self.sigma > 0:
                noise_L = self.rng.normal(scale=self.sigma) * self.noise_scale
                noise_R = self.rng.normal(scale=self.sigma) * self.noise_scale
                scaled_nL += noise_L
                scaled_nR += noise_R

            # Clip to reasonable bounds
            obs[OBS_N_LEFT] = np.clip(scaled_nL, 0.0, 1.1)
            obs[OBS_N_RIGHT] = np.clip(scaled_nR, 0.0, 1.1)
            obs[OBS_JUICE_POS] = np.clip(obs[OBS_JUICE_POS], -1.0, 1.0)

        # If Decision epoch, return zeros. The states are no longer observed.
        # They must be remembered from the Offer epoch.
        if current_epoch == EPOCH_DECISION:
            return np.zeros(OBS_DIM, dtype=np.float32)

        # During End epoch, return zero observation
        if current_epoch == EPOCH_END:
             obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Ensure observation fits the defined space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs


    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new trial.
        
        Args:
        -----------
        seed: int or None
            Random seed for reproducibility.

        Returns:
        -----------
        observation: np.ndarray
            Initial observation after reset.
        info: dict
            Auxiliary information about the initial state.
        """ 
        
        # Set random seed
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        # Reset trial state
        self.current_step = 0
        self._select_trial_conditions()
        self.current_epoch_name = self._get_current_epoch(self.current_step)
        self.t_choice_made_step = -1
        self.chosen_action = -1

        # Get initial observation and info
        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info["reward"] = 0.0
        info["action"] = None

        return observation, info


    def step(self, action):
        """
        Advances the environment by one time step.
        
        Args:
        -----------
        action: int
            Action taken by the agent.
        
        Returns:
        -----------
        observation: np.ndarray
            Observation after taking the action.
        reward: float
            Reward received after taking the action.
        terminated: bool
            Whether the trial has ended successfully.
        truncated: bool
            Whether the trial was truncated due to time limit.
        info: dict
            Auxiliary information about the current state.
        """

        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        # Initialize step variables
        terminated = False
        truncated = False
        reward = 0.0
        prev_epoch = self.current_epoch_name

        # --- Determine reward based on action in current step ---
        abort = False

        # Fixation epochs
        if prev_epoch == EPOCH_FIXATION or prev_epoch == EPOCH_OFFER:
            if action != ACT_FIXATE:
                # Broke fixation
                abort = True
                reward = self.R_ABORTED
            else:
                reward = self.R_fix_step

        # Decision epoch
        elif prev_epoch == EPOCH_DECISION:
            # Choosing to fixate during decision is penalized but doesn't end trial
            if action == ACT_FIXATE:
                reward = self.R_go_fix_step
            # Choosing a side ends the trial
            elif action in [ACT_CHOOSE_LEFT, ACT_CHOOSE_RIGHT]:
                self.t_choice_made_step = self.current_step
                self.chosen_action = action
                terminated = True
                # Assign final reward based on choice
                if action == ACT_CHOOSE_LEFT:
                    reward = self.trial_rL
                else:
                    reward = self.trial_rR
                self.current_epoch_name = EPOCH_END

        # If an abort occurred, set terminated and end epoch
        if abort:
            terminated = True
            self.current_epoch_name = EPOCH_END

        # --- Advance time and check for state transitions ---
        if not terminated:
            self.current_step += 1
            # Determine the epoch we would enter next step if trial continues
            next_epoch = self._get_current_epoch(self.current_step)

            # Check specifically for Decision timeout
            go_start, go_end = self.epochs[EPOCH_DECISION]
            if prev_epoch == EPOCH_DECISION and self.current_step >= go_end and self.t_choice_made_step == -1:
                 # Timeout: Exceeded Decision duration without making a choice
                 reward = self.R_ABORTED
                 terminated = True
                 next_epoch = EPOCH_END

            # Check for general truncation (overall time limit)
            elif self.current_step >= self.epochs['tmax_steps']:
                 truncated = True
                 reward = 0.0
                 next_epoch = EPOCH_END

            # Update the current epoch name based on time advancement
            self.current_epoch_name = next_epoch

        # --- Get next observation and info ---
        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info["action"] = action
        info["reward"] = reward

        # Ensure terminated and truncated are mutually exclusive
        if terminated:
            truncated = False

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        """
        Returns auxiliary information about the current state.
        
        Returns:
        -----------
        info: dict
            Information dictionary.
        """

        # Determine if the choice was correct (only if a choice was made)
        is_correct = None
        if self.chosen_action != -1:
             if self.chosen_action == ACT_CHOOSE_LEFT:
                 is_correct = self.trial_rL >= self.trial_rR
             elif self.chosen_action == ACT_CHOOSE_RIGHT:
                 is_correct = self.trial_rR >= self.trial_rL

        # Construct info dictionary
        info = {
            "step": self.current_step,
            "epoch": self.current_epoch_name,
            "juice_LR": self.trial_juice_LR,
            "offer_BA": self.trial_offer_BA,
            "nL": self.trial_nL,
            "nR": self.trial_nR,
            "rL": self.trial_rL,
            "rR": self.trial_rR,
            "chosen_action": self.chosen_action,
            "choice_time_step": self.t_choice_made_step,
            "is_correct_choice": is_correct,
            "A_to_B_ratio": self.A_to_B_ratio,
            "rewards_cfg": {
                "fix_step": self.R_fix_step,
                "go_fix_step": self.R_go_fix_step,
                "abort": self.R_ABORTED
             },
        }
        return info

    def render(self):
        """No rendering implemented."""
        pass

    def close(self):
        """Clean up any resources."""
        pass
"""
Wrapper for stable-baselines3 agents to match the interface of custom agents.
"""
import os
import numpy as np
import torch
from typing import Tuple, Optional
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import Env
import src.utils.context_provider as context_provider
from src.utils.constraint_monitor import ConstraintMonitor


class SB3Wrapper:
    """
    Wrapper for stable-baselines3 agents to work with the existing training/evaluation framework.
    
    This wrapper makes SB3 agents compatible with the custom agent interface, allowing
    them to be used seamlessly with train.py and evaluate_policy functions.
    """
    
    def __init__(self,
                 agent_type: str,  # 'a2c', 'ppo', 'dqn'
                 env: Env,
                 input_shape,
                 action_dim,
                 agent_kwargs=None,
                 model_path: Optional[str] = None,
                 train_new: bool = False,
                 num_episodes: int = 1000,
                 verbose: bool = False):
        """
        Initialize SB3 wrapper.
        
        Args:
            agent_type: Type of agent ('a2c', 'ppo', 'dqn')
            env: Gymnasium environment
            input_shape: Shape of input observations
            action_dim: Number of actions
            agent_kwargs: Hyperparameters for SB3 agent
            model_path: Path to pretrained model (or HuggingFace model ID)
            train_new: If True, train a new model; if False, load from model_path
            num_episodes: Number of episodes for training (if train_new=True)
            verbose: Verbose output
        """
        self.agent_type = agent_type.lower()
        self.env = env
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.verbose = verbose
        self.trained = not train_new and model_path is not None
        
        # Create VecEnv wrapper (SB3 requires vectorized environments)
        # For Atari games with CNN policies, SB3 expects frame stacking
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Determine policy type based on actual observation space
        # Check the observation space from the vec_env (after all wrappers)
        obs_space = self.vec_env.observation_space
        
        # Check if using RAM observations (1D vector) or images
        # RAM observations are 1D, images are 2D or 3D
        obs_shape = obs_space.shape
        
        print(f"[SB3Wrapper] Environment observation space: {obs_space}")
        print(f"[SB3Wrapper] Observation shape: {obs_shape}")
        
        # Determine if we should use CNN policy
        # CNN policy requires: 2D or 3D observation space that SB3 recognizes as images
        # For SB3's NatureCNN, it needs either:
        # - 2D (H, W) grayscale (but SB3 has issues with this)
        # - 3D (C, H, W) or (H, W, C) with proper channels
        # - 1D means we should use MlpPolicy
        if len(obs_shape) == 1:
            # 1D observation space = RAM or vector observations, use MLP
            self.use_cnn = False
            self._needs_frame_stack = False
            print(f"[SB3Wrapper] Detected 1D observation space (RAM/vector) - will use MlpPolicy")
        elif len(obs_shape) == 2:
            # 2D (H, W) grayscale - SB3's CnnPolicy has issues with this format
            # Check if this is actually a RAM wrapper by checking the shape
            # RAM wrapper gives (128,) which is 1D, so 2D here likely means grayscale image
            # But SB3 expects channels-first, so we might need to use MLP or adjust
            # For now, if it's exactly (84, 84) or similar, try to use CnnPolicy with normalization
            self.use_cnn = True  # Will try CnnPolicy but may need adjustment
            self._needs_frame_stack = False  # No frame stacking for 2D
            print(f"[SB3Wrapper] Detected 2D observation space {obs_shape} (grayscale image) - will use MlpPolicy (flattened)")
        elif len(obs_shape) == 3:
            # 3D (C, H, W) or (H, W, C) - can use CNN
            self.use_cnn = True
            # Check if this is an Atari environment that needs frame stacking
            env_name = getattr(env, 'spec', None)
            env_id = env_name.id if env_name and hasattr(env_name, 'id') else getattr(env, 'env_name', str(env))
            is_atari = 'ALE' in str(env_id) or 'Atari' in str(env_id) or 'Seaquest' in str(env_id) or 'DemonAttack' in str(env_id) or 'Freeway' in str(env_id)
            self._needs_frame_stack = is_atari and obs_shape[0] == 1  # Only if single channel
            print(f"[SB3Wrapper] Detected 3D observation space {obs_shape} - will use CnnPolicy")
            if self._needs_frame_stack:
                print(f"[SB3Wrapper] Frame stacking enabled for Atari environment (stacking {self._frame_stack_size} frames)")
        else:
            # Unknown format, default to MLP
            self.use_cnn = False
            self._needs_frame_stack = False
            print(f"[SB3Wrapper] Unknown observation format {obs_shape} - defaulting to MlpPolicy")
        
        self._frame_stack_size = 4 if self._needs_frame_stack else 1
        self._frame_buffer = []  # Store last N frames for stacking
        
        # Default hyperparameters (can be overridden by agent_kwargs)
        default_kwargs = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'verbose': 1 if verbose else 0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Agent-specific defaults
        if self.agent_type == 'ppo':
            default_kwargs.update({
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.01
            })
        elif self.agent_type == 'a2c':
            default_kwargs.update({
                'n_steps': 5,
                'ent_coef': 0.01
            })
        elif self.agent_type == 'dqn':
            default_kwargs.update({
                'buffer_size': 100000,
                'learning_starts': 1000,
                'batch_size': 32,
                'target_update_interval': 1000,
                'exploration_fraction': 0.1,
                'exploration_final_eps': 0.05
            })
        
        # Override with agent_kwargs if provided
        if agent_kwargs:
            default_kwargs.update(agent_kwargs)
        
        # Initialize or load model
        if train_new:
            # Train a new model
            print(f"[SB3Wrapper] Creating new {self.agent_type.upper()} model for training...")
            self.model = self._create_model(default_kwargs)
            self.trained = False  # Will be trained in run_training
            print(f"[SB3Wrapper] Model created successfully")
        elif model_path:
            # Load pretrained model
            print(f"[SB3Wrapper] Loading pretrained model from: {model_path}")
            self.model = self._load_model(model_path, default_kwargs)
            self.trained = True
            print(f"[SB3Wrapper] Model loaded successfully")
        else:
            # Create model but don't train (for evaluation-only mode)
            print(f"[SB3Wrapper] Creating {self.agent_type.upper()} model for evaluation...")
            self.model = self._create_model(default_kwargs)
            self.trained = False
            print(f"[SB3Wrapper] Model created successfully (evaluation mode)")
        
        # Initialize attributes needed for compatibility
        # (use_cnn already set above)
        self.use_shield_post = False
        self.use_shield_pre = False
        self.use_shield_layer = False
        # Create a dummy shield controller for compatibility
        # Use a default requirements path even though SB3 agents don't use shields
        from src.utils.shield_controller import ShieldController
        default_requirements_path = 'src/requirements/cliff_safe.cnf'
        self.shield_controller = ShieldController(
            requirements_path=default_requirements_path,
            num_actions=action_dim,
            mode='hard',
            verbose=verbose,
            is_shield_active=False
        )
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.epsilon = 0.0  # For DQN compatibility
        self.last_obs = None
        self.last_log_prob = None
        self.last_value = None
        self.lambda_sem = 0.0  # For compatibility with train.py
        
        # Track steps for training
        self.steps_done = 0
        self.learn_step_counter = 0
        
        # For SB3 training mode
        self._use_sb3_training = train_new
        self._total_episodes_trained = 0
        
    def _create_model(self, kwargs):
        """Create a new SB3 model."""
        # Check observation space again to ensure correct policy selection
        obs_space = self.vec_env.observation_space
        obs_shape = obs_space.shape
        
        # For 2D grayscale images (H, W), SB3's CnnPolicy has issues
        # We need to either use MLP or wrap the observation space
        if len(obs_shape) == 2:
            # 2D grayscale - SB3 CnnPolicy expects channels-first
            # Use MLP policy for now (can be changed later if needed)
            policy = 'MlpPolicy'
            print(f"[SB3Wrapper] Using MlpPolicy for 2D observation space {obs_shape}")
        elif len(obs_shape) == 1:
            # RAM or vector observations
            policy = 'MlpPolicy'
        elif len(obs_shape) == 3:
            # 3D image observations - use CNN
            policy = 'CnnPolicy'
        else:
            # Default to MLP for unknown formats
            policy = 'MlpPolicy'
        
        if self.agent_type == 'ppo':
            return PPO(policy, self.vec_env, **kwargs)
        elif self.agent_type == 'a2c':
            return A2C(policy, self.vec_env, **kwargs)
        elif self.agent_type == 'dqn':
            return DQN(policy, self.vec_env, **kwargs)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def _load_model(self, model_path: str, default_kwargs):
        """Load a pretrained SB3 model."""
        import os
        from pathlib import Path
        
        # Check if it's a HuggingFace model ID
        # HuggingFace repo IDs typically look like: "user/repo-name" or "organization/repo-name"
        looks_like_hf_id = (
            '/' in model_path and 
            not model_path.startswith('./') and 
            not model_path.startswith('/') and 
            not os.path.exists(model_path) and
            not model_path.endswith('.zip')
        )
        
        if looks_like_hf_id:
            try:
                from huggingface_sb3 import load_from_hub
                from huggingface_hub import list_repo_files
                print(f"[SB3Wrapper] Loading pretrained model from HuggingFace Hub: {model_path}")
                
                # First, try to list files in the repo to see what's available
                try:
                    repo_files = list(list_repo_files(repo_id=model_path, repo_type="model"))
                    zip_files = [f for f in repo_files if f.endswith('.zip')]
                    if zip_files:
                        print(f"[SB3Wrapper] Found .zip files in repo: {', '.join(zip_files)}")
                        # Try the first zip file found
                        possible_filenames = zip_files
                    else:
                        print(f"[SB3Wrapper] No .zip files found in repo. Available files: {', '.join(repo_files[:10])}")
                        # Fall back to common patterns
                        possible_filenames = [
                            f"{self.agent_type.upper()}.zip",
                            f"{model_path.split('/')[-1]}.zip",
                            f"{self.agent_type}_{model_path.split('/')[-1]}.zip",
                            "model.zip",
                            f"{self.agent_type.upper()}_model.zip",
                            "rl-trained-agent.zip"
                        ]
                except Exception as e:
                    print(f"[SB3Wrapper] Could not list repo files: {e}")
                    # Fall back to common patterns
                    possible_filenames = [
                        f"{self.agent_type.upper()}.zip",
                        f"{model_path.split('/')[-1]}.zip",
                        f"{self.agent_type}_{model_path.split('/')[-1]}.zip",
                        "model.zip",
                        f"{self.agent_type.upper()}_model.zip",
                        "rl-trained-agent.zip"
                    ]
                
                model_loaded = False
                last_error = None
                for filename in possible_filenames:
                    try:
                        print(f"[SB3Wrapper] Trying to load: {filename}")
                        model_path_hf = load_from_hub(repo_id=model_path, filename=filename)
                        
                        # Try loading with env first (preferred), fall back to without env if observation space mismatch
                        try:
                            if self.agent_type == 'ppo':
                                model = PPO.load(model_path_hf, env=self.vec_env)
                            elif self.agent_type == 'a2c':
                                model = A2C.load(model_path_hf, env=self.vec_env)
                            elif self.agent_type == 'dqn':
                                model = DQN.load(model_path_hf, env=self.vec_env)
                        except Exception as env_error:
                            # If observation space doesn't match, try loading without env
                            if "Observation spaces do not match" in str(env_error) or "observation space" in str(env_error).lower():
                                print(f"[SB3Wrapper] Observation space mismatch, loading without env...")
                                if self.agent_type == 'ppo':
                                    model = PPO.load(model_path_hf)
                                elif self.agent_type == 'a2c':
                                    model = A2C.load(model_path_hf)
                                elif self.agent_type == 'dqn':
                                    model = DQN.load(model_path_hf)
                                # Try to set the environment, but don't fail if observation spaces don't match
                                try:
                                    model.set_env(self.vec_env)
                                except Exception:
                                    print(f"[SB3Wrapper] Warning: Could not set env due to observation space mismatch, but model should still work for predictions.")
                                    # Store vec_env separately so we can use it for predictions
                                    model._vec_env_custom = self.vec_env
                            else:
                                raise env_error
                        
                        print(f"[SB3Wrapper] Successfully loaded model with filename: {filename}")
                        return model
                    except Exception as e:
                        last_error = str(e)
                        if "404" in last_error or "not found" in last_error.lower():
                            print(f"[SB3Wrapper] File {filename} not found in repo: {e}")
                        elif "No data found" in last_error:
                            print(f"[SB3Wrapper] File {filename} exists but contains no model data (may be metrics file)")
                        else:
                            print(f"[SB3Wrapper] Error loading {filename}: {e}")
                        continue  # Try next filename
                
                # If we tried all files, show the last error
                if last_error:
                    print(f"[SB3Wrapper] Last error encountered: {last_error}")
                
                # If no filename worked, raise an error with helpful info
                try:
                    repo_files = list(list_repo_files(repo_id=model_path, repo_type="model"))
                    raise FileNotFoundError(
                        f"Could not find model file in HuggingFace repo: {model_path}\n"
                        f"  Tried filenames: {', '.join(possible_filenames)}\n"
                        f"  Available files in repo: {', '.join(repo_files)}\n"
                        f"  Please check https://huggingface.co/{model_path} for the correct filename."
                    )
                except:
                    raise FileNotFoundError(
                        f"Could not find model file in HuggingFace repo: {model_path}\n"
                        f"  Tried filenames: {', '.join(possible_filenames)}\n"
                        f"  Please check https://huggingface.co/{model_path} for available files, or use a local model path.\n"
                        f"  To train a new model instead, use --sb3_train flag."
                    )
                
            except ImportError:
                print(f"[SB3Wrapper] huggingface_sb3 not installed. Install with: pip install huggingface-sb3")
                print(f"[SB3Wrapper] Falling back to local path...")
            except Exception as e:
                print(f"[SB3Wrapper] Failed to load from HuggingFace: {e}")
                print(f"[SB3Wrapper] Attempting local load...")
        
        # Try loading as local path
        # Check if path exists (with or without .zip extension)
        model_path_zip = model_path if model_path.endswith('.zip') else f"{model_path}.zip"
        model_path_no_ext = model_path.rstrip('.zip')
        
        # Try different variations of the path
        possible_paths = [model_path, model_path_zip, model_path_no_ext, f"{model_path_no_ext}.zip"]
        found_path = None
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                found_path = path
                break
        
        if found_path is None:
            # Check if it's a directory that might contain the model
            if os.path.isdir(model_path):
                # Look for .zip files in the directory
                zip_files = list(Path(model_path).glob("*.zip"))
                if zip_files:
                    found_path = str(zip_files[0])
                    print(f"[SB3Wrapper] Found model file in directory: {found_path}")
        
        if found_path is None:
            raise FileNotFoundError(
                f"[SB3Wrapper] Model file not found at: {model_path}\n"
                f"  Tried: {', '.join(possible_paths)}\n"
                f"  If you want to train a new model, use --sb3_train flag instead of --sb3_model_path.\n"
                f"  If you want to load from HuggingFace, install: pip install huggingface-sb3"
            )
        
        print(f"[SB3Wrapper] Loading pretrained model from local path: {found_path}")
        if self.agent_type == 'ppo':
            return PPO.load(found_path, env=self.vec_env)
        elif self.agent_type == 'a2c':
            return A2C.load(found_path, env=self.vec_env)
        elif self.agent_type == 'dqn':
            return DQN.load(found_path, env=self.vec_env)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def select_action(self, state, env=None, do_apply_shield=True):
        """
        Select an action using the SB3 model.
        
        Returns:
            Tuple of (selected_action, a_unshielded, a_shielded, context)
            For SB3 agents without shields: a_unshielded == a_shielded == selected_action
        """
        self.last_obs = state
        
        # Build context (needed for compatibility)
        context = context_provider.build_context(env or self.env, self)
        
        # Convert state to format SB3 expects
        # SB3 expects observations in the format from the environment
        # We may need to reshape if state was preprocessed
        obs = self._prepare_obs_for_sb3(state)
        
        # Get action from SB3 model (deterministic for evaluation, stochastic for training)
        # Since we're wrapping SB3, we use deterministic=False to get policy sampling
        action, _states = self.model.predict(obs, deterministic=False)
        
        # SB3 may return action as array, convert to scalar if needed
        if isinstance(action, np.ndarray):
            if action.ndim > 0:
                action = action[0] if len(action) > 0 else action.item()
            else:
                action = action.item()
        
        selected_action = int(action)
        a_unshielded = selected_action
        a_shielded = selected_action  # No shield for SB3 agents
        
        # Store last values for compatibility (SB3 doesn't expose these directly)
        self.last_log_prob = None
        self.last_value = None
        
        return selected_action, a_unshielded, a_shielded, context
    
    def _prepare_obs_for_sb3(self, state):
        """Convert preprocessed state back to format SB3 expects."""
        # Convert state to numpy array
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        elif not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        # Handle frame stacking for Atari/CNN environments
        if self._needs_frame_stack and self.use_cnn:
            # For Atari, SB3 expects (4, 84, 84) - 4 stacked frames
            # state might be (H, W) or (1, H, W) or (C, H, W) or already processed
            
            # Normalize to single frame (H, W) format
            if state.ndim == 4:
                # (1, 1, H, W) or (B, C, H, W) -> extract single frame
                state = state.squeeze()
                if state.ndim == 3:
                    state = state[0]  # Take first channel
                frame = state.squeeze() if state.ndim > 2 else state
            elif state.ndim == 3:
                # (C, H, W) or (1, H, W) -> take first channel or squeeze
                if state.shape[0] == 1:
                    frame = state.squeeze(0)  # (H, W)
                else:
                    frame = state[0]  # Take first channel
            elif state.ndim == 2:
                frame = state  # Already (H, W)
            else:
                frame = state.squeeze()
                if frame.ndim != 2:
                    raise ValueError(f"Cannot process state shape {state.shape} for Atari frame stacking")
            
            # Ensure frame is (H, W) with correct dtype
            if frame.ndim != 2:
                frame = frame.squeeze()
                if frame.ndim != 2:
                    raise ValueError(f"After processing, frame should be (H, W) but got shape {frame.shape}")
            
            # Add frame to buffer (copy to avoid reference issues)
            self._frame_buffer.append(frame.astype(np.float32).copy())
            
            # Keep only last N frames
            if len(self._frame_buffer) > self._frame_stack_size:
                self._frame_buffer.pop(0)
            
            # If we don't have enough frames yet, repeat the first frame
            while len(self._frame_buffer) < self._frame_stack_size:
                self._frame_buffer.insert(0, self._frame_buffer[0] if self._frame_buffer else frame.astype(np.float32))
            
            # Stack frames: (4, H, W) - this is what SB3 expects
            stacked = np.stack(self._frame_buffer, axis=0)
            
            # Verify shape is correct (4, H, W)
            if stacked.shape[0] != 4:
                raise ValueError(f"Expected 4 stacked frames, got {stacked.shape[0]}. Shape: {stacked.shape}")
            
            # SB3's predict() will handle the batch dimension automatically
            # Return (4, 84, 84) without batch dimension
            return stacked
        else:
            # Non-Atari: handle different input shapes
            if state.ndim == 1:
                # Flatten vector observation
                return state.reshape(1, -1)
            elif state.ndim == 2:
                # Already in (batch, features) format or (H, W) - add batch dim if needed
                if state.shape[0] > 100:  # Likely (H, W) for images
                    return state.reshape(1, *state.shape)
                else:
                    return state.reshape(1, -1) if len(state.shape) == 1 else state
            elif state.ndim == 3:
                # Image observation (C, H, W) - return as is, SB3 will handle batch dimension
                return state
            else:
                # Remove extra dimensions
                return state.squeeze() if state.ndim > 3 else state
    
    def store_transition(self, state, action, reward, next_state, context, done):
        """
        Store transition for training.
        
        For pretrained models, this is a no-op.
        For models to be trained, SB3 handles its own replay buffer/rollout collection.
        """
        # SB3 handles its own experience collection during learn()
        # This method exists for interface compatibility
        self.steps_done += 1
        pass
    
    def update(self):
        """
        Update the model.
        
        For pretrained models, this is a no-op.
        For training, SB3 uses batch learning via learn(), but we can also
        accumulate steps and train periodically.
        """
        # SB3 uses learn() method for updates, not individual update() calls
        # This method exists for interface compatibility
        # For training mode, we'll train periodically during run_training
        pass
    
    def learn(self, total_timesteps: int):
        """Train the SB3 model."""
        if self.trained and not self._use_sb3_training:
            print("[SB3Wrapper] Model is already trained (pretrained). Skipping training.")
            return
        
        print(f"\n[SB3Wrapper] Starting training...")
        print(f"[SB3Wrapper] Agent type: {self.agent_type.upper()}")
        print(f"[SB3Wrapper] Total timesteps: {total_timesteps:,}")
        print(f"[SB3Wrapper] Estimated episodes: ~{total_timesteps // 1000}")
        print(f"[SB3Wrapper] Observation space: {self.vec_env.observation_space}")
        print(f"[SB3Wrapper] Action space: {self.vec_env.action_space}")
        
        try:
            import time
            start_time = time.time()
            print(f"[SB3Wrapper] Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[SB3Wrapper] Progress will be shown below...\n")
            
            # SB3's learn() will handle all the training internally
            # It will print progress automatically if verbose=1
            self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=True)
            
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            print(f"\n[SB3Wrapper] Training completed successfully!")
            print(f"[SB3Wrapper] Total training time: {hours}h {minutes}m {seconds}s")
            print(f"[SB3Wrapper] Timesteps trained: {total_timesteps:,}")
            
            self.trained = True
            self.learn_step_counter += total_timesteps
        except KeyboardInterrupt:
            print(f"\n[SB3Wrapper] Training interrupted by user")
            self.trained = False
            raise
        except Exception as e:
            print(f"\n[SB3Wrapper] Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.trained = False
            raise
    
    def get_weights(self):
        """Get model weights (for checkpointing)."""
        # SB3 models store their state internally
        # Return a reference to the model itself for compatibility
        return self.model
    
    def load_weights(self, weights):
        """Load model weights."""
        # If weights is a model, we can't easily reload it
        # For SB3, we typically save/load using model.save() and model.load()
        if isinstance(weights, (PPO, A2C, DQN)):
            self.model = weights
            self.trained = True
        else:
            print("[SB3Wrapper] Warning: Cannot load weights in this format. Use model.save()/load() instead.")
    
    def start_new_episode(self):
        """Start a new episode (for compatibility with some agents)."""
        # Reset frame buffer at start of episode
        self._frame_buffer = []
    
    def save_model(self, save_path: str):
        """Save the SB3 model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n[SB3Wrapper] Saving model to: {save_path}")
        try:
            self.model.save(save_path)
            # Check if file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024 * 1024)  # Size in MB
                print(f"[SB3Wrapper] ✓ Model saved successfully!")
                print(f"[SB3Wrapper]   File: {save_path}")
                print(f"[SB3Wrapper]   Size: {file_size:.2f} MB")
            else:
                print(f"[SB3Wrapper] ⚠ Warning: Model save completed but file not found at {save_path}")
        except Exception as e:
            print(f"[SB3Wrapper] ✗ Error saving model: {e}")
            import traceback
            traceback.print_exc()
            raise


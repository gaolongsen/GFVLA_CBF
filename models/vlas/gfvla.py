"""
gfvla.py

Graph-Fused Vision-Language-Action (GF-VLA) model.
Integrates information-theoretic scene graphs with VLA reasoning.
"""

from __future__ import annotations
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizerFast

from models.backbones import LLMBackbone, VisionBackbone
from models.vlms import PrismaticVLM
from models.diffusion import create_diffusion
from util import FusedMLPProjector, LinearProjector, MLPProjector
from models.diffusion import ActionEmbedder, TimestepEmbedder, LabelEmbedder, FinalLayer
from overwatch import initialize_overwatch
from vla import ActionTokenizer

from models.vlms.pointcloud_processor.gfvla_env_dino import GfvlaEnvDinov2
from models.graphs import SceneGraphExtractor, GraphEncoder, TemporalGraphProcessor

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class ChainOfThoughtPrompter:
    """
    Chain-of-Thought prompting for interpretable subgoal decomposition.
    Based on Wei et al. 2023 "Chain-of-thought prompting elicits reasoning in large language models"
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def build_cot_prompt(
        self,
        instruction: str,
        scene_graph_description: Optional[str] = None
    ) -> str:
        """
        Build a Chain-of-Thought prompt that decomposes the task into subgoals.
        
        Args:
            instruction: High-level task instruction
            scene_graph_description: Optional description of scene graph structure
            
        Returns:
            Formatted CoT prompt
        """
        base_prompt = f"""Task: {instruction}

Let me think step by step:

1. First, I need to understand the current scene state."""
        
        if scene_graph_description:
            base_prompt += f"\n   Scene analysis: {scene_graph_description}"
        
        base_prompt += """
2. Next, I will identify the key objects and their relationships.
3. Then, I will plan the sequence of actions needed to complete the task.
4. Finally, I will execute the actions in the correct order.

Subgoals:"""
        
        return base_prompt
    
    def extract_subgoals(self, response: str) -> List[str]:
        """
        Extract subgoals from CoT response.
        
        Args:
            response: LLM response containing subgoals
            
        Returns:
            List of subgoal strings
        """
        subgoals = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered subgoals or bullet points
            if line.startswith('-') or line.startswith('•') or \
               (line and line[0].isdigit() and '.' in line[:3]):
                # Remove numbering/bullets
                subgoal = line.lstrip('-•0123456789. ').strip()
                if subgoal:
                    subgoals.append(subgoal)
        
        return subgoals


class CrossArmAllocator(nn.Module):
    """
    Cross-arm allocation strategy for dual-arm manipulation.
    Autonomously determines gripper assignment without explicit geometric modeling.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for allocation network
        """
        super().__init__()
        
        self.allocation_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # Output: probability for each arm
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        task_features: torch.Tensor,
        object_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine arm allocation for a task.
        
        Args:
            task_features: Task/instruction features [B, feature_dim]
            object_features: Optional object features [B, N, feature_dim]
            
        Returns:
            arm_assignment: [B, 2] probabilities for each arm
            arm_features: [B, 2, feature_dim] features for each arm
        """
        B = task_features.shape[0]
        
        # Combine task and object features if provided
        if object_features is not None:
            # Pool object features (mean pooling)
            pooled_obj_features = object_features.mean(dim=1)  # [B, feature_dim]
            combined_features = task_features + pooled_obj_features
        else:
            combined_features = task_features
        
        # Predict arm allocation
        arm_assignment = self.allocation_network(combined_features)  # [B, 2]
        
        # Generate arm-specific features
        arm_features = arm_assignment.unsqueeze(-1) * task_features.unsqueeze(1)  # [B, 2, feature_dim]
        
        return arm_assignment, arm_features


class SelfVerifier(nn.Module):
    """
    Self-verification module for execution reliability.
    Checks if generated actions are consistent with the task plan.
    """
    
    def __init__(
        self,
        action_dim: int = 7,
        feature_dim: int = 512,
        hidden_dim: int = 256,
    ):
        """
        Args:
            action_dim: Dimension of action space
            feature_dim: Dimension of task/plan features
            hidden_dim: Hidden dimension for verification network
        """
        super().__init__()
        
        self.verification_network = nn.Sequential(
            nn.Linear(action_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # Output: verification score
            nn.Sigmoid()
        )
    
    def forward(
        self,
        actions: torch.Tensor,
        task_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Verify if actions are consistent with task plan.
        
        Args:
            actions: Generated actions [B, T, action_dim]
            task_features: Task/plan features [B, feature_dim]
            
        Returns:
            verification_scores: [B, T] scores indicating consistency
        """
        B, T, action_dim = actions.shape
        
        # Expand task features to match action sequence
        task_features_expanded = task_features.unsqueeze(1).expand(B, T, -1)  # [B, T, feature_dim]
        
        # Concatenate actions and task features
        combined = torch.cat([actions, task_features_expanded], dim=-1)  # [B, T, action_dim + feature_dim]
        
        # Reshape for processing
        combined = combined.reshape(B * T, -1)
        
        # Compute verification scores
        scores = self.verification_network(combined)  # [B * T, 1]
        scores = scores.reshape(B, T)  # [B, T]
        
        return scores


class GFVLA(nn.Module):
    """
    Graph-Fused Vision-Language-Action model.
    Integrates structured scene graphs with VLA reasoning for robust manipulation.
    
    Hardware Configuration (matching paper):
    - Left Arm: UR5e with Robotiq Gripper
    - Right Arm: UR10e with Barrett BH282 Hand
    
    Action Space:
    - Single-arm: 7 dimensions (6 DOF + 1 gripper)
    - Dual-arm: 14 dimensions (2 × [6 DOF + 1 gripper])
      - [0:7]: UR5e (Robotiq) - [x, y, z, roll, pitch, yaw, gripper]
      - [7:14]: UR10e (Barrett) - [x, y, z, roll, pitch, yaw, gripper]
    """
    
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_tokenizer: ActionTokenizer,
        token_size: int = 4096,
        action_dim: int = 7,  # Single-arm: 7, Dual-arm: 14
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        use_diff: bool = False,
        diffusion_steps: int = 100,
        load_pointcloud: bool = False,
        action_chunk: int = 1,
        load_state: bool = True,
        action_tokenizer_exist: bool = False,
        lang_subgoals_exist: bool = False,
        graph_embed_dim: int = 512,
        enable_cot: bool = True,
        enable_verification: bool = True,
        enable_cross_arm: bool = True,
        dual_arm: bool = False,  # Enable dual-arm mode
        **kwargs,
    ) -> None:
        super().__init__()

        self.action_tokenizer = action_tokenizer
        self.load_pointcloud = load_pointcloud
        self.action_chunk = action_chunk
        self.load_state = load_state
        self.action_tokenizer_exist = action_tokenizer_exist
        self.lang_subgoals_exist = lang_subgoals_exist
        self.enable_cot = enable_cot
        self.enable_verification = enable_verification
        self.enable_cross_arm = enable_cross_arm
        self.dual_arm = dual_arm
        self.action_dim = action_dim
        
        # For dual-arm, action_dim should be 14 (2 × 7)
        if dual_arm and action_dim != 14:
            overwatch.log_warning(
                f"Dual-arm mode enabled but action_dim={action_dim}. "
                f"Expected 14 for dual-arm. Updating to 14."
            )
            self.action_dim = 14

        self.use_diff = use_diff
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.vlm.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.all_module_keys = []
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)
        self.norm_stats = norm_stats
        self._trainable_module_keys = []

        # Graph processing modules
        self.scene_graph_extractor = SceneGraphExtractor()
        self.graph_encoder = GraphEncoder(output_dim=graph_embed_dim)
        self.temporal_graph_processor = TemporalGraphProcessor(graph_embed_dim=graph_embed_dim)
        
        # Graph-VLA fusion
        self.graph_fusion = nn.Sequential(
            nn.Linear(graph_embed_dim + token_size, token_size),
            nn.LayerNorm(token_size),
            nn.GELU(),
            nn.Linear(token_size, token_size),
        )
        
        # Chain-of-Thought prompter
        if enable_cot:
            self.cot_prompter = ChainOfThoughtPrompter(self.vlm.llm_backbone.tokenizer)
        
        # Cross-arm allocator
        if enable_cross_arm:
            self.cross_arm_allocator = CrossArmAllocator(feature_dim=token_size)
        
        # Self-verifier
        if enable_verification:
            self.self_verifier = SelfVerifier(action_dim=action_dim, feature_dim=token_size)
        
        if self.use_diff:
            self.ddim_diffusion = None
            self.diffusion_steps = diffusion_steps
            self.diffusion = create_diffusion(
                timestep_respacing="",
                noise_schedule='squaredcos_cap_v2',
                diffusion_steps=self.diffusion_steps,
                sigma_small=True,
                learn_sigma=False
            )

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        # Add graph modules to trainable keys
        keys.extend([
            "graph_encoder",
            "temporal_graph_processor",
            "graph_fusion",
        ])
        if self.enable_cross_arm:
            keys.append("cross_arm_allocator")
        if self.enable_verification:
            keys.append("self_verifier")
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def extract_and_encode_graphs(
        self,
        hand_positions: Optional[np.ndarray] = None,
        object_positions: Optional[np.ndarray] = None,
        hand_features: Optional[np.ndarray] = None,
        object_features: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Extract scene graphs from demonstration data and encode them.
        
        Args:
            hand_positions: [T, H, 3] hand positions
            object_positions: [T, O, 3] object positions
            hand_features: Optional [T, H, D] hand features
            object_features: Optional [T, O, D] object features
            
        Returns:
            Graph embeddings [T, graph_embed_dim]
        """
        if hand_positions is None or object_positions is None:
            # Return zero embeddings if no graph data
            device = next(self.graph_encoder.parameters()).device
            return torch.zeros(1, self.graph_encoder.output_dim, device=device)
        
        # Extract scene graphs
        scene_graphs = self.scene_graph_extractor.extract_from_demonstration(
            hand_positions=hand_positions,
            object_positions=object_positions,
            hand_features=hand_features,
            object_features=object_features,
        )
        
        # Encode each graph
        graph_embeddings = []
        for scene_graph in scene_graphs:
            graph_emb = self.graph_encoder(scene_graph)
            graph_embeddings.append(graph_emb)
        
        # Stack and process temporally
        graph_embeddings = torch.stack(graph_embeddings)  # [T, graph_embed_dim]
        graph_embeddings = self.temporal_graph_processor(graph_embeddings)  # [T, graph_embed_dim]
        
        return graph_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 3,
        action_masks=None,
        use_diff: Optional[bool] = None,
        graph_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass through GF-VLA.
        
        Args:
            graph_embeddings: Optional pre-computed graph embeddings [T, graph_embed_dim]
            ... (other args same as FiSvla)
        """
        # Update diffusion mode flag
        if use_diff is not None:
            self.use_diff = use_diff
        
        # Repeat proprio inputs
        if self.load_state:
            proprio = self._repeat_tensor(proprio, repeated_diffusion_steps) if proprio is not None else None

        # Diffusion-specific processing
        if self.use_diff:
            # Repeat inputs
            actions = self._repeat_tensor(actions, repeated_diffusion_steps) if actions is not None else None
            input_ids = self._repeat_tensor(input_ids, repeated_diffusion_steps) if input_ids is not None else None
            attention_mask = self._repeat_tensor(attention_mask, repeated_diffusion_steps) if attention_mask is not None else None
            action_masks = self._repeat_tensor(action_masks, repeated_diffusion_steps) if action_masks is not None else None
            labels = self._repeat_tensor(labels, repeated_diffusion_steps) if labels is not None else None
            
            # Repeat pixel values
            pixel_values = self._repeat_pixel_values(pixel_values, repeated_diffusion_steps) if pixel_values is not None else None

            if point_cloud is not None:
                point_cloud = self._repeat_tensor(point_cloud, repeated_diffusion_steps)

            # Generate noise and timesteps for diffusion
            noise = torch.randn_like(actions)  # [B, T, C]
            timestep = torch.randint(
                0, 
                self.diffusion.num_timesteps, 
                (actions.size(0),), 
                device=actions.device
            )
            
            # Apply diffusion sampling
            x = self.diffusion.q_sample(actions, timestep, noise)

            # Fuse graph embeddings with VLM if provided
            if graph_embeddings is not None:
                # Get VLM embeddings (we'll need to modify VLM forward to return embeddings)
                # For now, we'll fuse at the input level
                pass  # Graph fusion will be handled in VLM forward

            # Run VLM with diffusion
            output, noise_pred = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                point_cloud=point_cloud,
                labels=labels,
                x=x,
                t=timestep,
                proprio=proprio,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_diff=self.use_diff,
            )

            # Compute loss
            assert noise_pred.shape == noise.shape == actions.shape
            loss = ((noise_pred - noise) ** 2).mean()
            
            # Add verification loss if enabled
            if self.enable_verification and self.training:
                if graph_embeddings is not None:
                    task_features = graph_embeddings.mean(dim=0).unsqueeze(0)  # [1, graph_embed_dim]
                else:
                    # Use zero features if no graph embeddings
                    task_features = torch.zeros(1, self.graph_encoder.output_dim, device=actions.device)
                verification_scores = self.self_verifier(actions, task_features)
                # Encourage high verification scores
                verification_loss = (1.0 - verification_scores.mean()) * 0.1
                loss = loss + verification_loss
            
            return loss, output

        # Non-diffusion mode
        else:
            output = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                point_cloud=point_cloud,
                labels=labels,
                proprio=proprio,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_diff=self.use_diff,
            )
            return output

    def _repeat_tensor(
        self, 
        tensor: Optional[torch.Tensor], 
        repeated_diffusion_steps: int
    ) -> Optional[torch.Tensor]:
        """Repeat a tensor along the first dimension."""
        if tensor is None:
            return None
        return tensor.repeat(repeated_diffusion_steps, *([1] * (tensor.ndimension() - 1)))

    def _repeat_pixel_values(
        self, 
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor], None], 
        repeated_diffusion_steps: int
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], None]:
        """Repeat pixel values, handling different input types."""
        if pixel_values is None:
            return None

        if isinstance(pixel_values, torch.Tensor):
            return pixel_values.repeat(repeated_diffusion_steps, *([1] * (pixel_values.ndimension() - 1)))
        
        if isinstance(pixel_values, dict):
            return {
                key: value.repeat(repeated_diffusion_steps, *([1] * (value.ndimension() - 1)))
                for key, value in pixel_values.items()
            }
        
        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={
                LinearProjector, MLPProjector, FusedMLPProjector,
                ActionEmbedder, TimestepEmbedder, LabelEmbedder, FinalLayer,
                GraphEncoder, TemporalGraphProcessor, CrossArmAllocator, SelfVerifier
            },
        )

        policies = [
            vision_fsdp_wrapping_policy,
            llm_fsdp_wrapping_policy,
            prismatic_fsdp_wrapping_policy,
        ]

        if self.load_pointcloud:
            pointcloud_fsdp_wrapping_policy = self.vlm.pointcloud_backbone.get_fsdp_wrapping_policy()
            policies.append(pointcloud_fsdp_wrapping_policy)

        return partial(_or_policy, policies=policies)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_proprio_stats(self, unnorm_key=None):
        """Dimensionality of the policy's proprio space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["proprio"]
    
    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        pointcloud_backbone: Optional[GfvlaEnvDinov2] = None,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        norm_stats=None,
        class_dropout_prob: float = 0.0,
        need_to_sub: int = 0,
        use_diff: bool = False,
        llm_middle_layer: int = 32,
        diffusion_steps: int = 100,
        action_tokenizer_exist: bool = False,
        training_mode: str = "",
        load_pointcloud: bool = False,
        pointcloud_pos: str = "slow",
        action_chunk: int = 1,
        load_state: bool = True,
        lang_subgoals_exist: bool = False,
        graph_embed_dim: int = 512,
        enable_cot: bool = True,
        enable_verification: bool = True,
        enable_cross_arm: bool = True,
        **kwargs,
    ) -> 'GFVLA':
        """Load GF-VLA from pretrained checkpoint."""
        # Load VLM backbone
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            pointcloud_backbone=pointcloud_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            class_dropout_prob=class_dropout_prob,
            use_diff=use_diff,
            action_dim=action_dim,
            token_size=llm_backbone.embed_dim,
            llm_middle_layer=llm_middle_layer,
            action_tokenizer_exist=action_tokenizer_exist,
            training_mode=training_mode,
            load_pointcloud=load_pointcloud,
            pointcloud_pos=pointcloud_pos,
            action_chunk=action_chunk,
            load_state=load_state,
            **kwargs,
        )
        
        # Load action tokenizer
        action_tokenizer = ActionTokenizer(llm_backbone.get_tokenizer(), need_to_sub)

        # Load from Checkpoint
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]

        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
        else:
            raise ValueError("no vision backbone found!")

        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])

        if load_state and "proprio_embedder" in model_state_dict.keys() and model_state_dict["proprio_embedder"]["mlp.fc1.weight"].shape[-1] == action_dim:
            vlm.proprio_embedder.load_state_dict(model_state_dict["proprio_embedder"])

        if "x_embedder" in model_state_dict.keys() and "t_embedder" in model_state_dict.keys() and "final_layer" in model_state_dict.keys() and use_diff:
            if model_state_dict["x_embedder"]["mlp.fc1.weight"].shape[-1] == action_dim:
                vlm.x_embedder.load_state_dict(model_state_dict["x_embedder"])
                vlm.t_embedder.load_state_dict(model_state_dict["t_embedder"])
                vlm.final_layer.load_state_dict(model_state_dict["final_layer"])

        if load_pointcloud and "pointcloud_backbone" in model_state_dict.keys() and "projector_3d_dino" in model_state_dict.keys() and "projector_3d_siglip" in model_state_dict.keys():
            vlm.pointcloud_backbone.load_state_dict(model_state_dict["pointcloud_backbone"])
            vlm.projector_3d_dino.load_state_dict(model_state_dict["projector_3d_dino"])
            vlm.projector_3d_siglip.load_state_dict(model_state_dict["projector_3d_siglip"])

        # Load graph modules if available
        # (These would be in the checkpoint if model was trained with graphs)

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize GFVLA
        gfvla = GFVLA(
            vlm,
            action_tokenizer,
            token_size=vlm.llm_backbone.llm.lm_head.in_features,
            action_dim=action_dim,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            norm_stats=norm_stats,
            use_diff=use_diff,
            diffusion_steps=diffusion_steps,
            load_pointcloud=load_pointcloud,
            action_chunk=action_chunk,
            load_state=load_state,
            action_tokenizer_exist=action_tokenizer_exist,
            lang_subgoals_exist=lang_subgoals_exist,
            graph_embed_dim=graph_embed_dim,
            enable_cot=enable_cot,
            enable_verification=enable_verification,
            enable_cross_arm=enable_cross_arm,
        )

        return gfvla


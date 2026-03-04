"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.backbones.llm import LLMBackbone
from models.backbones.llm.prompting import PromptBuilder
from models.backbones.vision import VisionBackbone
from models.vlms.base_vlm import VLM
from overwatch import initialize_overwatch
from util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from models.diffusion import ActionEmbedder, TimestepEmbedder, LabelEmbedder, FinalLayer

import random

from .pointcloud_processor.gfvla_env_dino import GfvlaEnvDinov2

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        pointcloud_backbone: Optional[GfvlaEnvDinov2] = None,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        action_dim = 7,
        token_size = 4096,
        future_action_window_size=0,
        past_action_window_size=0,
        class_dropout_prob=0.0,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        use_diff = False,
        llm_middle_layer = 32,
        action_tokenizer_exist=False,
        training_mode="async",
        load_pointcloud: bool = False,
        pointcloud_pos: str = "slow",
        action_chunk: int = 1,
        load_state: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )
        self.use_diff = use_diff

        self.llm_middle_layer = llm_middle_layer
        self.training_mode = training_mode
        self.action_tokenizer_exist = action_tokenizer_exist
        self.model_id = model_id

        self.action_chunk=action_chunk
        self.load_state = load_state

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)
        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")
        # Trackers
        self.vision_backbone_requires_grad = False


        self.load_pointcloud = load_pointcloud
        self.pointcloud_pos = pointcloud_pos
        if self.load_pointcloud:
            self.pointcloud_backbone = pointcloud_backbone
            if arch_specifier == "linear":
                self.projector_3d_dino = LinearProjector(self.pointcloud_backbone.feature_dim, vision_backbone.embed_dim_dino)
                self.projector_3d_siglip = LinearProjector(self.pointcloud_backbone.feature_dim, vision_backbone.embed_dim_siglip)
            elif arch_specifier.endswith("fused-gelu-mlp"):
                self.projector_3d_dino = FusedMLPProjector(self.pointcloud_backbone.feature_dim, vision_backbone.embed_dim_dino)
                self.projector_3d_siglip = FusedMLPProjector(self.pointcloud_backbone.feature_dim, vision_backbone.embed_dim_siglip)
            elif arch_specifier.endswith("gelu-mlp"):
                self.projector_3d_dino = MLPProjector(self.pointcloud_backbone.feature_dim, vision_backbone.embed_dim_dino)
                self.projector_3d_siglip = MLPProjector(self.pointcloud_backbone.feature_dim, vision_backbone.embed_dim_siglip)
            else:
                raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")
            self.pointcloud_backbone_requires_grad = True

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]


        self.norm_stats = norm_stats
        self.class_dropout_prob = class_dropout_prob
        self.future_action_window_size = future_action_window_size
        self.action_dim = action_dim

        if self.load_state:
            self.proprio_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)

        if self.use_diff:
            self.x_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)
            self.t_embedder = TimestepEmbedder(token_size)
            self.final_layer = FinalLayer(token_size, action_dim)

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        if load_pointcloud:
            self.all_module_keys.extend(["pointcloud_backbone", "projector_3d_dino", "projector_3d_siglip"])
        if self.use_diff:
            self.all_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
        if self.load_state:
            self.all_module_keys.extend(["proprio_embedder"])

        self.trainable_module_keys = []

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_diff:
            nn.init.normal_(self.x_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.x_embedder.mlp.fc2.weight, std=0.02)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            nn.init.constant_(self.final_layer.mlp.fc2.weight, 0)
            nn.init.constant_(self.final_layer.mlp.fc2.bias, 0)

        if self.load_state:
            nn.init.normal_(self.proprio_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.proprio_embedder.mlp.fc2.weight, std=0.02)

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
        class_dropout_prob: float = 0.0,
        use_diff: bool = False,
        llm_middle_layer: int = 32,
        action_tokenizer_exist: bool=False,
        training_mode: str = "async",
        load_pointcloud: bool = False,
        pointcloud_pos: str = 'slow',
        action_chunk: int = 1,
        load_state: bool = True,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            pointcloud_backbone=pointcloud_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            class_dropout_prob=class_dropout_prob,
            use_diff=use_diff,
            action_dim=action_dim,
            llm_middle_layer=llm_middle_layer,
            action_tokenizer_exist=action_tokenizer_exist,
            training_mode=training_mode,
            load_pointcloud=load_pointcloud,
            pointcloud_pos=pointcloud_pos,
            action_chunk=action_chunk,
            load_state=load_state,
            **kwargs,
        )

        if not isinstance(pretrained_checkpoint, dict):
            # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        else:
            model_state_dict = pretrained_checkpoint
        
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)
            if self.load_pointcloud:
                self.pointcloud_backbone.requires_grad_(True)
                self.projector_3d.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "projector_3d", "pointcloud_backbone"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.load_state:
                self.trainable_module_keys.extend(["proprio_embedder"])
            # Update Trackers
            self.vision_backbone_requires_grad = False
            self.pointcloud_backbone_requires_grad = True

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🔥 =>> Point Cloud Backbone ", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector_3d `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"finetune", "vla-train"}:
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            if self.load_pointcloud:
                self.pointcloud_backbone.requires_grad_(True)
                self.projector_3d_dino.requires_grad_(True)
                self.projector_3d_siglip.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone", "projector_3d_dino", "projector_3d_siglip", "pointcloud_backbone"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.load_state:
                self.trainable_module_keys.extend(["proprio_embedder"])
            # Update Trackers
            self.vision_backbone_requires_grad = False
            self.pointcloud_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🔥 =>> Point Cloud Backbone ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector_3d `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            if self.load_pointcloud:
                self.pointcloud_backbone.requires_grad_(True)
                self.projector_3d_dino.requires_grad_(True)
                self.projector_3d_siglip.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone", "pointcloud_backbone", "projector_3d_dino", "projector_3d_siglip"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.load_state:
                self.trainable_module_keys.extend(["proprio_embedder"])

            # Update Trackers
            self.vision_backbone_requires_grad = True
            self.pointcloud_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Point Cloud Backbone ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector_3d `{self.arch_specifier}`", ctx_level=1)

            
        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            if self.load_pointcloud:
                self.pointcloud_backbone.requires_grad_(False)
                self.projector_3d_dino.requires_grad_(False)
                self.projector_3d_siglip.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.load_state:
                self.trainable_module_keys.extend(["proprio_embedder"])

            # Update Trackers
            self.vision_backbone_requires_grad = False
            self.pointcloud_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Point Cloud Backbone ", ctx_level=1)
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]                    🥶   =>> Projector_3d `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)
            if self.load_pointcloud:
                self.pointcloud_backbone.requires_grad_(True)
                self.projector_3d_dino.requires_grad_(True)
                self.projector_3d_siglip.requires_grad_(True)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone", "pointcloud_backbone", "projector_3d_dino", "projector_3d_siglip"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.load_state:
                self.trainable_module_keys.extend(["proprio_embedder"])

            # fmt: on
            self.vision_backbone_requires_grad = True
            self.pointcloud_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Point Cloud Backbone ", ctx_level=1)
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector_3d `{self.arch_specifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        return None
    
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, ActionEmbedder, TimestepEmbedder, LabelEmbedder, FinalLayer},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        policies=[
            vision_fsdp_wrapping_policy,
            llm_fsdp_wrapping_policy,
            prismatic_fsdp_wrapping_policy,
        ]

        if self.load_pointcloud:
            pointcloud_fsdp_wrapping_policy = self.pointcloud_backbone.get_fsdp_wrapping_policy()
            policies.append(pointcloud_fsdp_wrapping_policy)

        return partial(
            _or_policy,
            policies=policies,
        )
    
    def _get_cognition(self, 
                    input_ids: Optional[torch.LongTensor] = None,
                    pixel_values: Optional[torch.FloatTensor] = None,
                    multimodal_indices: Optional[torch.LongTensor] = None,
                    ):
    
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                prefixes = [
                    ('head_slow_', 'slow'),
                    ('head_fast_', 'fast'),
                    ('right_slow_', 'slow'),
                    ('right_fast_', 'fast'),
                    ('left_slow_', 'slow'),
                    ('left_fast_', 'fast')
                ]
                
                slow_patch_embeddings_list = []
                fast_patch_embeddings_list = []
                
                for prefix, speed in prefixes:
                    filtered_keys = [k for k in pixel_values.keys() if k.startswith(prefix)]
                    if not filtered_keys:
                        continue
                    
                    sub_dict = {k[len(prefix):]: pixel_values[k] for k in filtered_keys}
                    patch_features = self.vision_backbone(
                        {k: sub_dict[k][multimodal_indices] for k in sub_dict}
                    )
                    
                    with torch.set_grad_enabled(self.training):
                        patch_embeddings = self.projector(patch_features)

                    if speed == 'slow':
                        slow_patch_embeddings_list.append(patch_embeddings)
                    elif speed == 'fast':
                        fast_patch_embeddings_list.append(patch_embeddings)
                    else:
                        raise ValueError("Not slow or fast image!!!")
                
                slow_projected_patch_embeddings = (
                    torch.cat(slow_patch_embeddings_list, dim=1) 
                    if len(slow_patch_embeddings_list) > 1 
                    else slow_patch_embeddings_list[0] 
                    if len(slow_patch_embeddings_list) == 1 
                    else None
                )
                fast_projected_patch_embeddings = (
                    torch.cat(fast_patch_embeddings_list, dim=1) 
                    if len(fast_patch_embeddings_list) > 1 
                    else fast_patch_embeddings_list[0] 
                    if len(fast_patch_embeddings_list) == 1 
                    else None
                )
            else:
                raise ValueError("No dict in pixel values!!!")

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        return input_embeddings, slow_projected_patch_embeddings, fast_projected_patch_embeddings

    def _get_model_tags(self) -> Tuple[int, int, int]:
        """
        Get specific tags based on model ID and training state
        
        Returns:
            Tuple of tags for different model configurations
        """
        
        # add_action_dim = self.action_dim if self.action_tokenizer_exist else 0

        if self.model_id == 'prism-dinosiglip-224px+7b':
            # return (2, 0, 2 + add_action_dim) if self.training else (32001, 0, 0)  ## EOD(32002) + EOS(2) ## + action_dim
            return (32001, 0, 0) 
        elif self.model_id == 'phi-2+3b':
            # return (50256, 0, 2 + add_action_dim) if self.training else (50296, 0, 0) ## EOD(50297) + EOS(50256) ## + action_dim
            return (50296, 0, 0)
        raise ValueError(f"Unsupported model: {self.model_id}")

    def get_pointcloud_embedding(
        self, 
        point_cloud,  
    ):
        if self.pointcloud_backbone is None or self.projector_3d_dino is None or self.projector_3d_siglip is None or point_cloud is None:
            raise ValueError
        
        with torch.set_grad_enabled(self.pointcloud_backbone_requires_grad):
            pc_tokens = self.pointcloud_backbone(point_cloud)
        pointcloud_projected_patch_embeddings_dino = self.projector_3d_dino(pc_tokens['dino'])
        pointcloud_projected_patch_embeddings_siglip = self.projector_3d_siglip(pc_tokens['siglip'])
        pointcloud_projected_patch_embeddings = {'dino': pointcloud_projected_patch_embeddings_dino, 'siglip': pointcloud_projected_patch_embeddings_siglip}
        pointcloud_projected_patch_embeddings = self.vision_backbone(pointcloud_projected_patch_embeddings, pc_forward = True)
        pointcloud_projected_patch_embeddings = self.projector(pointcloud_projected_patch_embeddings)
        return pointcloud_projected_patch_embeddings
    
    def _prepare_multimodal_inputs(
        self, 
        input_ids: torch.Tensor, 
        pixel_values: torch.Tensor, 
        point_cloud: torch.Tensor,
        multimodal_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare multimodal inputs and extract cognitive embeddings
        
        Args:
            input_ids: Input token IDs
            pixel_values: Image pixel values
            multimodal_indices: Multimodal indices
        
        Returns:
            Processed input embeddings and projected patch embeddings
        """
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        input_embeddings, slow_projected_patch_embeddings, fast_projected_patch_embeddings = self._get_cognition(
            input_ids, pixel_values, multimodal_indices
        )

        pointcloud_projected_patch_embeddings=None
        if point_cloud is not None:
            point_cloud = point_cloud.to(input_embeddings.device) 
            pointcloud_projected_patch_embeddings = self.get_pointcloud_embedding(point_cloud)

        if self.llm_middle_layer == 32 or self.training_mode == 'sync':
            slow_llm_embeddings = torch.cat([
                input_embeddings[:, :1, :], 
                pointcloud_projected_patch_embeddings if pointcloud_projected_patch_embeddings is not None and self.pointcloud_pos=='slow'
                    else torch.empty(input_embeddings.shape[0], 0, input_embeddings.shape[2], dtype=input_embeddings.dtype).to(input_embeddings.device),
                fast_projected_patch_embeddings, 
                input_embeddings[:, 1:, :]
            ], dim=1)
            return input_embeddings, slow_llm_embeddings, fast_projected_patch_embeddings, fast_projected_patch_embeddings, pointcloud_projected_patch_embeddings
        else:
            slow_llm_embeddings = torch.cat([
                input_embeddings[:, :1, :], 
                pointcloud_projected_patch_embeddings if pointcloud_projected_patch_embeddings is not None and self.pointcloud_pos=='slow'
                    else torch.empty(input_embeddings.shape[0], 0, input_embeddings.shape[2], dtype=input_embeddings.dtype).to(input_embeddings.device),
                slow_projected_patch_embeddings, 
                input_embeddings[:, 1:, :]
            ], dim=1)
            return input_embeddings, slow_llm_embeddings, slow_projected_patch_embeddings, fast_projected_patch_embeddings, pointcloud_projected_patch_embeddings

    def _handle_cache_forward(
        self, 
        input_ids: torch.Tensor, 
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        slow_past_key_values: Optional[List[torch.FloatTensor]] = None,
        gen_discret_action: Optional[torch.LongTensor] = None,
        ar_infer: Optional[bool] = None,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        Handle forward propagation with cache
        
        Args:
            input_ids: Input token IDs
                past_key_values: Previous key-value states
            kwargs: Additional keyword arguments
        
        Returns:
            Output from LLM backbone if cache is used
        """
        if input_ids.shape[1] == 1 and past_key_values is not None:
            if self.llm_middle_layer!=32 and slow_past_key_values is not None:
                output: CausalLMOutputWithPast = self.llm_backbone(
                    input_ids=input_ids,
                    past_key_values=slow_past_key_values,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    llm_layer_start=0,
                    llm_layer_end=self.llm_middle_layer,
                    **{k: v for k, v in kwargs.items() if v is not None}
                )
                slow_past_key_values = output.past_key_values

            output: CausalLMOutputWithPast = self.llm_backbone(
                inputs_embeds=output.hidden_states[-1] if self.llm_middle_layer!=32 else input_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                llm_layer_start=self.llm_middle_layer if self.llm_middle_layer!=32 else 0,
                llm_layer_end=32,
                **{k: v for k, v in kwargs.items() if v is not None}
            )
            output.slow_past_key_values = slow_past_key_values
            return output
        
        elif past_key_values is not None and self.use_diff and not gen_discret_action and not ar_infer:
            t = self.t_embedder(t).unsqueeze(1) if t is not None else None
            x = self.x_embedder(x)
            inputs_embeds = torch.cat([t, x], dim=1)
            past_key_values = tuple(
                (k[:, :, :-(t.shape[1]+x.shape[1]), :], v[:, :, :-(t.shape[1]+x.shape[1]), :]) for k, v in past_key_values
            )
            output = self.llm_backbone(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                llm_layer_start=self.llm_middle_layer if self.llm_middle_layer!=32 else 0,
                llm_layer_end=32,
                **{k: v for k, v in kwargs.items() if v is not None}
            )
            last_hidden = output.hidden_states[-1]
            last_hidden = self.final_layer(last_hidden)
            action_out = []
            for i, indices in enumerate(range(len(input_ids))):
                action_out.append(last_hidden[i, t.shape[1] : t.shape[1]+x.shape[1], :].unsqueeze(0)) # [B, A, D]
            action_out = torch.cat(action_out, dim=0)
            return output, action_out

        return None

    def _prepare_multimodal_embeddings(
        self, 
        slow_llm_embeddings: torch.Tensor, 
        input_ids: torch.Tensor, 
        multimodal_indices: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        slow_projected_patch_embeddings: torch.Tensor = None,
        fast_projected_patch_embeddings: torch.Tensor = None,
        pointcloud_projected_patch_embeddings: torch.Tensor = None,
    ):
        """
        Prepare multimodal embeddings with complex logic from original implementation
        
        Args:
            Various input tensors and embeddings
        
        Returns:
            Prepared multimodal embeddings, attention masks, and labels
        """
        tag_0, tag_1, tag_2 = self._get_model_tags()

        multimodal_embeddings = []
        multimodal_attention_mask = []
        multimodal_labels = []
        last_true_indices = []

        has_slow = slow_projected_patch_embeddings is not None
        has_fast = fast_projected_patch_embeddings is not None
        has_pc = pointcloud_projected_patch_embeddings is not None
        
        if has_slow:
            slow_vs_emb_shape0 = slow_projected_patch_embeddings.shape[0]
            slow_vs_emb_shape1 = slow_projected_patch_embeddings.shape[1]

        if has_pc:
            pointcloud_emb_shape0 = pointcloud_projected_patch_embeddings.shape[0]
            pointcloud_emb_shape1 = pointcloud_projected_patch_embeddings.shape[1]

        if has_fast:
            fast_vs_emb_shape0 = fast_projected_patch_embeddings.shape[0]
            fast_vs_emb_shape1 = fast_projected_patch_embeddings.shape[1]
            if has_pc and self.load_pointcloud and self.pointcloud_pos=='fast':
                slow_llm_embeddings = torch.cat([
                    slow_llm_embeddings[:, :1+slow_vs_emb_shape1],
                    pointcloud_projected_patch_embeddings,
                    fast_projected_patch_embeddings, 
                    slow_llm_embeddings[:, 1+slow_vs_emb_shape1:],
                ], dim=1)
            elif has_pc and self.load_pointcloud and self.pointcloud_pos=='slow':
                slow_llm_embeddings = torch.cat([
                    slow_llm_embeddings[:, :1+pointcloud_emb_shape1+slow_vs_emb_shape1],
                    fast_projected_patch_embeddings, 
                    slow_llm_embeddings[:, 1+pointcloud_emb_shape1+slow_vs_emb_shape1:],
                ], dim=1)
            else:
                slow_llm_embeddings = torch.cat([
                    slow_llm_embeddings[:, :1+slow_vs_emb_shape1],
                    fast_projected_patch_embeddings, 
                    slow_llm_embeddings[:, 1+slow_vs_emb_shape1:],
                ], dim=1)

        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (slow_vs_emb_shape0, slow_vs_emb_shape1),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            if has_fast:
                fast_projected_patch_attention_mask = torch.full(
                    (fast_vs_emb_shape0, fast_vs_emb_shape1),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            if has_pc:
                pointcloud_projected_patch_attention_mask = torch.full(
                    (pointcloud_emb_shape0, pointcloud_emb_shape1),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
        
        projected_patch_labels = None
        if labels is not None: 
            projected_patch_labels = torch.full(
                (slow_vs_emb_shape0, slow_vs_emb_shape1),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )

            if has_fast:
                fast_projected_patch_labels = torch.full(
                    (fast_vs_emb_shape0, fast_vs_emb_shape1),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )

            if has_pc:
                pointcloud_projected_patch_labels = torch.full(
                    (pointcloud_emb_shape0, pointcloud_emb_shape1),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
        for indice in multimodal_indices:
            if self.use_diff and not self.gen_discret_action:
                last_true_indice = torch.where(input_ids[indice] == tag_0)[tag_1][-1] + slow_vs_emb_shape1
                if has_fast: last_true_indice += fast_vs_emb_shape1
                if has_pc: last_true_indice += pointcloud_emb_shape1
                last_true_indices.append(last_true_indice)

            use_complex_mode = (self.use_diff and not self.gen_discret_action 
                            and x is not None and t is not None)
            dynamic_offset = (last_true_indice + 1 - tag_2) - slow_vs_emb_shape1
            if has_fast:
                dynamic_offset -= fast_vs_emb_shape1
            if has_pc:
                dynamic_offset -= pointcloud_emb_shape1

            if use_complex_mode:
                embed = torch.cat([
                    slow_llm_embeddings[indice, :last_true_indice + 1 - tag_2, :],
                    proprio[indice] if self.load_state and proprio is not None else torch.empty(0, slow_llm_embeddings.shape[2], dtype=torch.bool, device=slow_llm_embeddings.device),
                    t[indice],
                    x[indice],
                    slow_llm_embeddings[indice, last_true_indice + 1 - tag_2:, :],
                ], dim=0).unsqueeze(0)
                multimodal_embeddings.append(embed)
            else:
                multimodal_embeddings.append(slow_llm_embeddings[indice].unsqueeze(0))

            if attention_mask is not None:
                if use_complex_mode:
                    if self.pointcloud_pos=='slow':
                        attn_components = [
                            attention_mask[indice, :1],
                            pointcloud_projected_patch_attention_mask[indice] if has_pc 
                                else torch.empty(0, dtype=torch.bool, device=attention_mask.device),
                            projected_patch_attention_mask[indice],
                            fast_projected_patch_attention_mask[indice] if has_fast 
                                else torch.empty(0, dtype=torch.bool, device=attention_mask.device),
                            attention_mask[indice, 1:dynamic_offset],
                            torch.ones((proprio.shape[1]), dtype=torch.bool).to(attention_mask.device) if self.load_state and proprio is not None else torch.empty(0, dtype=torch.bool, device=attention_mask.device),
                            torch.ones((t.shape[1]), dtype=torch.bool).to(attention_mask.device),
                            torch.ones((x.shape[1]), dtype=torch.bool).to(attention_mask.device),
                            attention_mask[indice, dynamic_offset:]
                        ]
                    elif self.pointcloud_pos=='fast':
                        attn_components = [
                            attention_mask[indice, :1],
                            projected_patch_attention_mask[indice],
                            pointcloud_projected_patch_attention_mask[indice] if has_pc 
                                else torch.empty(0, dtype=torch.bool, device=attention_mask.device),
                            fast_projected_patch_attention_mask[indice] if has_fast 
                                else torch.empty(0, dtype=torch.bool, device=attention_mask.device),
                            attention_mask[indice, 1:dynamic_offset],
                            torch.ones((proprio.shape[1]), dtype=torch.bool).to(attention_mask.device) if self.load_state and proprio is not None else torch.empty(0, dtype=torch.bool, device=attention_mask.device),
                            torch.ones((t.shape[1]), dtype=torch.bool).to(attention_mask.device),
                            torch.ones((x.shape[1]), dtype=torch.bool).to(attention_mask.device),
                            attention_mask[indice, dynamic_offset:]
                        ]
                else:
                    attn_components = [
                        attention_mask[indice, :1],
                        pointcloud_projected_patch_attention_mask[indice] if has_pc 
                            else torch.empty(0, dtype=torch.bool, device=attention_mask.device),
                        projected_patch_attention_mask[indice],
                        attention_mask[indice, 1:]
                    ]
                multimodal_attention_mask.append(torch.cat([c for c in attn_components if c is not None], dim=0).unsqueeze(0))

            if labels is not None:
                if use_complex_mode:
                    if self.pointcloud_pos=='slow':
                        label_components = [
                            labels[indice, :1],
                            pointcloud_projected_patch_labels[indice] if has_pc 
                                else torch.empty(0, dtype=labels.dtype, device=labels.device),
                            projected_patch_labels[indice],
                            fast_projected_patch_labels[indice] if has_fast 
                                else torch.empty(0, dtype=labels.dtype, device=labels.device),
                            labels[indice, 1:dynamic_offset],
                            torch.full((proprio.shape[1],), -100).to(labels.device) if self.load_state and proprio is not None else torch.empty(0, dtype=torch.bool, device=labels.device),
                            torch.full((t.shape[1],), -100).to(labels.device),
                            torch.full((x.shape[1],), -100).to(labels.device),
                            labels[indice, dynamic_offset:]
                        ]
                    elif self.pointcloud_pos=='fast':
                        label_components = [
                            labels[indice, :1],
                            projected_patch_labels[indice],
                            pointcloud_projected_patch_labels[indice] if has_pc 
                                else torch.empty(0, dtype=labels.dtype, device=labels.device),
                            fast_projected_patch_labels[indice] if has_fast 
                                else torch.empty(0, dtype=labels.dtype, device=labels.device),
                            labels[indice, 1:dynamic_offset],
                            torch.full((proprio.shape[1],), -100).to(labels.device) if self.load_state and proprio is not None else torch.empty(0, dtype=torch.bool, device=labels.device),
                            torch.full((t.shape[1],), -100).to(labels.device),
                            torch.full((x.shape[1],), -100).to(labels.device),
                            labels[indice, dynamic_offset:]
                        ]
                else:
                    label_components = [
                        labels[indice, :1],
                        pointcloud_projected_patch_labels[indice] if has_pc 
                            else torch.empty(0, dtype=labels.dtype, device=labels.device),
                        projected_patch_labels[indice],
                        labels[indice, 1:]
                    ]
                multimodal_labels.append(torch.cat([c for c in label_components if c is not None], dim=0).unsqueeze(0))

        multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)
        multimodal_attention_mask = torch.cat(multimodal_attention_mask, dim=0) if multimodal_attention_mask else None
        multimodal_labels = torch.cat(multimodal_labels, dim=0) if multimodal_labels else None

        return multimodal_embeddings, multimodal_attention_mask, multimodal_labels, last_true_indices

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        slow_past_key_values: Optional[List[torch.FloatTensor]] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        use_diff: Optional[bool] = None,
        gen_discret_action: Optional[torch.LongTensor] = None,
        ar_infer: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[CausalLMOutputWithPast, Tuple[CausalLMOutputWithPast, torch.Tensor]]:
        """
        Forward pass for the Vision-Language Model (VLM)
        
        Args:
            Various input tensors and configuration parameters
        
        Returns:
            Model output with optional additional information
        """
        self.gen_discret_action = gen_discret_action
        if use_diff is not None:
            self.use_diff = use_diff

        cache_output = self._handle_cache_forward(input_ids, past_key_values, slow_past_key_values, gen_discret_action, ar_infer, x, t, **kwargs)
        if cache_output is not None:
            return cache_output

        if input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        input_embeddings, slow_llm_embeddings, slow_projected_patch_embeddings, fast_projected_patch_embeddings, pointcloud_projected_patch_embeddings = self._prepare_multimodal_inputs(
            input_ids, pixel_values, point_cloud, multimodal_indices,
        )

        if self.llm_middle_layer!=32:
            if self.pointcloud_pos=='fast':
                multimodal_embeddings, multimodal_attention_mask, multimodal_labels, last_true_indices = self._prepare_multimodal_embeddings(
                    slow_llm_embeddings = slow_llm_embeddings, input_ids = input_ids, multimodal_indices = multimodal_indices,
                    attention_mask = attention_mask, labels = labels, slow_projected_patch_embeddings = slow_projected_patch_embeddings,
                    pointcloud_projected_patch_embeddings = None,
                )
            elif self.pointcloud_pos=='slow':
                multimodal_embeddings, multimodal_attention_mask, multimodal_labels, last_true_indices = self._prepare_multimodal_embeddings(
                    slow_llm_embeddings = slow_llm_embeddings, input_ids = input_ids, multimodal_indices = multimodal_indices,
                    attention_mask = attention_mask, labels = labels, slow_projected_patch_embeddings = slow_projected_patch_embeddings,
                    pointcloud_projected_patch_embeddings = pointcloud_projected_patch_embeddings,
                )
        else:
            if self.load_state:
                proprio = self.proprio_embedder(proprio)
            if self.use_diff and not gen_discret_action:
                x = self.x_embedder(x)
                t = self.t_embedder(t).unsqueeze(1) if t is not None else None

            multimodal_embeddings, multimodal_attention_mask, multimodal_labels, last_true_indices = self._prepare_multimodal_embeddings(
                slow_llm_embeddings = slow_llm_embeddings, input_ids = input_ids, multimodal_indices = multimodal_indices,
                proprio = proprio, t = t, x = x, 
                attention_mask = attention_mask, labels = labels, slow_projected_patch_embeddings = slow_projected_patch_embeddings,
                pointcloud_projected_patch_embeddings = pointcloud_projected_patch_embeddings
            )

        fused_embeddings = multimodal_embeddings
        fused_attention_mask = multimodal_attention_mask
        fused_labels = multimodal_labels

        output: CausalLMOutputWithPast = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            past_key_values=None,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            llm_layer_start=0,
            llm_layer_end=self.llm_middle_layer,
            **{k: v for k, v in kwargs.items() if v is not None}
        )

        if self.llm_middle_layer!=32:
            slow_past_key_values = output.past_key_values
            if self.training:
                ##  False!!!!
                # selected_hidden_states = random.choice((output.hidden_states[8], output.hidden_states[16], output.hidden_states[-1]))
                #  True!!!!
                # selected_idx = torch.tensor([0], device='cuda')  # just a initialization number
                # if torch.distributed.get_rank() == 0:
                #     selected_idx.fill_(random.choice([8, 16, -1]))
                # torch.distributed.broadcast(selected_idx, src=0)
                # selected_hidden_states = output.hidden_states[selected_idx.item()]

                selected_hidden_states = output.hidden_states[-1]
            else:
                selected_hidden_states = output.hidden_states[-1]

            if self.load_state:
                proprio = self.proprio_embedder(proprio)
            if self.use_diff and not gen_discret_action:
                x = self.x_embedder(x)
                t = self.t_embedder(t).unsqueeze(1)

            if self.training_mode == 'sync': fast_projected_patch_embeddings = None
            multimodal_embeddings, multimodal_attention_mask, multimodal_labels, last_true_indices = self._prepare_multimodal_embeddings(
                slow_llm_embeddings = selected_hidden_states, input_ids = input_ids, multimodal_indices = multimodal_indices,
                proprio = proprio, t = t, x = x, 
                attention_mask = attention_mask, labels = labels, 
                slow_projected_patch_embeddings = slow_projected_patch_embeddings, fast_projected_patch_embeddings = fast_projected_patch_embeddings,
                pointcloud_projected_patch_embeddings = pointcloud_projected_patch_embeddings
            )

            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

            output: CausalLMOutputWithPast = self.llm_backbone(
                input_ids=None,
                attention_mask=fused_attention_mask,
                past_key_values=None,
                inputs_embeds=fused_embeddings,
                labels=fused_labels,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                llm_layer_start=self.llm_middle_layer,
                llm_layer_end=32,
                **{k: v for k, v in kwargs.items() if v is not None}
            )

        output.slow_past_key_values = slow_past_key_values
        
        if self.use_diff and not gen_discret_action and not ar_infer:
            last_hidden = output.hidden_states[-1]
            last_hidden = self.final_layer(last_hidden)
            
            tag_0, tag_1, tag_2 = self._get_model_tags()
            action_out = []
            for i, indices in enumerate(last_true_indices):
                action_start = int(indices) + 3 - tag_2 if self.load_state else int(indices) + 2 - tag_2
                action_end = action_start + x.shape[1]
                cur_action = last_hidden[i, action_start:action_end, :].unsqueeze(0)
                action_out.append(cur_action)
            
            action_out = torch.cat(action_out, dim=0)
            return output, action_out
        
        return output


    def slow_system_infer_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        gen_discret_action: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[CausalLMOutputWithPast, Tuple[CausalLMOutputWithPast, torch.Tensor]]:
        """
        Forward pass for the Vision-Language Model (VLM)
        
        Args:
            Various input tensors and configuration parameters
        
        Returns:
            Model output with optional additional information
        """

        self.gen_discret_action = gen_discret_action

        if input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        input_embeddings, slow_llm_embeddings, slow_projected_patch_embeddings, fast_projected_patch_embeddings, pointcloud_projected_patch_embeddings = self._prepare_multimodal_inputs(
            input_ids, pixel_values, point_cloud, multimodal_indices,
        )

        if self.pointcloud_pos=='fast':
            multimodal_embeddings, multimodal_attention_mask, multimodal_labels, last_true_indices = self._prepare_multimodal_embeddings(
                slow_llm_embeddings = slow_llm_embeddings, input_ids = input_ids, multimodal_indices = multimodal_indices,
                attention_mask = attention_mask, labels = labels, slow_projected_patch_embeddings = slow_projected_patch_embeddings,
                pointcloud_projected_patch_embeddings = None,
            )
        elif self.pointcloud_pos=='slow':
            multimodal_embeddings, multimodal_attention_mask, multimodal_labels, last_true_indices = self._prepare_multimodal_embeddings(
                slow_llm_embeddings = slow_llm_embeddings, input_ids = input_ids, multimodal_indices = multimodal_indices,
                attention_mask = attention_mask, labels = labels, slow_projected_patch_embeddings = slow_projected_patch_embeddings,
                pointcloud_projected_patch_embeddings = pointcloud_projected_patch_embeddings,
            )

        fused_embeddings = multimodal_embeddings
        fused_attention_mask = multimodal_attention_mask
        fused_labels = multimodal_labels

        output: CausalLMOutputWithPast = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            past_key_values=None,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            llm_layer_start=0,
            llm_layer_end=self.llm_middle_layer,
            **{k: v for k, v in kwargs.items() if v is not None}
        )

        return output.hidden_states[-1]

    def fast_system_infer_forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        slow_latent_embedding: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        slow_past_key_values: Optional[List[torch.FloatTensor]] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        use_diff: Optional[bool] = None,
        gen_discret_action: Optional[torch.LongTensor] = None,
        ar_infer: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[CausalLMOutputWithPast, Tuple[CausalLMOutputWithPast, torch.Tensor]]:
        """
        Forward pass for the Vision-Language Model (VLM)
        
        Args:
            Various input tensors and configuration parameters
        
        Returns:
            Model output with optional additional information
        """
        self.gen_discret_action = gen_discret_action
        if use_diff is not None:
            self.use_diff = use_diff
        
        tag_0, tag_1, tag_2 = self._get_model_tags()

        cache_output = self._handle_cache_forward(input_ids, past_key_values, slow_past_key_values, gen_discret_action, ar_infer, x, t, **kwargs)
        if cache_output is not None:
            return cache_output

        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)


        if isinstance(pixel_values, dict):
            prefixes = [
                ('head_fast_', 'fast'),
                ('right_fast_', 'fast'),
                ('left_fast_', 'fast')
            ]
            fast_patch_embeddings_list = []
            for prefix, speed in prefixes:
                filtered_keys = [k for k in pixel_values.keys() if k.startswith(prefix)]
                if not filtered_keys:
                    continue
                sub_dict = {k[len(prefix):]: pixel_values[k] for k in filtered_keys}
                patch_features = self.vision_backbone(
                    {k: sub_dict[k][multimodal_indices] for k in sub_dict}
                )
                patch_embeddings = self.projector(patch_features)
                fast_patch_embeddings_list.append(patch_embeddings)
            
            fast_projected_patch_embeddings = (
                torch.cat(fast_patch_embeddings_list, dim=1) 
                if len(fast_patch_embeddings_list) > 1 
                else fast_patch_embeddings_list[0] 
                if len(fast_patch_embeddings_list) == 1 
                else None
            )
        else:
            raise ValueError("No dict in pixel values!!!")
        
        point_cloud = point_cloud.to(multimodal_indices.device) 
        pointcloud_projected_patch_embeddings = self.get_pointcloud_embedding(point_cloud)

        selected_hidden_states = slow_latent_embedding
        
        if self.load_state:
            proprio = self.proprio_embedder(proprio)
        if self.use_diff and not gen_discret_action:
            x = self.x_embedder(x)
            t = self.t_embedder(t).unsqueeze(1)

        slow_llm_embeddings = torch.cat([
            selected_hidden_states[:, :1+(slow_latent_embedding.shape[1]-input_ids.shape[1])],
            pointcloud_projected_patch_embeddings,
            fast_projected_patch_embeddings, 
            selected_hidden_states[:, 1+(slow_latent_embedding.shape[1]-input_ids.shape[1]):],
        ], dim=1)

        last_true_indices = []
        multimodal_embeddings= []
        for indice in multimodal_indices:
            if self.use_diff and not self.gen_discret_action:
                last_true_indice = torch.where(input_ids[indice] == tag_0)[tag_1][-1] + (slow_latent_embedding.shape[1]-input_ids.shape[1])
                last_true_indice += fast_projected_patch_embeddings.shape[1]
                if self.load_pointcloud: last_true_indice += pointcloud_projected_patch_embeddings.shape[1]
                last_true_indices.append(last_true_indice)

            embed = torch.cat([
                slow_llm_embeddings[indice, :last_true_indice + 1 - tag_2, :],
                proprio[indice] if self.load_state and proprio is not None else torch.empty(0, slow_llm_embeddings.shape[2], dtype=torch.bool, device=slow_llm_embeddings.device),
                t[indice],
                x[indice],
                slow_llm_embeddings[indice, last_true_indice + 1 - tag_2:, :],
            ], dim=0).unsqueeze(0)
            multimodal_embeddings.append(embed)

        multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)
        multimodal_attention_mask = None
        multimodal_labels = None

        output: CausalLMOutputWithPast = self.llm_backbone(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            llm_layer_start=self.llm_middle_layer,
            llm_layer_end=32,
            **{k: v for k, v in kwargs.items() if v is not None}
        )
        
        last_hidden = output.hidden_states[-1]
        last_hidden = self.final_layer(last_hidden)
        
        action_out = []
        for i, indices in enumerate(last_true_indices):
            action_start = int(indices) + 3 - tag_2 if self.load_state else int(indices) + 2 - tag_2
            action_end = action_start + x.shape[1]
            action_out.append(last_hidden[i, action_start:action_end, :].unsqueeze(0))
        action_out = torch.cat(action_out, dim=0)
        return output, action_out
    
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        slow_past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        gen_discret_action: Optional[bool] = None,
        ar_infer: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({'gen_discret_action': gen_discret_action})
        model_inputs.update({'ar_infer': ar_infer})

        if "x" in kwargs:
            model_inputs.update({'x': kwargs['x']})
        if "proprio" in kwargs:
            model_inputs.update({'proprio': kwargs['proprio']})
        if "t" in kwargs:
            model_inputs.update({'t': kwargs['t']})

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "point_cloud": point_cloud,
                "past_key_values": past_key_values,
                "slow_past_key_values": slow_past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text
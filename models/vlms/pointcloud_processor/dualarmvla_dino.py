import os
from torch import Tensor
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from timm.models.layers import trunc_normal_ as timm_trunc_normal_

from .model_utils.mv_utils import PCViews
from .model_utils.networks import Point_PN_scan
from .model_utils.util_funcs import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)

from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable
from torch.nn.init import trunc_normal_ as torch_trunc_normal_

from .layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock

logger = logging.getLogger("dinov2")

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=NestedTensorBlock,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        torch_trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        torch_trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdapterSuper_noout(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 drop_rate_adapter=0
                        ):
        super(AdapterSuper_noout, self).__init__()
    
        self.embed_dims = embed_dims

        # Follow towards unified
        self.super_reductuion_dim = reduction_dims
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        if identity is None:
            identity = x
        return out

def bilinear_interpolation_3d_to_2d(x, y, pos_embed):

    img_size = 518
    patch_size = 14
    grid_size = 37  
    # 从 [0, img_size-1] 范围映射到 [-1, 1] 范围
    grid_x = (2.0 * x / (img_size-1)) - 1
    grid_y = (2.0 * y / (img_size-1)) - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(2)

    pos_embed_reshaped = pos_embed.permute(0, 2, 1).view(1, -1, int(img_size / patch_size), int(img_size / patch_size)).repeat(grid.shape[0],1,1,1)
    pos_embed_reshaped = pos_embed_reshaped.cuda()
    interpolated_pos_embed = F.grid_sample(pos_embed_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return interpolated_pos_embed.squeeze()

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,init_values=1e-4,adapter_dim=None,drop_rate_adapter=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) #if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) #if init_values else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
     
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.adapter = AdapterSuper_noout(embed_dims=dim, reduction_dims=adapter_dim, drop_rate_adapter=drop_rate_adapter)
        self.out_transform_3d = nn.Sequential(nn.BatchNorm1d(dim), nn.GELU())
        self.out_transform_2d = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(dim), nn.GELU()) for i in range(6)])

    def forward(self, x, center1=None, idx_ptr=None, sorted_cluster_indices=None, cluster=None,grid_shape=None, mask=None, flat_grid_index=None,attn1=None, norm3=None, args=None):
        x = x + self.ls1(self.drop_path(self.attn(self.norm1(x))))

        x_ffn = self.ls2(self.drop_path(self.mlp(self.norm2(x))))
        x = x + x_ffn + args.scale_factor*self.adapter(x_ffn)        
        return x

class Attention1(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mid_dim=12):
        super().__init__()
        self.num_heads = num_heads 
        self.scale = qk_scale or mid_dim ** -0.5
        self.qkv = nn.Linear(dim, mid_dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop) #now is 0, because for grid sample, it's not necessary to drop
        self.proj = nn.Linear(mid_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) 
        self.mid_dim = mid_dim

    def forward(self, x, mask=None):
        B, N, C = x.shape[0]//128, 128, x.shape[-1]  # B, N, C are batch size, number of points, and channel dimension respectively
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.mid_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Reshape and apply the mask
            mask = torch.where(mask, torch.tensor(-100000.0, device=mask.device), torch.tensor(0.0, device=mask.device))
            attn = attn + mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * N, self.mid_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LoRA(nn.Module):
    """Low-Rank Adaptation for the Query (Q), Key (K), and Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        scaling: float = 1.0,
        r: int = 0,
        dim: int = 0
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = dim
        self.scaling = scaling
        self.r = r
        self.merged = False
        self.params_with_lora = {'q': 'linear_a_q', 'k': 'linear_a_q', 'v': 'linear_a_v'}

    def merge_BA(self, param_name: str):
        lora_A = getattr(self, f'linear_a_{param_name}')
        lora_B = getattr(self, f'linear_b_{param_name}')
        
        # 计算合并后的权重
        merged_weight = (lora_B.weight @ lora_A.weight).view(-1, self.dim)
        return merged_weight
    
    def add_lora_data(self):
        """NOT differentiable"""
        for param_name in ['q', 'v']:
            main_param_weight = getattr(self.qkv, 'weight')  # Assuming you want to modify the weight parameter
            lora_weight = self.merge_BA(param_name)  # 获取合并后的权重和偏置
            if param_name == 'q':
                main_param_weight.data[:self.dim] += lora_weight * self.scaling
            elif param_name == 'v':
                main_param_weight.data[-self.dim:] += lora_weight * self.scaling
    

    def forward(self, x) -> torch.Tensor:
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v

        return qkv

# finetune model
class GfvlaEnvDinov2(nn.Module):
    """
    GF-VLA Environment DINOv2 model for point cloud processing.
    Renamed from Lift3dDinov2 to avoid naming conflicts.
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config.transformer
        config = config.transformer
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.patch_embed = Point_PN_scan(k_neighbors=config.patchknn)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.tokens_pos = nn.Parameter(torch.randn(1, 128, self.trans_dim))
        self.img_size = config.img_size
        self.patch_size = config.patch_size

        # pc_views = PCViews()
        # self.get_pos_2d = pc_views.get_pos_2d

        # lora 
        self.r = config.r
        self.dim = None

        # timm_trunc_normal_(self.cls_token, std=.02)
        # timm_trunc_normal_(self.cls_pos, std=.02)
        # self.apply(self._init_weights)

        # self.ckpt_dir = config.ckpt_dir
        # self.base_ckpt_path = os.path.join(config.ckpt_dir, config.base_ckpt_path)
        # self.mae_ckpt_path = os.path.join(config.ckpt_dir, config.mae_ckpt_path)
        # base_ckpt = torch.load(self.base_ckpt_path, weights_only=True)

        # print(base_ckpt)
        # print("\n\n\n\n\n")
        
        # self.pos_embed_2d = base_ckpt['pos_embed'] #[1, 1370, 768]
        # self.pos_embed_2d.requires_grad = False

        # self.use_mae = True   
        # if self.use_mae:
        #     self.apply_lora_from_mae()
        # else:
        #     self.apply_lora_from_base()
        # self.norm = nn.LayerNorm(self.trans_dim)
        self.feature_dim = self.trans_dim
        

    def bilinear_interpolation_3d_to_2d(self, x, y, pos_embed):
        grid_x = (2.0 * x / (self.img_size - 1)) - 1
        grid_y = (2.0 * y / (self.img_size - 1)) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(2)

        pos_embed_reshaped = (
            pos_embed.permute(0, 2, 1)
            .view(
                1,
                -1,
                int(self.img_size / self.patch_size),
                int(self.img_size / self.patch_size),
            )
            .repeat(grid.shape[0], 1, 1, 1)
        )
        pos_embed_reshaped = pos_embed_reshaped.to(x.device)
        interpolated_pos_embed = F.grid_sample(
            pos_embed_reshaped,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return interpolated_pos_embed.squeeze()
    
    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def apply_lora(self,r=3):
        
        self.lora_layers = list(range(len(self.blocks)))
        self.w_a = []
        self.w_b = []
        for i, block in enumerate(self.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attn.qkv
            self.dim = w_qkv_linear.in_features if self.dim is None else self.dim
            w_a_linear_q, w_b_linear_q = self._create_lora_layer(self.dim, r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(self.dim, r)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                dim=self.dim
            )
        self._reset_lora_parameters()

    def apply_lora_again(self,r=3):
        
        self.lora_layers = list(range(len(self.blocks)))
        self.w_a = []
        self.w_b = []
        for i, block in enumerate(self.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attn.qkv.qkv
            self.dim = w_qkv_linear.in_features if self.dim is None else self.dim
            w_a_linear_q, w_b_linear_q = self._create_lora_layer(self.dim, r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(self.dim, r)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                dim=self.dim
            )
        self._reset_lora_parameters()

    def merge_lora(self):
        self.apply_lora(self.r)
        self.load_model_from_ckpt_mae(self.mae_ckpt_path) #TODO
        for i in range(len(self.blocks)):
            self.blocks[i].attn.qkv.add_lora_data()
        self.apply_lora_again(self.r)

    def apply_lora_from_base(self):
        # self.apply_lora(self.r)
        self.load_model_from_ckpt(self.base_ckpt_path)
        self.apply_lora(self.r)

    def apply_lora_from_mae(self):
        self.apply_lora(self.r)
        self.load_model_from_ckpt_mae(self.mae_ckpt_path) #TODO
        for i in range(len(self.blocks)):
            self.blocks[i].attn.qkv.add_lora_data()
        self.apply_lora_again(self.r)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)

            for key in ckpt.keys():
                if key in ['model', 'net', 'network', 'state_dict', 'base_model']:
                    ckpt = ckpt[key]
            ckpt_state_dict = ckpt
            # cls_token, pos_embed, mask_token, patch_embed, norm, 12-blocks
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt_state_dict.items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print(
                    get_missing_parameters_message(incompatible.missing_keys),
                )
            if incompatible.unexpected_keys:
                print(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print('Training from scratch!!!')
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            timm_trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            timm_trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt_mae(self, ckpt_path):
        """
        Args:
            ckpt_path (_type_): clip_lora from mae pretraining
        """
        if not os.path.exists(ckpt_path):
            repo_id = "jiayueru/gfvla_env"
            filename = "dinov2_base_mae.ckpt"
            cache_dir = self.ckpt_dir
            ckpt_path = hf_hub_download(
                repo_id=repo_id, filename=filename, cache_dir=cache_dir
            )      
        base_ckpt = torch.load(ckpt_path)

        # print(base_ckpt)
        # print("\n\n\n\n\n")

        incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            print(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
            )
    
    def forward(self, pts):
        """
        Forward pass for processing point clouds and generating visual tokens.

        Args:
            pts (Tensor): Input tensor of point clouds of shape [B, N, 3] or [N, 3]

        Returns:
            Tensor: The class token, [B, 768].
        """
        tokens, pos = [], []
        if len(pts.shape) == 2:
            pts = pts.unsqueeze(0).float()
        pts = pts[:, :, :3]
        batch_size = pts.shape[0]
        pts_trans = pts.clone().transpose(1, 2).contiguous()
        center, group_input_tokens = self.patch_embed(pts_trans, pts)
        
        group_input_tokens = group_input_tokens.transpose(1, 2)
        # pos_x, pos_y, _ = self.get_pos_2d(center)
        # self.patch_pos_embed_2D = self.pos_embed_2d[:, 1:]

        # interpolated_pos_embed = self.bilinear_interpolation_3d_to_2d(
        #     pos_x, pos_y, self.patch_pos_embed_2D
        # )
        # interpolated_pos_embed = interpolated_pos_embed.reshape(
        #     center.shape[0], -1, center.shape[1], self.trans_dim
        # ) #[B, 6, 128, 768]
        # interpolated_pos_embed= interpolated_pos_embed.mean(dim=1)

        tokens.append(group_input_tokens)
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # tokens.insert(0, cls_tokens)

        tokens_pos = self.tokens_pos.expand(batch_size, -1, -1)
        pos.append(tokens_pos)
        # cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        # pos.insert(0, cls_pos)
        
        
        pos = torch.cat(pos, dim=1)  # [B, 129, 768]
        tokens = torch.cat(tokens, dim=1) #x: [B, 129, 768]
        x = tokens + pos

        x = {'dino': x, 'siglip': x}

        # for _, block in enumerate(self.blocks):
        #     x = block(x)
        # x = self.norm(x) # [B, 129, 768]
        # return x[:, 1:, :]

        return x

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP policy that wraps each transformer block and the entire model.
        """
        # Policy to wrap the entire GfvlaEnvDinov2 model
        model_wrap_policy = partial(_module_wrap_policy, module_classes={DinoVisionTransformer})
        
        # Policy to wrap individual transformer blocks
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, 
            transformer_layer_cls={Block}  # Assuming you're using the standard Block class
        )
        
        return partial(_or_policy, policies=[model_wrap_policy, transformer_block_policy])
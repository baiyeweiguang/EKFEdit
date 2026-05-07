import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention

class SD3RFEditingAttnProcessor:
    """
    SD3 Attention Processor for RF-Solver Editing.
    """

    def __init__(self):
        self.memory = {}

    def clear_memory(self):
        self.memory.clear()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        timestep: int = -1,
        inject: bool = False,
        inverse: bool = False,
        editing_strategy: str | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        
        
        joint_attn_kwargs = kwargs.get("joint_attention_kwargs", {})
        if joint_attn_kwargs is None:
            joint_attn_kwargs = {}
        
        if timestep == -1:
            timestep = joint_attn_kwargs.get("timestep", -1)
        

        if "inject" in joint_attn_kwargs:
            inject = joint_attn_kwargs["inject"]
        if "inverse" in joint_attn_kwargs:
            inverse = joint_attn_kwargs["inverse"]
        if "editing_strategy" in joint_attn_kwargs:
            editing_strategy = joint_attn_kwargs["editing_strategy"]
            
        if editing_strategy is None:
            editing_strategy = "replace_v"

        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # Projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape [bs, heads, sqlen, head_dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Norm
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # KV Cache replace
        feature_key_v = f"{timestep}_v"        
        if inverse:
            if editing_strategy == "replace_v":
                self.memory[feature_key_v] = value.detach().cpu()
        else:
            if inject and editing_strategy == "replace_v":
                if feature_key_v in self.memory:
                    recorded_v = self.memory[feature_key_v].to(value.device, dtype=value.dtype)
                    # 处理CFG
                    recorded_v = recorded_v.repeat(value.shape[0] // recorded_v.shape[0], 1, 1, 1)
                    if recorded_v.shape == value.shape:
                        value = recorded_v

        # text
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
        

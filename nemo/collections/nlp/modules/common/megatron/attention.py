# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    LoraKQVAdapterConfig,
    LoraKVAdapterConfig,
    LoraQAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.fused_softmax import MatchedScaleMaskSoftmax
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.position_embedding import XPOSPositionEmbedding
from nemo.collections.nlp.modules.common.megatron.position_embedding.rotary_position_embedding import (
    apply_rotary_pos_emb,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    _cast_if_autocast_enabled,
    attention_mask_func,
    _pad_to_multiple,
    _split_into_blocks,
    _concatenate_n_blocks,
    _make_3block_relative_position_ids,
    _mask_local_attention_mask,
    _get_local_attention_mask,
    build_rel_pos_long_attention,
    get_global_token_idx,
)

from nemo.collections.nlp.parts import utils_funcs
from nemo.core import adapter_mixins

try:
    from apex.normalization import MixedFusedRMSNorm
    from apex.transformer.enums import AttnMaskType, AttnType
    from apex.transformer.utils import divide as safe_divide

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


try:
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    from flash_attn.flash_attn_triton import flash_attn_func

    HAVE_FLASH_ATTENTION = True

except (ImportError, ModuleNotFoundError):

    HAVE_FLASH_ATTENTION = False

    flash_attn_unpadded_func, flash_attn_func = None, None
    unpad_input, pad_input = None, None

try:
    from ocean.nn.flash_attention import flash_attention

    HAVE_FLASH_ATTENTION_OCEAN = True

except (ImportError, ModuleNotFoundError):

    flash_attention = None
    HAVE_FLASH_ATTENTION_OCEAN = False

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class ParallelAttention(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        num_attention_heads,
        hidden_size,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        use_cpu_initialization=False,
        megatron_amp_O2=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        layer_type=None,
        megatron_legacy=False,
        bias=True,
        headscale=False,
        position_embedding_type='learned_absolute',
        multi_query_attention=False,
        activations_checkpoint_granularity=None,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        normalize_attention_scores=True,
        use_flash_attention=False,
        use_long_attention=None,  # None / local / global / tglobal / global_to_global / global_limited
        local_context=128,
        global_block_size=16,
        use_long_attn_decoding=True,
        global_tokens_at_start=128,
        global_tokens_cache=1024,
        relative_distance_limit=4096,
    ):
        super(ParallelAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.normalize_attention_scores = normalize_attention_scores
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention

        self.megatron_legacy = megatron_legacy
        self.dtype = utils_funcs.dtype_from_precision(precision, megatron_amp_O2)

        self.use_long_attention = use_long_attention
        self.local_context = local_context
        self.global_block_size = global_block_size
        self.global_tokens_at_start = global_tokens_at_start
        if use_long_attention == "local":
            self.global_tokens_at_start = 0
        self.global_tokens_cache = global_tokens_cache
        self.relative_distance_limit = relative_distance_limit

        self.use_long_attn_decoding = use_long_attn_decoding

        self.update_global_during_decoding = True

        self.set_accepted_adapter_types(
            [
                InfusedAdapterConfig._target_,
                LoraKQVAdapterConfig._target_,
                LoraQAdapterConfig._target_,
                LoraKVAdapterConfig._target_,
            ]
        )

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads
        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
            self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() > 1 and not sequence_parallel
        )

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method,
                use_cpu_initialization=use_cpu_initialization,
                params_dtype=self.dtype,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
                use_cpu_initialization=use_cpu_initialization,
                params_dtype=self.dtype,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )

            self.key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method,
                use_cpu_initialization=use_cpu_initialization,
                params_dtype=self.dtype,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )

        if use_long_attention == "tglobal" or use_long_attention == "global_norm":
            self.global_norm = MixedFusedRMSNorm(hidden_size, eps=1e-6)

        self.core_attention = CoreAttention(
            layer_number=self.layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=self.attention_type,
            attn_mask_type=self.attn_mask_type,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            multi_query_attention=multi_query_attention,
            sequence_parallel=sequence_parallel,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=position_embedding_type,
            use_flash_attention=use_flash_attention,
            use_long_attention=use_long_attention,
            local_context=local_context,
            global_block_size=global_block_size,
            global_tokens_at_start=global_tokens_at_start,
            relative_distance_limit=relative_distance_limit,
        )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=self.dtype,
            bias=bias,
            sequence_parallel_enabled=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )

        self.headscale = headscale
        if headscale:
            self.head_scale_tensor = torch.nn.Parameter(
                torch.ones(1, self.num_attention_heads_per_partition, 1, 1), requires_grad=True
            )

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0
        self.inference_current_local_len = 0
        self.inference_current_global_len = 0

        # relative position embedding
        self.layer_type = layer_type

    def _checkpointed_attention_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor=None,
        global_pos_idx=None,
    ):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            if len(inputs) == 7:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = inputs[4]
                relative_position_bias = inputs[5]
                headscale_tensor = inputs[6]
            elif len(inputs) == 8:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = (inputs[4], inputs[5])
                relative_position_bias = inputs[6]
                headscale_tensor = inputs[7]
            else:
                raise ValueError('unexpected number of inputs')
            output_ = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=headscale_tensor,
                global_pos_idx=global_pos_idx,
            )
            return output_

        if rotary_pos_emb is None:
            rot_tuple = (rotary_pos_emb,)
        else:
            rot_tuple = (rotary_pos_emb[0], rotary_pos_emb[1])

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            *rot_tuple,
            relative_position_bias,
            headscale_tensor,
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype, device):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        input_shape = mixed_layer.size()
        if num_splits_first:
            """[s, b, num_splits * np * hn]
            -->(view) [s, b, num_splits, np, hn]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                num_splits,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
        else:
            """[s, b, np * hn * num_splits]
            -->(view) [s, b, np, hn, num_splits]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
                num_splits,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)

        return mixed_layer

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,  # rotary positional embedding
        relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )

            self.inference_current_sequence_len = 0
            self.inference_current_local_len = 0
            self.inference_current_global_len = 0

            self.inference_current_block_len = 0

        # Some consistency check.
        if inference_max_sequence_len and not self.use_long_attn_decoding:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        is_decoding = hidden_states.shape[0] == 1
        use_long_attention = self.use_long_attention

        if is_decoding:
            if not self.use_long_attn_decoding:
                use_long_attention = None
        else:
            if self.use_long_attention == "tglobal" and hidden_states.shape[0] < self.global_block_size:
                use_long_attention = "local"

        if (not is_decoding) and (use_long_attention is not None) and use_long_attention != "local":

            if use_long_attention == "global_start_only":
                global_pos_idx = torch.arange(0, self.global_tokens_at_start, device=hidden_states.device)
            else:
                global_pos_idx = torch.arange(
                    self.global_tokens_at_start,
                    hidden_states.shape[0],
                    self.global_block_size,
                    device=hidden_states.device,
                )
                if self.global_tokens_at_start > 0:
                    global_pos_idx = torch.cat(
                        (torch.arange(0, self.global_tokens_at_start, device=hidden_states.device), global_pos_idx)
                    )

            if use_long_attention == "tglobal":
                global_hidden_states = _pad_to_multiple(hidden_states, self.global_block_size, 0)
                global_hidden_states = global_hidden_states.reshape(
                    -1, self.global_block_size, hidden_states.shape[-2], hidden_states.shape[-1]
                )
                global_hidden_states = global_hidden_states.sum(dim=1)
                global_hidden_states = self.global_norm(global_hidden_states)
                hidden_states = torch.cat((global_hidden_states, hidden_states), dim=0)
            else:
                global_hidden_states = hidden_states[global_pos_idx]
                if use_long_attention == "global_norm":
                    global_hidden_states = self.global_norm(global_hidden_states)
                hidden_states = torch.cat((global_hidden_states, hidden_states), dim=0)

            if global_pos_idx is not None:
                global_pos_idx = global_pos_idx[None, :].expand(hidden_states.shape[1], -1)
        else:
            global_pos_idx = None

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)
            if self.is_adapter_available():
                lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
                if lora_kqv_adapter:
                    lora_mixed_x_layer = lora_kqv_adapter(hidden_states)
                    mixed_x_layer = mixed_x_layer + lora_mixed_x_layer

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            if self.megatron_legacy:
                mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_x_layer, 3, contiguous_split_chunks=True
            )
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)
            if self.is_adapter_available():
                lora_kv_adapter = self.get_adapter_module(AdapterName.LORA_KV_ADAPTER)
                if lora_kv_adapter:
                    lora_mixed_kv_layer = lora_kv_adapter(encoder_output)
                    mixed_kv_layer = mixed_kv_layer + lora_mixed_kv_layer

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            if self.megatron_legacy:
                mixed_kv_layer = self._transpose_last_dim(mixed_kv_layer, 2, True)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_kv_layer, 2, contiguous_split_chunks=True
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            if self.is_adapter_available():
                lora_q_adapter = self.get_adapter_module(AdapterName.LORA_Q_ADAPTER)
                if lora_q_adapter:
                    lora_q_layer = lora_q_adapter(hidden_states)
                    query_layer = query_layer + lora_q_layer
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        if self.is_adapter_available():
            key_infused_adapter = self.get_adapter_module(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_adapter_module(AdapterName.VALUE_INFUSED)
            if key_infused_adapter:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key_layer.shape
                key_layer = key_infused_adapter(key_layer.reshape(kls[0], kls[1], -1)).reshape(kls)
            if value_infused_adapter:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value_layer.shape
                value_layer = value_infused_adapter(value_layer.reshape(vls[0], vls[1], -1)).reshape(vls)

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else ((rotary_pos_emb,) * 2)

        if inference_max_sequence_len:
            seq_len = key_layer.size(0)

            if self.use_long_attn_decoding and use_long_attention is not None:
                if not set_inference_key_value_memory:
                    ### add or re-write local

                    if self.inference_current_local_len < self.local_context:
                        # if there's space left then just add to local past

                        start_local = -self.local_context + self.inference_current_local_len
                        self.inference_current_local_len += seq_len
                        end_local = -self.local_context + self.inference_current_local_len
                        if end_local == 0:
                            end_local = None

                        self.inference_key_memory[start_local:end_local, ...] = key_layer[-seq_len:]
                        self.inference_value_memory[start_local:end_local, ...] = value_layer[-seq_len:]
                    else:

                        self.inference_key_memory[-self.local_context : -seq_len] = self.inference_key_memory[
                            -self.local_context + seq_len :
                        ].clone()
                        self.inference_value_memory[-self.local_context : -seq_len] = self.inference_value_memory[
                            -self.local_context + seq_len :
                        ].clone()

                        self.inference_key_memory[-seq_len:] = key_layer
                        self.inference_value_memory[-seq_len:] = value_layer

                    self.inference_current_block_len += 1

                    if (
                        self.inference_current_block_len >= self.global_block_size
                        and self.update_global_during_decoding
                        and use_long_attention != "local"
                        and use_long_attention != "global_start_only"
                    ):
                        if self.inference_current_global_len < self.global_tokens_cache:
                            self.inference_key_memory[self.inference_current_global_len] = key_layer[-1]
                            self.inference_value_memory[self.inference_current_global_len] = value_layer[-1]
                            self.inference_current_global_len += 1
                        else:
                            self.inference_key_memory[
                                self.global_tokens_at_start : -self.local_context - 1
                            ] = self.inference_key_memory[self.global_tokens_at_start + 1 : self.local_context].clone()
                            self.inference_key_memory[-self.local_context - 1] = key_layer[-1]
                            self.inference_value_memory[
                                self.global_tokens_at_start : -self.local_context - 1
                            ] = self.inference_value_memory[
                                self.global_tokens_at_start + 1 : self.local_context
                            ].clone()
                            self.inference_value_memory[-self.local_context - 1] = value_layer[-1]

                    self.inference_current_block_len %= self.global_block_size

                    last_local = -self.local_context + self.inference_current_local_len
                    if last_local == 0:
                        last_local = None

                    key_layer = torch.cat(
                        (
                            self.inference_key_memory[: self.inference_current_global_len],
                            self.inference_key_memory[-self.local_context : last_local],
                        ),
                        dim=0,
                    )
                    value_layer = torch.cat(
                        (
                            self.inference_value_memory[: self.inference_current_global_len],
                            self.inference_value_memory[-self.local_context : last_local],
                        ),
                        dim=0,
                    )

                    if relative_position_bias is not None:
                        # TODO: issue if starting with less tokens that start globals...
                        if self.global_tokens_at_start > 0:
                            relative_position_bias_start = relative_position_bias[
                                :, :, :, : self.global_tokens_at_start
                            ]
                            relative_position_bias = torch.cat(
                                (
                                    relative_position_bias_start,
                                    relative_position_bias[
                                        :,
                                        :,
                                        :,
                                        -self.local_context
                                        - self.inference_current_global_len
                                        + self.global_tokens_at_start : -self.local_context,
                                    ],
                                    relative_position_bias[:, :, :, -self.inference_current_local_len :],
                                ),
                                dim=-1,
                            )
                        else:
                            relative_position_bias = torch.cat(
                                (
                                    relative_position_bias[
                                        :,
                                        :,
                                        :,
                                        -self.local_context - self.inference_current_global_len : -self.local_context,
                                    ],
                                    relative_position_bias[:, :, :, -self.inference_current_local_len :],
                                ),
                                dim=-1,
                            )

                    if rotary_pos_emb is not None:
                        q_pos_emb, k_pos_emb = rotary_pos_emb
                        q_pos_emb = q_pos_emb[self.inference_current_local_len - 1 : self.inference_current_local_len]

                        k_pos_emb_new = k_pos_emb[: self.inference_current_local_len, :, :, :]

                        if self.inference_current_global_len > 0:
                            k_start = -k_pos_emb[
                                self.relative_distance_limit
                                - self.inference_current_local_len : self.relative_distance_limit
                                - self.inference_current_local_len
                                + 1
                            ]
                            k_start = k_start.repeat(self.global_tokens_at_start, 1, 1, 1)

                            if use_long_attention == "global":

                                non_start_globals = self.inference_current_global_len - self.global_tokens_at_start

                                pos_idx = torch.arange(
                                    non_start_globals * self.global_block_size,
                                    0,
                                    -self.global_block_size,
                                    device=hidden_states.device,
                                )

                                pos_idx[pos_idx > self.relative_distance_limit] = self.relative_distance_limit

                                k_global = -k_pos_emb[pos_idx]

                                k_start = torch.cat((k_start, k_global), dim=0)

                            k_pos_emb_new = torch.cat((k_start, k_pos_emb_new), dim=0)

                        rotary_pos_emb = (q_pos_emb, k_pos_emb_new)

                else:
                    ### store global and local
                    if global_pos_idx is None:
                        global_tokens = 0
                    else:
                        global_tokens = global_pos_idx.shape[1]

                    global_tokens_cache = self.global_tokens_cache

                    self.inference_key_memory[: min(global_tokens, global_tokens_cache)] = key_layer[
                        max(0, global_tokens - global_tokens_cache) : global_tokens
                    ]
                    self.inference_value_memory[: min(global_tokens, global_tokens_cache)] = value_layer[
                        max(0, global_tokens - global_tokens_cache) : global_tokens
                    ]

                    self.inference_current_global_len = global_tokens

                    local_tokens = min(self.local_context, key_layer.shape[0] - global_tokens)

                    last_local = -self.local_context + local_tokens
                    if last_local == 0:
                        last_local = None

                    self.inference_key_memory[-self.local_context : last_local] = key_layer[-local_tokens:]
                    self.inference_value_memory[-self.local_context : last_local] = value_layer[-local_tokens:]

                    self.inference_current_local_len = local_tokens

                    self.inference_current_block_len = (key_layer.shape[0] - global_tokens) % self.global_block_size
            else:

                if use_long_attention is not None and use_long_attention != "local":
                    seq_len -= global_pos_idx.shape[1]

                # Adjust the range variables.
                start = self.inference_current_sequence_len
                self.inference_current_sequence_len += seq_len
                end = self.inference_current_sequence_len
                # Copy key and values.

                self.inference_key_memory[start:end, ...] = key_layer[-seq_len:]
                self.inference_value_memory[start:end, ...] = value_layer[-seq_len:]

                if not set_inference_key_value_memory:
                    key_layer = self.inference_key_memory[:end, ...]
                    value_layer = self.inference_value_memory[:end, ...]
                # Adjust attention mask
                if attention_mask is not None:
                    attention_mask = attention_mask[..., start:end, :end]
                # adjust the key rotary positional embedding
                if rotary_pos_emb is not None:
                    q_pos_emb, k_pos_emb = rotary_pos_emb
                    if not set_inference_key_value_memory:
                        # In inference, we compute one token at a time.
                        # Select the correct positional embedding.
                        q_pos_emb = q_pos_emb[end - 1 : end]
                    else:
                        q_pos_emb = q_pos_emb[:end, :, :, :]
                    k_pos_emb = k_pos_emb[:end, :, :, :]
                    rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # if self.layer_number == 1:
        #    print(self.inference_current_sequence_len, self.inference_current_local_len,
        #          self.inference_current_global_len)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)

        if get_key_value:
            present = (key_layer, value_layer)

        if checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
                global_pos_idx=global_pos_idx,
            )
        else:
            context_layer = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
                global_pos_idx=global_pos_idx,
            )

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


class ParallelChunkedCrossAttention(MegatronModule):
    """Parallel chunked cross-attention layer class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        num_attention_heads,
        hidden_size,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        use_cpu_initialization=False,
        megatron_amp_O2=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        megatron_legacy=False,
        chunk_size=64,  # each chunk, how many tokens
        bias=True,
        headscale=False,
        gradient_accumulation_fusion=False,
        normalize_attention_scores=True,
    ):
        super(ParallelChunkedCrossAttention, self).__init__()
        self.cross_attention = ParallelAttention(
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=AttnType.cross_attn,
            attn_mask_type=AttnMaskType.padding,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            use_cpu_initialization=use_cpu_initialization,
            megatron_amp_O2=megatron_amp_O2,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            megatron_legacy=megatron_legacy,
            bias=bias,
            headscale=headscale,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            normalize_attention_scores=normalize_attention_scores,
        )
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,
        checkpoint_core_attention=False,
    ):
        if checkpoint_core_attention:
            raise ValueError(
                'checkpoint_core_attention during forward not implemented yet for ParallelChunkedCrossAttention'
            )

        # hidden_states is assumed to have dimension [token length, batch, dimension]
        # derive variables
        # encoder_output here is the retrieved context
        context = encoder_output
        # context is assumed to have dimension [num_chunks, num_neighbors, context_token_len, batch, dimension]
        chunk_size = self.chunk_size
        b, n, dim = (
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.shape[2],
        )
        default_bias = self.cross_attention.dense.bias
        if set_inference_key_value_memory:
            seq_index = (n // chunk_size) * chunk_size
            self.current_len = n
        elif inference_max_sequence_len is not None:
            # only handles single token increment
            assert n == 1
            self.current_len += n
            chunk_id = self.current_len // chunk_size
            if chunk_id <= 0:
                # if sequence length less than chunk size, do an early return
                return torch.zeros_like(hidden_states), default_bias
            causal_padding = chunk_size - 1
            # pad it as a full chunk, put it at the end of the chunk position
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, causal_padding, 0), value=0.0)
            # only use the relevant context
            context = context[chunk_id - 1 : chunk_id, :, :, :, :]
            attention_mask = rearrange(attention_mask, '(b k) 1 q v -> b k 1 q v', b=b)
            # select the relevant chunk attn mask
            attention_mask = attention_mask[:, chunk_id - 1]
            seq_index = chunk_size
        else:
            # this is normal forward without inference
            seq_index = (n // chunk_size) * chunk_size

        # if sequence length less than chunk size, do an early return
        if n < self.chunk_size and set_inference_key_value_memory and inference_max_sequence_len is not None:
            return torch.zeros_like(hidden_states), default_bias

        num_chunks, num_retrieved = (
            context.shape[-5],
            context.shape[-4],
        )

        # causal padding
        causal_padding = chunk_size - 1

        x = F.pad(hidden_states, (0, 0, 0, 0, -causal_padding, causal_padding), value=0.0)

        # remove sequence which is ahead of the neighbors retrieved (during inference)

        # seq_index = (n // chunk_size) * chunk_size
        x, x_remainder = x[:seq_index], x[seq_index:]

        seq_remain_len = x_remainder.shape[0]

        # take care of rotary positional embedding
        # make sure queries positions are properly shifted to the future

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # currently implementation is broken
            # q need to extend to causal_padding, and just do
            # q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, 0), value = 0.)
            if inference_max_sequence_len is not None and not set_inference_key_value_memory:
                token_pos = (self.current_len - 1) % chunk_size
                q_pos_emb = F.pad(
                    q_pos_emb, (0, 0, 0, 0, 0, 0, -causal_padding - token_pos, -causal_padding + token_pos), value=0.0
                )
            else:
                q_pos_emb = F.pad(q_pos_emb, (0, 0, 0, 0, 0, 0, -causal_padding, 0), value=0.0)

            k_pos_emb = repeat(k_pos_emb, 'n b h d -> (r n) b h d', r=num_retrieved)
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # make sure number context chunks is enough
        assert x.shape[0] // chunk_size == num_chunks

        # reshape so we have chunk to chunk attention, without breaking causality
        x = rearrange(x, '(k n) b d -> n (b k) d', k=num_chunks)
        context = rearrange(context, 'k r n b d -> (r n) (b k) d')
        # cross attention
        out, bias = self.cross_attention(x, attention_mask, encoder_output=context, rotary_pos_emb=rotary_pos_emb)

        # reshape back to original sequence

        out = rearrange(out, 'n (b k) d -> (k n) b d', b=b)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)

        out = F.pad(out, (0, 0, 0, 0, causal_padding, -causal_padding + seq_remain_len), value=0.0)
        if not set_inference_key_value_memory and inference_max_sequence_len is not None:
            out = out[-1:]
        return out, bias


class CoreAttention(MegatronModule):
    """ Region where selective activation recomputation is applied.
        See Figure 3. in Reducing Activation Recomputation in Large Transformer Models
        https://arxiv.org/pdf/2205.05198.pdf for more details.

    """

    def __init__(
        self,
        layer_number,
        num_attention_heads,
        hidden_size,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        sequence_parallel=False,
        normalize_attention_scores=True,
        multi_query_attention=False,
        position_embedding_type='learned_absolute',
        use_flash_attention=False,
        use_long_attention=None,  # None / local / global / tglobal / global_to_global / global_sep?
        local_context=128,
        global_block_size=16,
        global_tokens_at_start=128,
        relative_distance_limit=4096,
    ):

        super(CoreAttention, self).__init__()

        self.precision = precision
        self.fp16 = False
        self.bf16 = False
        if precision == 'bf16':
            self.bf16 = True
        elif int(precision) == 16:
            self.fp16 = True
        self.multi_query_attention = multi_query_attention
        self.position_embedding_type = position_embedding_type

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = False
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = sequence_parallel
        # If True, will scale attention scores by 1 / sqrt(hidden_size_per_attention_head).
        # This arg is been provided mostly to support weight conversion of Huggingface models. (ex: T5v1.1)
        self.normalize_attention_scores = normalize_attention_scores

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = safe_divide(projection_size, world_size)
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
            self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        scale_mask_softmax_causal = self.attn_mask_type
        if use_long_attention:
            scale_mask_softmax_causal = AttnMaskType.padding

        self.scale_mask_softmax = MatchedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            scale_mask_softmax_causal,
            masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout_p = attention_dropout
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        if use_flash_attention:
            self.attn_fn = self.flash_attention
        else:
            self.attn_fn = self.torch_attention

        if position_embedding_type.lower() == 'xpos':
            self.xpos = XPOSPositionEmbedding(kv_channels)

        self.use_flash_attention = use_flash_attention

        self.use_long_attention = use_long_attention
        self.local_context = local_context
        self.global_block_size = global_block_size

        self.global_tokens_at_start = global_tokens_at_start
        self.relative_distance_limit = relative_distance_limit

        self.local_context_shift = self.position_embedding_type == "rope"

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor=None,
        global_pos_idx=None,
    ):
        b, np, sq, sk, hn = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
            query_layer.size(3),
        )

        use_long_attention = self.use_long_attention if ((self.use_long_attention is not None) and (sq > 1)) else None
        causal = self.attn_mask_type == AttnMaskType.causal
        global_attn_separate = use_long_attention == "global_to_global"

        if use_long_attention and attention_mask is not None and len(attention_mask.shape) == 4:
            attention_mask = ~torch.any(torch.eq(attention_mask, False), dim=3).squeeze(1)

        if sq == 1:
            attention_mask = None

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[..., sq - 1, :sk].unsqueeze(2)
                else:
                    attention_mask = attention_mask[..., :sq, :sk]

        # ==================================================
        # Update attention bias. [b, np, sq, sk]
        # ==================================================
        relative_position_bias_global = None
        if relative_position_bias is not None:
            if isinstance(relative_position_bias, tuple):
                relative_position_bias, relative_position_bias_global = relative_position_bias

            relative_position_bias = relative_position_bias[
                :,
                self.num_attention_heads_partition_offset : self.num_attention_heads_partition_offset
                + self.num_attention_heads_per_partition,
            ]

            if not use_long_attention:
                relative_position_bias = relative_position_bias[
                    :, :, -sq:, -sk:,
                ]

        # ==================================================
        # Update query_layer, key_layer, value_layer
        # ==================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if use_long_attention is not None and global_pos_idx is not None:
                total_global_tokens = global_pos_idx.shape[1]

                query_layer = torch.cat(
                    [query_layer[:total_global_tokens],
                     apply_rotary_pos_emb(query_layer[total_global_tokens:], q_pos_emb)], dim=0)

                key_layer = torch.cat(
                    [key_layer[:total_global_tokens],
                     apply_rotary_pos_emb(key_layer[total_global_tokens:], k_pos_emb)], dim=0)

                # TODO: this is not correct, need to respect distance limit
                if use_long_attention == "global":
                    non_start_globals = total_global_tokens - self.global_tokens_at_start

                    
                    k_pos_global = torch.arange(
                        self.global_tokens_at_start,
                        self.global_tokens_at_start + non_start_globals * self.global_block_size,
                        self.global_block_size,
                        device=key_layer.device,
                    )
                    k_pos_global_emb = k_pos_emb[k_pos_global]

                    key_layer[self.global_tokens_at_start : total_global_tokens] = apply_rotary_pos_emb(
                        key_layer[self.global_tokens_at_start : total_global_tokens], k_pos_global_emb
                    )

            else:
                query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
                key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)

        if self.position_embedding_type.lower() == 'xpos':
            query_layer = self.xpos(query_layer, offset=key_layer.shape[-2] - query_layer.shape[-2], downscale=False)
            key_layer = self.xpos(key_layer, offset=0, downscale=True)

        # ==================================================
        # Local-Global Attention
        # ==================================================

        if use_long_attention:
            (
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                relative_position_bias,
                global_qkv,
            ) = self.get_long_attention_qkv(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                relative_position_bias,
                rotary_pos_emb,
                global_pos_idx,
                global_attn_separate,
                causal,
            )

        # ==================================================
        # query_layer [sq, b, np, hn]
        # key_layer   [sk, b, np, hn]
        # value_layer [sk, b, np, hn]
        # attention_mask [b, 1, sq, sk] or [b, s]
        # relative_position_bias [b, np, sq, sk]
        # context_layer [b, np, sq, hn]
        # ==================================================
        #if self.layer_number == 1:
        #    print('query_layer', query_layer.shape)
        #    print('key_layer', key_layer.shape)
        context_layer = self.attn_fn(query_layer, key_layer, value_layer, attention_mask, relative_position_bias)

        if use_long_attention:
            assert sq == sk, 'Long Attention can only be used for self-attention.'

            if causal and self.use_flash_attention:
                # hack
                context_layer = context_layer[:, :, -self.local_context :]

            sq_true = sq
            if global_pos_idx is not None:
                sq_true -= global_pos_idx.shape[1]
            nb = (sq_true + self.local_context - 1) // self.local_context
            context_layer = rearrange(context_layer, '(b nb) np sq hn -> b np (nb sq) hn', nb=nb)
            context_layer = context_layer[:, :, :sq_true]

            if global_attn_separate:
                global_q, global_k, global_v = global_qkv

                if relative_position_bias_global is None:
                    relative_position_bias_global = torch.zeros(
                        1,
                        self.num_attention_heads_per_partition,
                        global_q.shape[0],
                        global_k.shape[0],
                        device=global_q.device,
                    )

                else:
                    relative_position_bias_global = relative_position_bias_global[
                        :,
                        self.num_attention_heads_partition_offset : self.num_attention_heads_partition_offset
                        + self.num_attention_heads_per_partition,
                    ]

                if causal:
                    nd, ns = relative_position_bias_global.shape[-2:]
                    i = torch.arange(nd, device=relative_position_bias_global.device)[:, None]
                    j = torch.arange(ns, device=relative_position_bias_global.device)
                    causal_mask = i >= j - ns + nd
                    relative_position_bias_global[:, :] += torch.where(causal_mask, 0.0, -1e10)

                if relative_position_bias_global.shape[0] < b:
                    relative_position_bias_global = relative_position_bias_global.expand(
                        b // relative_position_bias_global.shape[0],
                        *([-1] * (relative_position_bias_global.dim() - 1)),
                    )

                global_token_context = self.attn_fn(
                    query_layer=global_q,
                    key_layer=global_k,
                    value_layer=global_v,
                    attention_mask=None,
                    attention_bias=relative_position_bias_global,
                )

                context_layer.scatter_(
                    2,
                    global_pos_idx[:, None, :, None].expand(-1, context_layer.shape[1], -1, context_layer.shape[3]),
                    global_token_context,
                )

        if headscale_tensor is not None:
            context_layer = context_layer * headscale_tensor

        if self.use_long_attention is not None and self.local_context_shift:
            # shift back
            context_layer = torch.cat([
                context_layer[:, :self.num_attention_heads_per_partition // 2],
                context_layer[:, self.num_attention_heads_per_partition // 2:].roll(self.local_context // 2, dims=2)
            ], dim=1)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def torch_attention(self, query_layer, key_layer, value_layer, attention_mask, attention_bias):
        sq, b, np, hn = query_layer.shape
        sk = key_layer.shape[0]

        if self.multi_query_attention:
            query_layer = rearrange(query_layer, 'sq b np hn -> b (np sq) hn')
            key_layer = rearrange(key_layer, 'sk b 1 hn -> b hn sk')
            value_layer = rearrange(value_layer, 'sv b np hn -> (b np) sv hn')
        else:
            query_layer = rearrange(query_layer, 'sq b np hn -> (b np) sq hn')
            key_layer = rearrange(key_layer, 'sk b np hn -> (b np) hn sk')
            value_layer = rearrange(value_layer, 'sv b np hn -> (b np) sv hn')

        matmul_input_buffer = torch.empty(
            query_layer.shape[0],
            query_layer.shape[1],
            key_layer.shape[2],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,
            key_layer,
            beta=0.0,
            alpha=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(b, np, sq, sk)

        if attention_bias is not None:
            attention_scores += attention_bias

        is_causal = self.attn_mask_type == AttnMaskType.causal and sq == sk
        if attention_mask is None and is_causal:
            s = max(sq, sk)
            attention_mask = attention_scores.new_ones(b, 1, s, s, dtype=torch.bool)
            attention_mask = attention_mask.tril()
            attention_mask = ~attention_mask
            attention_mask = attention_mask[-sq:, -sk:]

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        if attention_mask is not None:
            all_k_masked = attention_mask.all(axis=-1)
            zero_attention_mask = (1.0 - all_k_masked.type(attention_probs.type()))[:, :, :, None]
            attention_probs = attention_probs * zero_attention_mask

        if not self.sequence_parallel:
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # change view [b * np, sq, sk]
        attention_probs = rearrange(attention_probs, 'b np sq sk -> (b np) sq sk')

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = rearrange(context_layer, '(b np) sq hn -> b np sq hn', np=np)

        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer, attention_mask, attention_bias):
        # Use to ensure dtype cast to fp16 or bf16
        query_layer = _cast_if_autocast_enabled(query_layer)
        key_layer = _cast_if_autocast_enabled(key_layer)
        value_layer = _cast_if_autocast_enabled(value_layer)
        attention_bias = _cast_if_autocast_enabled(attention_bias)

        if attention_bias is not None:
            if HAVE_FLASH_ATTENTION_OCEAN:
                return self.flash_attention_triton_ocean(
                    query_layer, key_layer, value_layer, attention_mask, attention_bias,
                )
            else:
                return self.flash_attention_triton(
                    query_layer, key_layer, value_layer, attention_mask, attention_bias,
                )
        else:
            return self.flash_attention_cuda(query_layer, key_layer, value_layer, attention_mask,)

    def reset_is_causal(self, query_length, key_length, causal):
        if query_length != key_length:
            if query_length == 1:
                return False
            raise NotImplementedError(
                "Flash attention does not support query and key with different number of tokens, unless number of query tokens is 1."
            )
        return causal

    def flash_attention_cuda(self, query_layer, key_layer, value_layer, attention_mask):
        query_layer = rearrange(query_layer, 'sq b np hn -> b sq np hn')
        key_layer = rearrange(key_layer, 'sk b np hn -> b sk np hn')
        value_layer = rearrange(value_layer, 'sv b np hn -> b sv np hn')

        batch_size, seqlen, nheads, _ = query_layer.shape

        # True: attend / False: not attend
        if attention_mask is None:
            attention_mask_q = torch.ones(batch_size, query_layer.shape[1], device=query_layer.device).bool()
            attention_mask_kv = torch.ones(batch_size, key_layer.shape[1], device=query_layer.device).bool()
        elif len(attention_mask.shape) == 4:
            # [b, 1, sq, sk] -> [b, sq] / [b, sk]
            attention_mask_q = torch.any(torch.eq(attention_mask, False), dim=3).squeeze(1)
            attention_mask_kv = torch.any(torch.eq(attention_mask, False), dim=2).squeeze(1)
        else:
            assert len(attention_mask.shape) == 2
            attention_mask_q = attention_mask
            attention_mask_kv = attention_mask

        q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_layer, attention_mask_q)
        k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_layer, attention_mask_kv)
        v, _, _, _ = unpad_input(value_layer, attention_mask_kv)
        causal = self.reset_is_causal(
            query_layer.shape[1], key_layer.shape[1], self.attn_mask_type == AttnMaskType.causal
        )
        context_layer = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            causal=causal,
        )

        # [b, sq, np, hn]
        context_layer = pad_input(context_layer, indices_q, batch_size, seqlen)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)
        return context_layer

    def flash_attention_triton(self, query_layer, key_layer, value_layer, attention_mask, attention_bias):
        if self.attention_dropout_p > 0.0:
            raise NotImplementedError(f'attention_dropout not implemented for flash_attention with attention bias')

        query_layer = rearrange(query_layer, 'sq b np hn -> b sq np hn')
        key_layer = rearrange(key_layer, 'sk b np hn -> b sk np hn')
        value_layer = rearrange(value_layer, 'sv b np hn -> b sv np hn')

        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                # [b, 1, sq, sk] -> [b, 1, sq, 1] / [b, 1, 1, sk]
                attention_mask_q = torch.any(torch.eq(attention_mask, False), dim=3).unsqueeze(3)
                attention_mask_kv = torch.any(torch.eq(attention_mask, False), dim=2).unsqueeze(2)
            else:
                # [b, s] -> [b, 1, s, 1] / [b, 1, 1, s]
                assert len(attention_mask.shape) == 2
                attention_mask_q = ~attention_mask.unsqueeze(1).unsqueeze(3)
                attention_mask_kv = ~attention_mask.unsqueeze(1).unsqueeze(2)

            if attention_bias.shape[2] == attention_mask_q.shape[2]:
                attention_bias = attention_bias.masked_fill(~attention_mask_q, torch.finfo(query_layer.dtype).min)
            if attention_bias.shape[3] == attention_mask_kv.shape[3]:
                attention_bias = attention_bias.masked_fill(~attention_mask_kv, torch.finfo(query_layer.dtype).min)

        is_causal = self.attn_mask_type == AttnMaskType.causal and query_layer.shape[1] == key_layer.shape[1]
        context_layer = flash_attn_func(query_layer, key_layer, value_layer, attention_bias, is_causal,)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)

        if attention_mask is not None:
            context_layer = context_layer * attention_mask_q

        return context_layer

    def flash_attention_triton_ocean(self, query_layer, key_layer, value_layer, attention_mask, attention_bias):
        q_len, kv_len = None, None
        if attention_mask is not None:
            """
            attention_mask: 0 is not masked; 1 is masked
            """
            if len(attention_mask.shape) == 4:
                # [b, 1, sq, sk]
                q_len = torch.any(torch.eq(attention_mask, False), dim=3).sum(-1).squeeze()
                kv_len = torch.any(torch.eq(attention_mask, False), dim=2).sum(-1).squeeze()
            else:
                # [b, s]
                assert len(attention_mask.shape) == 2
                q_len = (~attention_mask).sum(-1).squeeze()
                kv_len = (~attention_mask).sum(-1).squeeze()

        is_causal = self.attn_mask_type == AttnMaskType.causal

        if self.position_embedding_type == 'alibi':
            if is_causal:
                bias_type = 'vector'
                assert len(attention_bias.size()) == 4
            else:
                bias_type = 'alibi'
                attention_bias = attention_bias.squeeze()
                assert len(attention_bias.size()) == 1, 'Alibi bias should only contain head scales.'
        else:
            bias_type = 'vector'
            assert len(attention_bias.size()) == 4

        context_layer = flash_attention(
            q=query_layer,
            k=key_layer,
            v=value_layer,
            q_len=q_len,
            kv_len=kv_len,
            softmax_scale=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
            causal=is_causal,
            bias=attention_bias,
            bias_type=bias_type,
            dropout=self.attention_dropout_p if self.training else 0.0,
            seed=torch.seed(),
            layout='sbnd',
        )

        # [sq, b, np, hn] -> [b, np, sq, hn]
        context_layer = context_layer.permute(1, 2, 0, 3)

        return context_layer

    def get_long_attention_qkv(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        relative_position_bias,
        rotary_pos_emb,
        global_pos_idx,
        global_attn_separate,
        causal,
    ):

        # ==================================================
        # Update query_layer, key_layer, value_layer
        # ==================================================
        if global_pos_idx is not None:
            if self.use_long_attention is not None:

                total_global_tokens = global_pos_idx.shape[1]

                side_k = key_layer[:total_global_tokens]
                side_v = value_layer[:total_global_tokens]

                side_k = side_k.permute(1, 2, 0, 3)
                side_v = side_v.permute(1, 2, 0, 3)

                query_layer = query_layer[total_global_tokens:]
                key_layer = key_layer[total_global_tokens:]
                value_layer = value_layer[total_global_tokens:]

        if self.local_context_shift:
            query_layer = torch.cat([
                query_layer[:, :, :self.num_attention_heads_per_partition // 2],
                query_layer[:, :, self.num_attention_heads_per_partition // 2:].roll(-self.local_context // 2, dims=0)
            ], dim=2)
            key_layer = torch.cat([
                key_layer[:, :, :self.num_attention_heads_per_partition // 2],
                key_layer[:, :, self.num_attention_heads_per_partition // 2:].roll(-self.local_context // 2, dims=0)
            ], dim=2)
            value_layer = torch.cat([
                value_layer[:, :, :self.num_attention_heads_per_partition // 2],
                value_layer[:, :, self.num_attention_heads_per_partition // 2:].roll(-self.local_context // 2, dims=0)
            ], dim=2)

        # [sq/k, b, np, hn] -> [b, hn, sq/k, np]
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)

        sq0 = query_layer.shape[2]

        # Split into blocks -> (batch_size, n_heads, num_blocks, block_len, dim_per_head)
        query_layer = _split_into_blocks(query_layer, self.local_context, dim=2)
        key_layer = _split_into_blocks(key_layer, self.local_context, dim=2)
        value_layer = _split_into_blocks(value_layer, self.local_context, dim=2)

        # don't need this with shift
        if not self.local_context_shift:
            # Concatenate 3 blocks for keys and values -> (batch_size, n_heads, num_blocks, 3 * block_len, dim_per_head)
            key_layer = _concatenate_n_blocks(key_layer, block_dim=2, sequence_dim=3, concat_num=3 - causal)
            value_layer = _concatenate_n_blocks(value_layer, block_dim=2, sequence_dim=3, concat_num=3 - causal)

        if global_pos_idx is not None:
            # Tile side inputs across local key/value blocks
            # New shape: (batch_size, n_heads, num_blocks, global_seq_len, dim_per_head)
            reps = [1] * (side_k.ndim + 1)
            reps[2] = key_layer.shape[2]
            side_k = side_k.unsqueeze(2).repeat(reps)
            side_v = side_v.unsqueeze(2).repeat(reps)

            if rotary_pos_emb is not None:
                _, k_pos_emb = rotary_pos_emb

                rot_idx = torch.arange(
                    -self.relative_distance_limit,
                    -self.relative_distance_limit + self.local_context * key_layer.shape[2],
                    self.local_context,
                    device=k_pos_emb.device,
                )
                negative = rot_idx < 0

                rots = k_pos_emb[rot_idx.abs().long()].unsqueeze(0).unsqueeze(0).squeeze(3)
                rots = rots.repeat(1, 1, 1, self.global_tokens_at_start, 1)
                rots[0, 0, negative, :, :] = k_pos_emb[: self.global_tokens_at_start].squeeze()

                side_k[:, :, :, : self.global_tokens_at_start] = apply_rotary_pos_emb(
                    side_k[:, :, :, : self.global_tokens_at_start], rots
                )

                if self.use_long_attention == "global":
                    non_start_global_idx = global_pos_idx[:, self.global_tokens_at_start:]
                    non_start_global_idx = non_start_global_idx.unsqueeze(2).repeat(reps)

                    local_block_starts = torch.arange(0, key_layer.shape[2] * self.local_context, self.local_context, device=global_pos_idx.device)
                    local_block_starts -= self.relative_distance_limit
                    local_block_starts = local_block_starts[None, None, :, None, None]

                    non_start_global_idx[non_start_global_idx <= local_block_starts] = local_block_starts[:, :, (non_start_global_idx <= local_block_starts).nonzero()[:, 2]].squeeze()

                    rots = k_pos_emb[non_start_global_idx].squeeze() # blocks x global_tokens x dim
                    side_k[:, :, :, self.global_tokens_at_start:self.global_tokens_at_start+rots.shape[1]] = apply_rotary_pos_emb(
                        side_k[:, :, :, self.global_tokens_at_start:self.global_tokens_at_start+rots.shape[1]], rots
                    )


            # Concatenate "local" and "side"/"global" key/value states
            # New shape: (batch_size, n_heads, num_blocks, 3 * block_len + global_seq_len, dim_per_head)
            key_layer = torch.cat([side_k, key_layer], dim=3)
            value_layer = torch.cat([side_v, value_layer], dim=3)

        # ==================================================
        # Update attention_mask, attention_bias
        # ==================================================
        # [TODO] unified attention_mask dimension
        # attention_mask: [b, 1, sk]
        bs, np, nb, sk = key_layer.shape[:4]

        if relative_position_bias is None:
            _, _, attention_mask = build_rel_pos_long_attention(
                seq_length=sq0,
                use_long_attention=self.use_long_attention,
                local_context=self.local_context,
                global_tokens_at_start=self.global_tokens_at_start,
                relative_distance_limit=self.relative_distance_limit,
                global_block_size=self.global_block_size,
                global_tokens_cache=0,
                global_segment_pos_idx=False,
                causal=causal,
                get_mask=True,
                vector_only=self.use_flash_attention,
                shift_mode=self.local_context_shift,
            )
            relative_position_bias = torch.where(attention_mask, -1e10, 0).unsqueeze(0).unsqueeze(0)
        else:
            relative_position_bias = relative_position_bias[..., :nb, :, :sk]

        query_layer = rearrange(query_layer, "bs np nb sq hn -> sq (bs nb) np hn")
        key_layer = rearrange(key_layer, "bs np nb sk hn -> sk (bs nb) np hn")
        value_layer = rearrange(value_layer, "bs np nb sv hn -> sv (bs nb) np hn")
        attention_mask = None
        if relative_position_bias is not None:
            if relative_position_bias.shape[0] == 1:
                relative_position_bias = relative_position_bias.repeat(bs, 1, 1, 1, 1)
            relative_position_bias = rearrange(relative_position_bias, 'b np nb sq sk -> (b nb) np sq sk')

        if global_attn_separate:
            global_query_layer = rearrange(side_q, "bs np sq hn -> sq bs np hn")
            global_key_layer = rearrange(side_k, "bs np sq hn -> sq bs np hn")
            global_value_layer = rearrange(side_v, "bs np sq hn -> sq bs np hn")

            global_qkv = (global_query_layer, global_key_layer, global_value_layer)
        else:
            global_qkv = None

        return query_layer, key_layer, value_layer, attention_mask, relative_position_bias, global_qkv

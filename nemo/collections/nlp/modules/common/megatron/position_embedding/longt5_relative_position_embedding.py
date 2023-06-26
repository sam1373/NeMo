# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


class LongT5RelativePositionEmbedding(torch.nn.Module):
    """Global/Local Position Embedding implementation from the LongT5 paper : https://arxiv.org/abs/2112.07916"""

    def __init__(
        self,
        init_method,
        bidirectional,
        num_attention_heads,
        layer_type,
        relative_position_num_buckets=32,
        relative_position_max_distance=128,
    ):
        super(LongT5RelativePositionEmbedding, self).__init__()
        self.relative_position_num_buckets = relative_position_num_buckets
        self.relative_position_max_distance = relative_position_max_distance
        self.self_attention_relative_position_bucket = None
        self.inter_attention_relative_position_bucket = None
        self.self_attention_relative_position_bias = None
        self.inter_attention_relative_position_bias = None
        self.bidirectional = bidirectional

        # LayerType.encoder or LayerType.decoder. Is only needed to determine the group for the all_reduce
        self.layer_type = layer_type

        # Relative position Embedding
        # Relative Position embedding (all attention layers).
        self.local_position_embedding = torch.nn.Embedding(self.relative_position_num_buckets, num_attention_heads).to(
            torch.cuda.current_device()
        )
        self._local_position_embedding_key = 'local_position_embedding'
        init_method(self.local_position_embedding.weight)

        self.global_position_embedding = torch.nn.Embedding(
            self.relative_position_num_buckets, num_attention_heads
        ).to(torch.cuda.current_device())
        self._global_position_embedding_key = 'global_position_embedding'
        init_method(self.global_position_embedding.weight)

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from HuggingFace T5 Model:
        https://github.com/huggingface/transformers/blob/b5e2b183af5e40e33a4dc7659e697d137259d56e
        /src/transformers/models/t5/modeling_t5.py#L354
        Translate relative position to a bucket number for relative attention. The relative position
        is defined as memory_position - query_position, i.e. the distance in tokens from the attending
        position to the attended-to position. If bidirectional=False, then positive relative positions
        are invalid. We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions. All relative positions >=max_distance map to the same
        bucket. All relative positions <=-max_distance map to the same bucket. This should allow for
        more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position,
            containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def forward(self, query_seq_length, key_seq_length):
        device = self.local_position_embedding.weight.device
        context_position = torch.arange(query_seq_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_seq_length, dtype=torch.long, device=device)[None, :]

        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket_tensor = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.relative_position_num_buckets,
            max_distance=self.relative_position_max_distance,
        )
        bias = self.global_position_embedding(relative_position_bucket_tensor)
        bias = bias.permute(2, 0, 1).unsqueeze(0)

        return bias

    def get_bias(self, relative_position, global_tokens=0):

        relative_position_buckets = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_position_num_buckets,
            max_distance=self.relative_position_max_distance,
        )

        relative_position_local_buckets = relative_position_buckets[:, :, :-global_tokens]
        relative_position_global_buckets = relative_position_buckets[:, :, -global_tokens:]

        local_bias = self.local_position_embedding(relative_position_local_buckets)
        global_bias = self.global_position_embedding(relative_position_global_buckets)

        bias = torch.cat((local_bias, global_bias), dim=2)
        bias = bias.permute(0, 3, 1, 2)

        return bias

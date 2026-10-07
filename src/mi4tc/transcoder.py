"""Native Transcoder architecture used by the released checkpoints."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class Transcoder(nn.Module):
    """MLP Transcoder. Decoder rows are unit-normalized only when that option is on."""

    def __init__(
        self,
        d_model: int,
        expansion: int,
        device: torch.device,
        dtype: torch.dtype,
        topk: int = 0,
        *,
        encoder_dtype: torch.dtype | None = None,
        decoder_dtype: torch.dtype | None = None,
        topk_input: str = "preactivation",
        normalize_decoder: bool = True,
        decoder_init: str = "kaiming",
        shared_init: bool = False,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_model * expansion
        self.topk = topk
        self.topk_input = topk_input
        self.normalize_decoder_enabled = normalize_decoder
        enc_dtype = encoder_dtype or dtype
        dec_dtype = decoder_dtype or dtype
        generator = None
        if shared_init:
            # One FP32 draw on the parameter device, then cast. A local generator
            # keeps the four diagnostic runs aligned without a CPU-sized copy.
            generator = torch.Generator(device=device)
            generator.manual_seed(init_seed)
            encoder = torch.empty(self.d_feature, d_model, device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(encoder, a=math.sqrt(5), generator=generator)
            self.W_enc = nn.Parameter(encoder.to(dtype=enc_dtype))
            del encoder
        else:
            self.W_enc = nn.Parameter(torch.empty(self.d_feature, d_model, device=device, dtype=enc_dtype))
            nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        self.b_enc = nn.Parameter(torch.zeros(self.d_feature, device=device, dtype=enc_dtype))
        if decoder_init == "zeros":
            self.W_dec = nn.Parameter(torch.zeros(self.d_feature, d_model, device=device, dtype=dec_dtype))
        elif decoder_init == "kaiming":
            if shared_init:
                decoder = torch.empty(self.d_feature, d_model, device=device, dtype=torch.float32)
                nn.init.kaiming_uniform_(decoder, a=math.sqrt(5), generator=generator)
                self.W_dec = nn.Parameter(decoder.to(dtype=dec_dtype))
                del decoder
            else:
                self.W_dec = nn.Parameter(torch.empty(self.d_feature, d_model, device=device, dtype=dec_dtype))
                nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        else:
            raise ValueError(f"Unknown decoder init: {decoder_init}")
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device, dtype=dec_dtype))
        if normalize_decoder:
            self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.W_dec.float().norm(dim=1, keepdim=True).clamp_min_(1e-6)
        self.W_dec.div_(norms.to(dtype=self.W_dec.dtype))

    @torch.no_grad()
    def calibrate_biases_(self, mean_x: torch.Tensor, mean_y: torch.Tensor) -> None:
        mean_x = mean_x.to(device=self.W_enc.device, dtype=torch.float32)
        mean_y = mean_y.to(device=self.b_dec.device, dtype=torch.float32)
        b_enc = -(mean_x @ self.W_enc.detach().float().T)
        self.b_enc.copy_(b_enc.to(dtype=self.b_enc.dtype))
        self.b_dec.copy_(mean_y.to(dtype=self.b_dec.dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # F.linear rejects mixed dtypes. Cast only when a diagnostic matrix
        # uses a wider dtype than the incoming activation.
        encoder_input = x if x.dtype == self.W_enc.dtype else x.to(dtype=self.W_enc.dtype)
        preactivation = F.linear(encoder_input, self.W_enc, self.b_enc)
        if self.topk > 0:
            scores = F.relu(preactivation) if self.topk_input == "relu" else preactivation
            k = min(self.topk, scores.shape[-1])
            values, indices = torch.topk(scores, k, dim=-1)
            features = torch.zeros_like(scores).scatter(-1, indices, values)
        else:
            features = F.relu(preactivation)
        decoder_input = features if features.dtype == self.W_dec.dtype else features.to(dtype=self.W_dec.dtype)
        reconstruction = F.linear(decoder_input, self.W_dec.t(), self.b_dec)
        return reconstruction, features, preactivation

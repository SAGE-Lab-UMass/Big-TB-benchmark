"""Small Evo2 wrapper that isolates model-specific tokenization and pooling."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
import sys
import types
from typing import Literal

import numpy as np
import torch


EmbeddingType = Literal["mean_dim", "mean_seq", "token"]


@dataclass(frozen=True)
class Evo2ModelConfig:
    model_name: str = "evo2_7b"
    local_path: str | None = None
    layer_name: str = "blocks.28.mlp.l3"
    max_length: int = 5000
    pad_char: str = "N"
    use_kernels: bool = False
    save_dtype: str = "float16"


class Evo2Embedder:
    def __init__(self, config: Evo2ModelConfig) -> None:
        from evo2 import Evo2

        self.config = config
        _patch_vortex_rotary_fallback()
        evo2_kwargs = {
            "model_name": config.model_name,
            "local_path": config.local_path,
        }
        if "use_kernels" in inspect.signature(Evo2.__init__).parameters:
            evo2_kwargs["use_kernels"] = config.use_kernels

        self.model = Evo2(**evo2_kwargs)
        self.model.model.eval()
        self.device = _first_parameter_device(self.model.model)
        self.pad_token_id = self.model.tokenizer.tokenize(config.pad_char)[0]
        print(f"Loaded {config.model_name} on input device {self.device}")

    def embed_sequences(self, sequences: list[str], embed_type: EmbeddingType) -> np.ndarray:
        input_ids, attention_mask = self._tokenize_batch(sequences)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        _, embeddings = self.model(
            input_ids,
            return_embeddings=True,
            layer_names=[self.config.layer_name],
        )
        hidden_states = embeddings[self.config.layer_name]
        attention_mask = attention_mask.to(hidden_states.device)

        if embed_type == "token":
            pooled = hidden_states * attention_mask.unsqueeze(-1)
        elif embed_type == "mean_seq":
            pooled = masked_mean_sequence(hidden_states, attention_mask)
        elif embed_type == "mean_dim":
            pooled = masked_mean_hidden_dim(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unsupported embed_type: {embed_type}")

        pooled = pooled.detach().cpu().numpy()
        return pooled.astype(self.config.save_dtype, copy=False)

    def ensure_autograd_compatible(self) -> tuple[int, int]:
        """Replace Vortex checkpoint tensors created under inference mode.

        Vortex currently loads Evo2 checkpoints inside ``torch.inference_mode()``
        and performs dtype/device conversions there.  Those conversions can leave
        module parameters/buffers backed by *inference tensors*.  Such tensors are
        valid for inference but cannot be saved by autograd for backward, even if
        the pretrained parameter itself is frozen.  LoRA training still needs
        autograd to save frozen weights/constants in order to differentiate with
        respect to activations and adapter parameters.

        Clone only inference tensors *outside* inference mode and replace the
        registered Parameter/buffer objects before LoRA injection or optimizer
        construction.  Aliased Parameters/Buffers remain aliased.  Values, dtype,
        device, and ``requires_grad`` are preserved.

        Returns
        -------
        (num_parameters_replaced, num_buffers_replaced)
        """
        inner_model = self.model.model

        is_inference = getattr(torch, "is_inference", None)
        if is_inference is None:
            def is_inference(tensor: torch.Tensor) -> bool:
                checker = getattr(tensor, "is_inference", None)
                return bool(checker()) if callable(checker) else False

        parameter_memo: dict[int, torch.nn.Parameter] = {}
        buffer_memo: dict[int, torch.Tensor] = {}
        num_parameters_replaced = 0
        num_buffers_replaced = 0

        # ``no_grad`` is intentional here; unlike ``inference_mode`` it does not
        # create inference tensors.  Cloning outside inference mode materializes
        # ordinary tensors with normal version counters.
        with torch.no_grad():
            for module in inner_model.modules():
                for name, param in list(module._parameters.items()):
                    if param is None:
                        continue
                    old_id = id(param)
                    if old_id in parameter_memo:
                        module._parameters[name] = parameter_memo[old_id]
                        continue

                    if is_inference(param):
                        replacement = torch.nn.Parameter(
                            param.detach().clone(),
                            requires_grad=param.requires_grad,
                        )
                        parameter_memo[old_id] = replacement
                        module._parameters[name] = replacement
                        num_parameters_replaced += 1
                    else:
                        parameter_memo[old_id] = param

                for name, buffer in list(module._buffers.items()):
                    if buffer is None:
                        continue
                    old_id = id(buffer)
                    if old_id in buffer_memo:
                        module._buffers[name] = buffer_memo[old_id]
                        continue

                    if is_inference(buffer):
                        replacement = buffer.detach().clone()
                        buffer_memo[old_id] = replacement
                        module._buffers[name] = replacement
                        num_buffers_replaced += 1
                    else:
                        buffer_memo[old_id] = buffer

        remaining_parameters = [
            name for name, param in inner_model.named_parameters() if is_inference(param)
        ]
        remaining_buffers = [
            name for name, buffer in inner_model.named_buffers() if is_inference(buffer)
        ]
        if remaining_parameters or remaining_buffers:
            raise RuntimeError(
                "Failed to materialize all Evo2 inference tensors for autograd. "
                f"Remaining parameters={remaining_parameters[:5]}, "
                f"buffers={remaining_buffers[:5]}"
            )

        print(
            "[Evo2] Autograd compatibility: replaced "
            f"{num_parameters_replaced} inference Parameters and "
            f"{num_buffers_replaced} inference buffers with normal tensors"
        )
        return num_parameters_replaced, num_buffers_replaced

    @property
    def hidden_size(self) -> int:
        """Hidden width of the loaded StripedHyena backbone."""
        config = self.model.model.config
        hidden_size = config.get("hidden_size") if isinstance(config, dict) else getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Loaded Evo2/StripedHyena config does not expose hidden_size")
        return int(hidden_size)

    @property
    def num_layers(self) -> int:
        """Number of StripedHyena blocks in the loaded model."""
        return len(self.model.model.blocks)

    def extract_layer_tensor(
        self,
        sequences: list[str],
        *,
        layer_name: str | None = None,
        mask_padding: bool = True,
    ) -> torch.Tensor:
        """Return an intermediate token representation without breaking autograd.

        The public :class:`evo2.Evo2` ``return_embeddings=True`` path is an
        inference API: it executes the Vortex model under ``torch.no_grad()``
        and detaches hook outputs.  That behavior is correct for offline
        embedding generation but cannot be used for supervised LoRA training.

        This method intentionally calls the underlying ``StripedHyena`` module
        directly and captures the requested module output with a forward hook.
        It contains no ``no_grad``/``detach`` operation; the caller controls the
        autograd context.  Padding is masked *after* extraction exactly as in
        :meth:`embed_sequences` so the token tensor supplied to the downstream
        classifier matches the frozen-embedding baseline.

        Notes
        -----
        ``padding_mask`` is deliberately not passed into ``StripedHyena`` here.
        The historical baseline ``Evo2Embedder.embed_sequences`` invokes the
        stock Evo2 wrapper, which calls ``StripedHyena.forward(input_ids)``
        without a padding mask.  Passing a mask inside the backbone would change
        the representation and invalidate the baseline-vs-finetuned comparison.
        """
        if not sequences:
            raise ValueError("sequences must contain at least one DNA sequence")

        target_name = layer_name or self.config.layer_name
        inner_model = self.model.model
        try:
            target_module = inner_model.get_submodule(target_name)
        except AttributeError as exc:
            raise ValueError(f"Evo2 layer {target_name!r} was not found") from exc

        input_ids, attention_mask = self._tokenize_batch(sequences)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        captured: dict[str, torch.Tensor] = {}

        def _capture(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output from {target_name}, got {type(output).__name__}"
                )
            captured["hidden"] = output

        handle = target_module.register_forward_hook(_capture)
        try:
            # IMPORTANT: no torch.no_grad() here.  During LoRA training the
            # classification loss must remain connected to adapter parameters.
            _ = inner_model(input_ids)
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError(f"Forward hook for {target_name!r} did not fire")

        hidden_states = captured["hidden"]
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"Expected [batch, length, hidden] from {target_name}, got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Hidden width mismatch at {target_name}: got {hidden_states.shape[-1]}, "
                f"model config reports {self.hidden_size}"
            )

        if mask_padding:
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        return hidden_states

    def _tokenize_batch(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(sequences)
        input_ids = torch.full(
            (batch_size, self.config.max_length),
            fill_value=self.pad_token_id,
            dtype=torch.int,
        )
        attention_mask = torch.zeros((batch_size, self.config.max_length), dtype=torch.float32)

        for row_idx, sequence in enumerate(sequences):
            token_ids = self.model.tokenizer.tokenize(str(sequence))
            token_ids = token_ids[: self.config.max_length]
            if not token_ids:
                continue
            token_tensor = torch.tensor(token_ids, dtype=torch.int)
            input_ids[row_idx, : len(token_ids)] = token_tensor
            attention_mask[row_idx, : len(token_ids)] = 1.0

        return input_ids, attention_mask


def masked_mean_sequence(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
    token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return masked_hidden.sum(dim=1) / token_counts


def masked_mean_hidden_dim(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
    return masked_hidden.sum(dim=2) / attention_mask.clamp(min=1.0)


def _first_parameter_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _install_transformer_engine_import_stub() -> None:
    try:
        import transformer_engine  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    if "transformer_engine" not in sys.modules:
        transformer_engine_module = types.ModuleType("transformer_engine")
        transformer_engine_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["transformer_engine"] = transformer_engine_module

    if "transformer_engine.pytorch" not in sys.modules:
        pytorch_module = types.ModuleType("transformer_engine.pytorch")
        sys.modules["transformer_engine.pytorch"] = pytorch_module


def _patch_vortex_rotary_fallback() -> None:
    import vortex.model.rotary as vortex_rotary

    def _apply_rotary_view(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
    ) -> torch.Tensor:
        rotated = vortex_rotary.apply_rotary_emb_torch(tensor, cos, sin, interleaved=interleaved)
        tensor.copy_(rotated)
        return tensor

    def _apply_rotary_qkv_torch(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cos_k: torch.Tensor | None = None,
        sin_k: torch.Tensor | None = None,
        interleaved: bool = False,
        seqlen_offsets: int | torch.Tensor = 0,
        num_heads_q: int | None = None,
    ) -> torch.Tensor:
        if isinstance(seqlen_offsets, torch.Tensor):
            if torch.any(seqlen_offsets != 0):
                raise NotImplementedError("Rotary fallback only supports zero seqlen offsets")
        elif seqlen_offsets != 0:
            raise NotImplementedError("Rotary fallback only supports zero seqlen offsets")

        if qkv.dim() == 5:
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
        else:
            if num_heads_q is None:
                raise ValueError("num_heads_q is required for MQA/GQA rotary fallback")
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            q = qkv[:, :, :num_heads_q]
            k = qkv[:, :, num_heads_q : num_heads_q + num_heads_k]

        _apply_rotary_view(q, cos, sin, interleaved=interleaved)
        _apply_rotary_view(k, cos if cos_k is None else cos_k, sin if sin_k is None else sin_k, interleaved=interleaved)
        return qkv

    def _apply_rotary_kv_torch(
        kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        seqlen_offsets: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        if isinstance(seqlen_offsets, torch.Tensor):
            if torch.any(seqlen_offsets != 0):
                raise NotImplementedError("Rotary fallback only supports zero seqlen offsets")
        elif seqlen_offsets != 0:
            raise NotImplementedError("Rotary fallback only supports zero seqlen offsets")

        _apply_rotary_view(kv[:, :, 0], cos, sin, interleaved=interleaved)
        return kv

    vortex_rotary.apply_rotary_emb_qkv_ = _apply_rotary_qkv_torch
    vortex_rotary.apply_rotary_emb_kv_ = _apply_rotary_kv_torch

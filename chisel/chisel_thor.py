"""Chisel draft vocabulary with draft-only FP8 conversion for a BF16 target head."""
import os
import torch
from torch.nn import Parameter
from vllm.logger import init_logger
import chisel_draft_vocab as chisel

log = init_logger("vllm.chisel_thor")

def convert_fp8(head):
    from compressed_tensors.quantization import QuantizationArgs
    from mjolnir.lmhead import quantize_rows
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    original = head.weight.detach()
    quantized, scale = quantize_rows(original)
    scheme = CompressedTensorsW8A8Fp8(QuantizationArgs(num_bits=8, type="float", strategy="channel", symmetric=True), False)
    scheme.fp8_linear = init_fp8_linear_kernel(
        activation_quant_key=scheme.activation_quant_key, weight_quant_key=scheme.weight_quant_key,
        input_dtype=original.dtype, out_dtype=original.dtype, weight_shape=original.shape,
        module_name="ThorChiselDraftHead")
    head.logical_widths = [original.shape[0]]
    head.weight = Parameter(quantized, requires_grad=False)
    head.weight_scale = Parameter(scale[:, None], requires_grad=False)
    scheme.process_weights_after_loading(head)
    class DraftFP8Method(UnquantizedEmbeddingMethod):
        def apply(self, layer, x, bias=None):
            return scheme.apply_weights(layer, x, bias)
    head.quant_method = DraftFP8Method()
    generator = torch.Generator(device=original.device).manual_seed(27)
    hidden = torch.randn(8, original.shape[1], device=original.device, dtype=original.dtype, generator=generator)
    got = head.quant_method.apply(head, hidden).float()
    reference = torch.nn.functional.linear(hidden, original).float()
    error = float((got-reference).norm()/reference.norm())
    assert error < 0.05, ("FP8 head numerical check failed", error)
    log.warning("CHISEL_THOR_FP8 kernel=%s rows=%d relative_error=%.6f", type(scheme.fp8_linear).__name__, original.shape[0], error)

def install():
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator
    cls = speculator.DFlashSpeculator
    if getattr(cls, "_chisel_thor_installed", False):
        return
    original_load = cls.load_draft_model
    def load(self, target_model, *args, **kwargs):
        model = original_load(self, target_model, *args, **kwargs)
        target_head = model.lm_head
        target_weight = target_head.weight
        target_ptr = target_weight.data_ptr()
        sample = target_weight[::4096].clone()
        chisel._reduce(model, self.vllm_config)
        assert model.lm_head is not target_head
        ids = chisel._load_ids(chisel._setting(self.vllm_config))
        assert len(ids) in (100032, 248320)
        actual = model.draft_id_to_target_id + torch.arange(len(ids), device=target_weight.device)
        assert torch.equal(actual, torch.tensor(ids, device=target_weight.device))
        if os.environ.get("CHISEL_THOR_FP8") == "1":
            convert_fp8(model.lm_head)
        hidden = torch.zeros(2, target_weight.shape[1], device=target_weight.device, dtype=target_weight.dtype)
        hidden[:, 0] = 1
        raw_ids, raw_values = model.candidate_logits_processor.get_top_k_tokens(model.lm_head, hidden, model.model.candidate_selector.top_k)
        mapped_ids, mapped_values = model.compute_candidates(hidden)
        assert torch.equal(mapped_ids, actual[raw_ids])
        assert torch.equal(mapped_values, raw_values)
        assert target_head.weight is target_weight and target_weight.data_ptr() == target_ptr
        assert torch.equal(sample, target_weight[::4096])
        assert target_weight.shape == (248320, 5120) and target_weight.dtype == torch.bfloat16
        log.warning("CHISEL_THOR_VERIFIED vocabulary=%d mapping=PASS target_head_unchanged=PASS draft_fp8=%s", len(ids), os.environ.get("CHISEL_THOR_FP8", "0"))
        torch.cuda.empty_cache()
        return model
    cls.load_draft_model = load
    cls._chisel_thor_installed = True

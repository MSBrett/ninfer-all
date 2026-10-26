"""Exact ModelOpt checkpoint-to-artifact coverage, including an encoded MTP shortlist."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.nvfp4 import encode_nvfp4
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import qwen3_8_27b_nvfp4_nvidia
from tools.convert.pipeline import convert
from tools.convert.proposal import add_official_proposal
from tools.convert.recipe import Recipe
from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.logical import LogicalSource, array_source
from tools.convert.sources.safetensors import SafetensorsSource


@pytest.mark.parametrize("weight_scales,input_scales,pair_policy", [
    ((2.0, 2.0), (3.0, 3.0), "AllowA4"),
    ((2.0, 8.0), (3.0, 3.0), "AllowA8"),
    ((2.0, 2.0), (3.0, 4.0), "AllowA8"),
    ((2.0, 8.0), (3.0, 4.0), "AllowA8"),
])
def test_nvidia_head_proposal_and_distinct_mlp_scales_are_imported_exactly(
    tmp_path, weight_scales, input_scales, pair_policy,
):
    k, vocab = 128, 256
    model = Model({"text": {"config": {"vocab_size": vocab}},
                   "mtp": {"config": {}, "target": "text"}}, token_count=vocab)
    embedding = torch.zeros((vocab, k), dtype=torch.bfloat16)
    model.add(Parameter("text/token_embedding", embedding.shape,
                        array_source(embedding, "embedding")))
    tensors, expected = {}, {}

    def no_values(begin, end):
        raise AssertionError("encoded import must not decode or requantize source weights")

    for index, (name, prefix, rows, multiplier) in enumerate((
        ("text/output_head", "lm_head", vocab, 4.0),
        ("text/layers/0/mlp/gate", "gate", 128, weight_scales[0]),
        ("text/layers/0/mlp/up", "up", 128, weight_scales[1]),
    )):
        codes = ((torch.arange(rows * k // 2) + 17 * index) % 256).to(torch.uint8).reshape(rows, k // 2)
        scales = ((torch.arange(rows * k // 16) + index) % 127).to(torch.uint8).reshape(rows, k // 16)
        tensors.update({prefix + ".weight": codes,
                        prefix + ".weight_scale": scales.view(torch.float8_e4m3fn),
                        prefix + ".weight_scale_2": torch.tensor([multiplier]),
                        prefix + ".input_scale": torch.tensor([2.0 if index == 0 else input_scales[index - 1]])})
        expected[name] = (codes, scales, struct.pack("<f", 1.0 / multiplier))

        def source_factory(store, fmt, prefix=prefix, rows=rows):
            source = matrix_source(store, prefix + ".weight", (rows, k), fmt)
            return LogicalSource(source.shape, source.label, no_values,
                                 source.read_encoded, source.weight_divisor, source.input_divisor)

        model.add(Parameter(name, (rows, k), LogicalSource((rows, k), name, no_values),
                            source_factory=source_factory,
                            inputs=("text/final_hidden", "mtp/final_hidden")
                            if name == "text/output_head" else ("input",)))
    model.packing_groups = [("text/layers/0/mlp/gate", "text/layers/0/mlp/up")]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    ranking = tmp_path / "ranking.i64"
    np.arange(vocab, dtype="<i8").tofile(ranking)
    output = tmp_path / "nvidia.ninfer"
    with SafetensorsSource(tmp_path) as store:
        recipe = Recipe(model)
        qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": store})
        add_official_proposal(recipe, ranking=ranking, rows=128)
        ids = model.parameters["proposal/token_ids"].source.values().long()
        codes, scales, divisor = expected["text/output_head"]
        expected["proposal/head"] = (codes[ids], scales[ids], divisor)
        convert(model, recipe, output, device="cpu", rows_per_chunk=128)
    with Artifact(output) as artifact:
        parents = {}
        for name in expected:
            (parent, begin, end), = binding_parts(artifact.directory.bindings[name], artifact.by_id)
            parents[name] = parent
            obj = artifact.object(parent)
            assert obj.format == "nvfp4"
            if name in ("text/output_head", "proposal/head"):
                codes, scales, divisor = expected[name]
                assert (begin, end) == (0, codes.shape[0] * k)
                assert artifact.read_object(parent) == encode_nvfp4(codes, scales, divisor, (codes.shape[0], k))
        gate, up = "text/layers/0/mlp/gate", "text/layers/0/mlp/up"
        assert parents[gate] == parents[up]
        gate_codes, gate_scales, gate_divisor = expected[gate]
        up_codes, up_scales, up_divisor = expected[up]
        combined = encode_nvfp4(torch.cat((gate_codes, up_codes)),
                               torch.cat((gate_scales, up_scales)), gate_divisor, (256, k))
        if gate_divisor != up_divisor:
            combined += up_divisor
        assert artifact.read_object(parents[gate]) == combined
        assert artifact.object(parents[gate]).divisors == (1 if gate_divisor == up_divisor else 2)
        uses = {(use["parameter"], use["input"]): use for use in artifact.directory.uses}
        for name, input_name, input_scale in (
            ("text/output_head", "text/final_hidden", 2.0),
            ("text/output_head", "mtp/final_hidden", 2.0),
            ("proposal/head", "mtp/final_hidden", 2.0),
            ("text/layers/0/mlp/gate", "input", input_scales[0]),
            ("text/layers/0/mlp/up", "input", input_scales[1]),
        ):
            use = uses[(name, input_name)]
            assert use["activation_policy"] == (pair_policy if "/mlp/" in name else "AllowA4")
            (parent, _, _), = binding_parts(use["auxiliaries"]["activation_input_divisor"], artifact.by_id)
            assert artifact.read_object(parent) == struct.pack("<f", 1.0 / input_scale)

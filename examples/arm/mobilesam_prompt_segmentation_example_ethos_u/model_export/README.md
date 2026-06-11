# MobileSAM Export, Quantization, and Debugging

This directory exports MobileSAM `vit_t` for the ExecuTorch Ethos-U backend.
The exporter freezes one or more positive point prompts into the graph, so the
runtime app has one tensor input: the preprocessed image. The model returns one
mask-logit tensor for those prompts. The default export uses a `512x512` input,
which produces a `128x128` mask-logit tensor.

Production SAM applications usually pass prompts dynamically from a UI or
tracking pipeline. This example freezes the prompt embeddings so the FVP
runtime stays small and demonstrates the Ethos-U flow with the same image
encoder and mask decoder used by MobileSAM.

## Requirements

- Python 3.10+ with `executorch`.
- Dependencies from
  `examples/arm/mobilesam_prompt_segmentation_example_ethos_u/requirements.txt`.
- Git and internet access to prepare the pinned external MobileSAM checkout and
  download the official checkpoint, unless both are already cached.
- Ethos-U dependencies from `examples/arm/setup.sh`.

## Export

Run from the ExecuTorch repo root:

```bash
MOBILE_SAM_SOURCE="$HOME/.cache/executorch/mobilesam/f706ad9c4eb7f219c00d9050e46328518ffb65d2/source"
python examples/arm/mobilesam_prompt_segmentation_example_ethos_u/model_export/prepare_mobilesam.py \
  --source-dir "$MOBILE_SAM_SOURCE"

python examples/arm/mobilesam_prompt_segmentation_example_ethos_u/model_export/export_mobilesam.py \
  --output-path ./mobilesam_point_ethos_u85_512.pte \
  --mobile-sam-source "$MOBILE_SAM_SOURCE" \
  --calibration-image examples/models/dinov2/dog.jpg \
  --eval-image examples/models/dinov2/dog.jpg \
  --point 210 220 \
  --artifact-dir ./mobilesam_point_artifacts \
  --debug-output-dir ./mobilesam_point_debug
```

The default configuration is:

- Model source: `https://github.com/ChaoningZhang/MobileSAM`
- MobileSAM source revision: `f706ad9c4eb7f219c00d9050e46328518ffb65d2`
- External source patch: `0001-Make-TinyViT-image-size-configurable.patch`
- Checkpoint URL:
  `https://github.com/ChaoningZhang/MobileSAM/raw/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt`
- Checkpoint SHA256:
  `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`
- MobileSAM source-code license: Apache-2.0
- Static input shape: `[1, 3, 512, 512]`
- Output mask logits shape: `[1, 1, 128, 128]`
- Positive point prompt: `(256, 256)` in the padded `512x512` model input
  frame
- Target: `ethos-u85-256`
- System config: `Ethos_U85_SYS_DRAM_Mid`
- Memory mode: `Dedicated_Sram_384KB`
- Calibration samples: `4`
- Validation samples: `4`

The MobileSAM source and checkpoint are not redistributed by this example. The
preparation script clones the pinned official source into a managed external
cache and applies the configurable-input patch. The exporter downloads the
pinned official checkpoint and verifies its SHA256 unless `--checkpoint-path`
is provided. The source repository and checkpoint may have separate terms; the
export metadata records only the source-code license and does not assign a
license to the checkpoint.

For a quick offline smoke test after caching the official checkpoint and source
checkout:

```bash
python examples/arm/mobilesam_prompt_segmentation_example_ethos_u/model_export/prepare_mobilesam.py \
  --source-dir "$MOBILE_SAM_SOURCE" \
  --local-files-only

python examples/arm/mobilesam_prompt_segmentation_example_ethos_u/model_export/export_mobilesam.py \
  --output-path /tmp/mobilesam_smoke.pte \
  --local-files-only \
  --mobile-sam-source "$MOBILE_SAM_SOURCE" \
  --calibration-image examples/models/dinov2/dog.jpg \
  --eval-image examples/models/dinov2/dog.jpg \
  --point 210 220 \
  --num-calibration-samples 1 \
  --num-eval-samples 1
```

Repeat `--point X Y` to freeze a multi-point prompt into the graph. Multiple
positive points can improve mask quality for reduced input sizes:

```bash
python examples/arm/mobilesam_prompt_segmentation_example_ethos_u/model_export/export_mobilesam.py \
  --output-path /tmp/mobilesam_multipoint_smoke.pte \
  --local-files-only \
  --mobile-sam-source "$MOBILE_SAM_SOURCE" \
  --calibration-image examples/models/dinov2/dog.jpg \
  --eval-image examples/models/dinov2/dog.jpg \
  --point 190 180 \
  --point 250 220 \
  --point 330 210 \
  --num-calibration-samples 1 \
  --num-eval-samples 1 \
  --debug-output-dir /tmp/mobilesam_multipoint_debug
```

Pass `--input-size 1024` to reproduce the original MobileSAM resolution. Smaller
inputs export and run faster, but very small inputs such as `224` or `256`
usually produce lower-quality masks because the checkpoint was trained for
`1024x1024` images.

To validate against a known binary mask, pass one `--eval-mask` per
`--eval-image`. Non-zero mask pixels are treated as foreground. When no
reference mask is provided, validation reports FP32/quantized mask agreement.

The export flow:

1. Loads the MobileSAM `vit_t` checkpoint through the patched external API.
2. Builds a fixed-prompt wrapper containing the image encoder and mask decoder.
3. Calibrates PT2E quantization with `EthosUQuantizer`.
4. Uses stable softmax decomposition for transformer attention blocks.
5. Lowers the quantized graph with `EthosUPartitioner`.
6. Writes an ExecuTorch `.pte` program.

The quantization recipe uses int8 activations and int8 weights globally, with
A16W8 quantization for the TinyViT attention modules. Static int8 activation
quantization collapses MobileSAM attention features, while the selective A16W8
attention path preserves mask quality and still lowers as one Ethos-U delegate.

## Outputs

For `--output-path ./mobilesam_point_ethos_u85_512.pte`, the script writes:

- `mobilesam_point_ethos_u85_512.pte` - Ethos-U-ready ExecuTorch program.
- `mobilesam_point_ethos_u85_512.json` - Export metadata.
- `mobilesam_point_ethos_u85_512_metrics.json` - FP32/quantized mask
  agreement and optional reference-mask IoU.
- `mobilesam_point_ethos_u85_512_delegation.txt` - Operator delegation
  summary.
- `mobilesam_point_artifacts/` - Optional TOSA/Vela intermediate artifacts.
- `mobilesam_point_debug/` - Optional per-sample masks, overlays, mismatch
  heatmaps, and mask summaries.

## Interpreting the Debug Artifacts

Each debug sample contains:

- `input.png` - The resized RGB input used by the exported model.
- `reference_mask.png` - Optional binary reference mask resized to the model
  output-mask size.
- `fp32_mask.png` - Host-side FP32 model prediction.
- `fp32_overlay.png` - Colored FP32 prediction blended over the input image.
- `quantized_mask.png` - Host-side PT2E quantized prediction before lowering.
- `quantized_overlay.png` - Colored quantized prediction blended over the input
  image.
- `mismatch_heatmap.png` - Green for FP32/quantized agreement and red for
  mismatch.
- `mask_summary.json` - Foreground/background pixel counts and FP32/quantized
  IoU.

The runtime app thresholds mask logits on target, logs a mask hash and
foreground/background counts, and can dump the thresholded mask as RLE chunks.
Host-side debug masks are intentionally generated before lowering so users can
inspect quantization quality without needing target-side image output.

## Limitations

- The example freezes positive point prompts into the exported graph to keep
  the target app to a single image input. Changing the prompt requires
  re-exporting the `.pte`.
- The export uses `multimask_output=False` and does not include candidate-mask
  argmax selection, upsampling, or thresholding in the graph. Keep those steps
  in host-side or target-side post-processing when comparing mask quality.
- The runtime app logs a mask hash and foreground/background counts; it does
  not render a color image on target.
- The first supported runtime target is Corstone-320/Ethos-U85-256.
- Reduced input sizes require the patch applied by `prepare_mobilesam.py`; the
  MobileSAM source remains outside the ExecuTorch checkout.
- The default `512x512` input is a compromise between FVP runtime and mask
  quality. `1024x1024` gives better masks, but is too slow for a practical FVP
  smoke example. `224x224` and `256x256` are faster, but produced poor masks on
  local demo images.
- MobileSAM PTQ is sensitive to the calibration set and the mask-logit
  threshold. Inspect `*_metrics.json` and the debug overlays before treating
  the quantized mask as an accuracy result.

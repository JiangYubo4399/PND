# PND: Breaking the Illusion: When Positive Meets Negative in Multimodal Decoding

Official implementation of **Breaking the Illusion: When Positive Meets Negative in Multimodal Decoding**, accepted at **CVPR 2026**.

PND reduces object hallucination in multimodal large language models through contrastive decoding over the original image, a positively augmented image, and a negatively augmented image. The release includes adapters for LLaVA, Qwen-VL, Qwen3-VL, and InternVL, together with POPE evaluation outputs.

## Method overview

For each image-question pair, PND:

1. obtains image-text attention maps from a BLIP image-text matching model;
2. constructs a positive view that retains question-relevant visual evidence;
3. constructs a negative view by corrupting or suppressing relevant regions;
4. combines the original, positive, and negative logits during decoding.

The decoding weights are exposed as `--alpha`, `--beta`, and `--gamma` in the model-specific entry points.

## Repository layout

```text
code/
├── script/       # PND augmentation, sampling, and evaluation entry points
├── llava/        # LLaVA model integration
├── Qwen_VL/      # Qwen-VL model integration
├── InternVL/     # InternVL model integration
└── lavis/        # BLIP/LAVIS components used for image-text attention
result/            # Released POPE evaluation outputs
```

## Environment

Create a Python environment with a CUDA-compatible PyTorch build, then install the runtime dependencies used by the selected backbone. The core PND scripts additionally require:

```bash
pip install transformers torchvision numpy scipy opencv-python matplotlib tqdm shortuuid
```

Model weights and benchmark images are not included. Download the backbone checkpoints from their official releases and prepare POPE question files in JSONL format.

## Evaluation

Run commands from the repository root.

### LLaVA

```bash
python code/script/run_llava.py \
  --model-path /path/to/llava-v1.5-7b \
  --image-folder /path/to/coco/val2014 \
  --question-file /path/to/coco_pope_adversarial.json \
  --answers-file result/llava_pnd.jsonl \
  --alpha 2.2 --beta 0.4 --gamma 0.4
```

### Qwen3-VL

```bash
python code/script/run_qwen3vl.py \
  --model-path /path/to/Qwen3-VL-2B-Instruct \
  --image-folder /path/to/coco/val2014 \
  --question-file /path/to/coco_pope_adversarial.json \
  --answers-file result/qwen3vl_pnd.jsonl \
  --use_pnd --use_negative_aug \
  --alpha 2.2 --beta 0.4 --gamma 0.4
```

### InternVL

```bash
python code/script/run_internvl_pope.py \
  --model-path /path/to/InternVL2-2B \
  --image-folder /path/to/coco/val2014 \
  --question-file /path/to/coco_pope_adversarial.json \
  --answers-file result/internvl_pnd.jsonl \
  --alpha 2.2 --beta 0.4 --gamma 0.4
```

## Released results

The `result/` directory contains outputs for multiple backbones and POPE splits, including random, popular, and adversarial settings. Files ending in `_pnd` contain PND decoding results; corresponding original-model outputs are included where available.

## Acknowledgements

This repository incorporates components from LAVIS, LLaVA, Qwen-VL, and InternVL. Their original copyright and attribution notices are preserved in the respective source files.

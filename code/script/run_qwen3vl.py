import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import set_seed, AutoTokenizer, AutoProcessor, AutoModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, '..')
from PIL import Image
import math
from lavis.models import load_model_and_preprocess
from sample_new import evolve_pnd_sampling
from torchvision import transforms
from lavis.common.registry import registry
from augmentation import augmentation 
from neg_augmentation import negative_augmentation
import types
from typing import List, Dict, Any, Tuple, Optional

def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[Optional[List], Optional[List]]:
    """Process vision information from messages"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if "content" in message:
            for content_item in message["content"]:
                if content_item.get("type") == "image":
                    image_path = content_item.get("image")
                    if image_path:
                        if isinstance(image_path, str):
                            image = Image.open(image_path).convert("RGB")
                        elif isinstance(image_path, Image.Image):
                            image = image_path.convert("RGB")
                        else:
                            image = image_path
                        image_inputs.append(image)
                        
                elif content_item.get("type") == "video":
                    video_path = content_item.get("video")
                    if video_path:
                        video_inputs.append(video_path)
    
    image_inputs = image_inputs if image_inputs else None
    video_inputs = video_inputs if video_inputs else None
    
    return image_inputs, video_inputs

def disable_torch_init():
    """Disable default initialization for faster model loading"""
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# Inject CD methods for Qwen3-VL
def prepare_inputs_for_generation_cd(self, input_ids, **kwargs):
    """Prepare inputs for contrastive decoding with positive augmentation"""
    kwargs_clean = {k: v for k, v in kwargs.items() 
                   if k not in ['pixel_values', 'image_grid_thw', 'images_cd',
                              'pixel_values_cd', 'image_grid_thw_cd',
                              'pixel_values_cd_negative', 'image_grid_thw_cd_negative',
                              'images_cd_negative', 'cd_alpha', 'cd_beta', 'cd_gamma']}
    
    if "pixel_values_cd" in kwargs and kwargs["pixel_values_cd"] is not None:
        kwargs_clean["pixel_values"] = kwargs["pixel_values_cd"]
        # print(f"[CD Positive] Using pixel_values_cd")
    
    if "image_grid_thw" in kwargs and kwargs["image_grid_thw"] is not None:
        kwargs_clean["image_grid_thw"] = kwargs["image_grid_thw"]
        # print(f"[CD Positive] Using original image_grid_thw to match input_ids")
    
    model_inputs = self.prepare_inputs_for_generation(input_ids, **kwargs_clean)
    
    return model_inputs

def prepare_inputs_for_generation_cd_negative(self, input_ids, **kwargs):
    """Prepare inputs for contrastive decoding with negative augmentation"""
    kwargs_clean = {k: v for k, v in kwargs.items() 
                   if k not in ['pixel_values', 'image_grid_thw', 'images_cd',
                              'pixel_values_cd', 'image_grid_thw_cd',
                              'pixel_values_cd_negative', 'image_grid_thw_cd_negative',
                              'images_cd_negative', 'cd_alpha', 'cd_beta', 'cd_gamma']}
    
    if "pixel_values_cd_negative" in kwargs and kwargs["pixel_values_cd_negative"] is not None:
        kwargs_clean["pixel_values"] = kwargs["pixel_values_cd_negative"]
        # print(f"[CD Negative] Using pixel_values_cd_negative")
    
    if "image_grid_thw" in kwargs and kwargs["image_grid_thw"] is not None:
        kwargs_clean["image_grid_thw"] = kwargs["image_grid_thw"]
        # print(f"[CD Negative] Using original image_grid_thw to match input_ids")
    
    model_inputs = self.prepare_inputs_for_generation(input_ids, **kwargs_clean)
    
    return model_inputs

evolve_pnd_sampling()

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Qwen3-VL model
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen3-vl'
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
    
    # print("Injecting CD methods into model...")
    model.prepare_inputs_for_generation_cd = types.MethodType(prepare_inputs_for_generation_cd, model)
    model.prepare_inputs_for_generation_cd_negative = types.MethodType(prepare_inputs_for_generation_cd_negative, model)
    # print("CD methods injected successfully!")
    # ========================================================

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Load ITM model for augmentation
    model_itm, image_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )
    loader = transforms.Compose([transforms.ToTensor()])

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]
        
        # For POPE
        prompt = question + " Please answer this question with one word."
        image_path = os.path.join(args.image_folder, image_file)
        
        # Load raw image
        raw_image = Image.open(image_path).convert("RGB")
        
        # Prepare messages for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process standard input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        # Initialize variables for augmented images
        pixel_values_cd = None
        image_grid_thw_cd = None
        pixel_values_cd_negative = None
        image_grid_thw_cd_negative = None
        
        # Prepare common preprocessing for both positive and negative augmentation
        if args.use_pnd or args.use_negative_aug:
            tensor_image = loader(raw_image.resize((384, 384)))
            image = image_processors["eval"](raw_image).unsqueeze(0).to(device)
            question_processed = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(
                question_processed, padding='longest', truncation=True, return_tensors="pt"
            ).to('cuda')

        if args.use_pnd:
            augmented_image = augmentation(
                image, question_processed, tensor_image, model_itm, tokenized_text, raw_image
            )
            # Resize augmented image to match original image size to ensure same token count
            augmented_image = augmented_image.resize(raw_image.size, Image.LANCZOS)
            
            # Process augmented image - use same text to ensure token alignment
            messages_cd = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": augmented_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_cd = processor.apply_chat_template(messages_cd, tokenize=False, add_generation_prompt=True)
            image_inputs_cd, video_inputs_cd = process_vision_info(messages_cd)
            inputs_cd = processor(
                text=[text_cd],
                images=image_inputs_cd,
                videos=video_inputs_cd,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            # Only extract pixel_values and image_grid_thw, don't use input_ids
            pixel_values_cd = inputs_cd.pixel_values
            image_grid_thw_cd = inputs_cd.image_grid_thw
            
        # Apply negative augmentation
        if args.use_negative_aug:
            negative_image = negative_augmentation(
                image, question_processed, tensor_image, model_itm, tokenized_text, raw_image
            )
            # Resize negative image to match original image size to ensure same token count
            negative_image = negative_image.resize(raw_image.size, Image.LANCZOS)
            
            # Process negative augmented image - use same text to ensure token alignment
            messages_neg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": negative_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_neg = processor.apply_chat_template(messages_neg, tokenize=False, add_generation_prompt=True)
            image_inputs_neg, video_inputs_neg = process_vision_info(messages_neg)
            inputs_cd_negative = processor(
                text=[text_neg],
                images=image_inputs_neg,
                videos=video_inputs_neg,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            # Only extract pixel_values and image_grid_thw, don't use input_ids
            pixel_values_cd_negative = inputs_cd_negative.pixel_values
            image_grid_thw_cd_negative = inputs_cd_negative.image_grid_thw
        
        # Generate with contrastive decoding
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                do_sample=True,
                max_new_tokens=20,
                min_new_tokens=1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                # For old Qwen-VL, these were called images_cd
                # For Qwen3-VL, we pass pixel_values_cd
                images_cd=pixel_values_cd,  # This will be used by sample_new.py
                pixel_values_cd=pixel_values_cd,
                image_grid_thw_cd=image_grid_thw_cd,
                images_cd_negative=pixel_values_cd_negative,  # This will be used by sample_new.py
                pixel_values_cd_negative=pixel_values_cd_negative,
                image_grid_thw_cd_negative=image_grid_thw_cd_negative,
                cd_alpha=args.alpha,
                cd_beta=args.beta,
                cd_gamma=args.gamma,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        outputs = outputs.strip()
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--image-folder", type=str, default=".coco/val2014")
    parser.add_argument("--question-file", type=str, default="./data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="./qwen3vl_coco_pope_adversarial_output.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_pnd", action='store_true', default=False)
    parser.add_argument("--alpha", type=float, default=2.2)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--use_negative_aug", action='store_true', default=False)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
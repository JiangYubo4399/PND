import argparse
import torch
import os

import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import set_seed, AutoTokenizer
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria    
from PIL import Image
import math
from lavis.models import load_model_and_preprocess
from sample_internvl import evolve_pnd_sampling
from torchvision import transforms
from augmentation import augmentation 
from neg_augmentation import negative_augmentation
from InternVL.modeling_internvl_chat import InternVLChatModel

evolve_pnd_sampling()


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_transform(input_size):
    from torchvision.transforms.functional import InterpolationMode
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= 1)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load InternVL2-2B model
    model_path = os.path.expanduser(args.model_path)
    assert os.path.exists(model_path), f"Model path not found: {model_path}"
    model_name = 'internvl2-2b'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

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
        
        # Format prompt for POPE
        prompt = question + " Please answer this question with one word."
        image_path = os.path.join(args.image_folder, image_file)
        # Load and process image
        raw_image = Image.open(image_path).convert("RGB")
        raw_image = raw_image.resize((448, 448), Image.BICUBIC)
        pixel_values = load_image(raw_image, input_size=448).to(torch.bfloat16).cuda()
        
        # Initialize variables for augmented images
        pixel_values_cd = None
        pixel_values_cd_negative = None
        
        # Prepare preprocessing for augmentation if needed
        if args.use_pnd or args.use_negative_aug:
            tensor_image = loader(raw_image.resize((448, 448)))
            image = image_processors["eval"](raw_image).unsqueeze(0).to(device)
            question_processed = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(
                question_processed, 
                padding='longest', 
                truncation=True, 
                return_tensors="pt"
            ).to(device)

        # Apply positive augmentation
        if args.use_pnd:
            augmented_image = augmentation(
                image, question_processed, tensor_image, 
                model_itm, tokenized_text, raw_image
            )
            pixel_values_cd = load_image(augmented_image, input_size=448).to(torch.bfloat16).cuda()
            
        # Apply negative augmentation
        if args.use_negative_aug:
            negative_image = negative_augmentation(
                image, question_processed, tensor_image, 
                model_itm, tokenized_text, raw_image
            )
            pixel_values_cd_negative = load_image(negative_image, input_size=448).to(torch.bfloat16).cuda()
        
        # Generate response with InternVL2-2B using contrastive decoding
        generation_config = {
            'do_sample': True,
            'max_new_tokens': 20,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
        }
        
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
            pixel_values_cd=pixel_values_cd,
            pixel_values_cd_negative=pixel_values_cd_negative,
            cd_alpha=args.alpha,
            cd_beta=args.beta,
            cd_gamma=args.gamma,
        )
        
        outputs = response.strip()
        
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "text": outputs,
            "model_id": model_name,
            "image": image_file,
            "metadata": {}
        }) + "\n")
        ans_file.flush()
    
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./InternVL2-2B")
    parser.add_argument("--image-folder", type=str, default="./coco/val2014")
    parser.add_argument("--question-file", type=str, default="./coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="./InternVL_pope_adversarial.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_pnd", action='store_true', default=True)
    parser.add_argument("--alpha", type=float, default=2.2)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--use_negative_aug", action='store_true', default=True)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
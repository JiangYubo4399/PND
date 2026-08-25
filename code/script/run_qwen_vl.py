import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import set_seed, AutoTokenizer
from Qwen_VL.modeling_qwen import QWenLMHeadModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, '..')

from llava.utils import disable_torch_init
from PIL import Image
import math

from sample_new import evolve_pnd_sampling
from torchvision import transforms
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from augmentation import augmentation 
from neg_augmentation import negative_augmentation
evolve_pnd_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_mme_data(mme_folder):
    """Parse MME benchmark data structure"""
    categories = []
    
    # Get all subdirectories in MME_Benchmark
    for category in os.listdir(mme_folder):
        category_path = os.path.join(mme_folder, category)
        if os.path.isdir(category_path):
            categories.append(category)
    
    return categories

def load_mme_questions(category_path):
    """Load questions from MME category folder"""
    questions = []
    
    # Get all txt files in the category folder
    for filename in os.listdir(category_path):
        if filename.endswith('.txt'):
            txt_path = os.path.join(category_path, filename)
            img_filename = filename.replace('.txt', '.jpg')
            img_path = os.path.join(category_path, img_filename)
            
            # Check if corresponding image exists
            if os.path.exists(img_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if line:
                        # Split question and answer
                        parts = line.split('\t')
                        if len(parts) == 2:
                            question = parts[0]
                            ground_truth = parts[1]
                            
                            questions.append({
                                'image_file': img_filename,
                                'question': question,
                                'ground_truth': ground_truth
                            })
    
    return questions

def eval_model(args):
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Qwen-VL model (replacing LLaVA)
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen-vl'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    # Create output directory
    output_dir = args.output_folder
    os.makedirs(output_dir, exist_ok=True)

    # Load ITM model for augmentation
    model_itm, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    loader = transforms.Compose([transforms.ToTensor()])

    # Get all MME categories
    mme_folder = args.mme_folder
    categories = parse_mme_data(mme_folder)
    
    print(f"Found {len(categories)} categories: {categories}")
    
    # Process each category
    for category in tqdm(categories, desc="Processing categories"):
        print(f"\nProcessing category: {category}")
        
        category_path = os.path.join(mme_folder, category)
        questions = load_mme_questions(category_path)
        
        if not questions:
            print(f"No questions found in category {category}")
            continue
            
        # Create output file for this category
        output_file = os.path.join(output_dir, f"{category}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(questions, desc=f"Processing {category}"):
                image_file = item['image_file']
                question = item['question']
                ground_truth = item['ground_truth']
                
                # Load and process image
                image_path = os.path.join(category_path, image_file)
                try:
                    raw_image = Image.open(image_path).convert("RGB")
                    
                    # Construct prompt for Qwen-VL (replacing LLaVA format)
                    qwen_question = '<img>{}</img>{} Answer:'.format(image_path, question)
                    
                    # Tokenize input for Qwen-VL
                    input_ids = tokenizer([qwen_question], return_tensors='pt', padding='longest')
                    
                    # Process image for Qwen-VL
                    image_tensor = model.transformer.visual.image_transform(raw_image).unsqueeze(0).to(model.device)
                    
                    # Initialize variables for augmented images
                    image_tensor_cd = None
                    image_tensor_cd_negative = None
                    
                    # Prepare common preprocessing for both positive and negative augmentation
                    if args.use_pnd or args.use_negative_aug:
                        tensor_image = loader(raw_image.resize((384,384)))
                        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                        question_processed = text_processors["eval"](question)
                        tokenized_text = model_itm.tokenizer(question_processed, padding='longest', truncation=True, return_tensors="pt").to('cuda')

                    # Apply positive augmentation
                    if args.use_pnd:
                        augmented_image = augmentation(image, question_processed, tensor_image, model_itm, tokenized_text, raw_image)
                        image_tensor_cd = model.transformer.visual.image_transform(augmented_image).unsqueeze(0).to(model.device)
                        # print("Positive augmentation applied")
                        
                    # Apply negative augmentation
                    if args.use_negative_aug:
                        negative_image = negative_augmentation(image, question_processed, tensor_image, model_itm, tokenized_text, raw_image)
                        image_tensor_cd_negative = model.transformer.visual.image_transform(negative_image).unsqueeze(0).to(model.device)
                        # print("Negative augmentation applied")
                    
                    # Generate answer with Qwen-VL (replacing LLaVA generation)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids=input_ids.input_ids.cuda(),
                            attention_mask=input_ids.attention_mask.cuda(),
                            images=image_tensor,
                            images_cd=image_tensor_cd,
                            images_cd_negative=image_tensor_cd_negative,
                            cd_alpha=args.alpha,
                            cd_beta=args.beta,
                            cd_gamma=args.gamma,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_new_tokens=1024,
                            min_new_tokens=1,
                            length_penalty=1,
                            num_return_sequences=1,
                            output_hidden_states=True,
                            use_cache=True,
                            pad_token_id=tokenizer.eod_id,
                            eos_token_id=tokenizer.eod_id
                        )

                    # Decode Qwen-VL output
                    outputs = [
                        tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                                         skip_special_tokens=True).strip() for _ in output_ids
                    ][0]
                    output = outputs.strip()
                    
                    # Write to output file in MME format: image_file\tquestion\tground_truth\tmodel_answer
                    f.write(f"{image_file}\t{question}\t{ground_truth}\t{output}\n")
                    f.flush()
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    # Write error case
                    f.write(f"{image_file}\t{question}\t{ground_truth}\tERROR\n")
                    f.flush()
        
        print(f"Completed category {category}, results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model related arguments (updated for Qwen-VL)
    parser.add_argument("--model-path", type=str, default="./Qwen-VL",
                        help="Path to the Qwen-VL model")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path")
    parser.add_argument("--conv-mode", type=str, default="qwen_vl",
                        help="Conversation mode")

    # Data arguments
    parser.add_argument("--mme-folder", type=str, default="./MME_Benchmark", 
                        help="Path to MME_Benchmark folder")
    parser.add_argument("--output-folder", type=str, default="./OURS-qwen-vl-pnd-all", 
                        help="Output folder for evaluation results")
    
    # Processing arguments
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    
    # Augmentation arguments
    parser.add_argument("--use_pnd", action='store_true', default=True)
    parser.add_argument("--alpha", type=float, default=2.2)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--use_negative_aug", action='store_true', default=True)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
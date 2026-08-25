import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import matplotlib.pyplot as plt
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from lavis.common.gradcam import getAttMap
from torchvision import transforms
def negative_augmentation(
    image,                # torch.Tensor, 1×C×H×W  
    question,             # preprocessed text input
    tensor_image,         # 
    model,                # BLIP-ITM 
    tokenized_text,       # tokenized_text
    raw_image,            # PIL.Image, 
    strategy="attention_reversal",  # 
    **kwargs
):

    if strategy == "attention_reversal":
        return targeted_destruction_augmentation(image, question, tensor_image, model, tokenized_text, raw_image)
    elif strategy == "noise_injection":
        return noise_injection_augmentation(image, question, tensor_image, model, tokenized_text, raw_image)
    elif strategy == "region_removal":
        return region_removal_augmentation(image, question, tensor_image, model, tokenized_text, raw_image)
    elif strategy == "semantic_corruption":
        return semantic_corruption_augmentation(raw_image)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def targeted_destruction_augmentation(
    image, 
    question, 
    tensor_image,
    model, 
    tokenized_text, 
    raw_image, 
    method='diffusion_ddpm', 
    destruction_level=999,  
    threshold=0.6 
):
   
    block_nums = (3, 5, 7, 9)
    target_size = 384
    all_positive_cams = []

    for block_num in block_nums:
        gradcams, _ = compute_gradcam(
            model=model, visual_input=image, text_input=question, 
            tokenized_text=tokenized_text, block_num=block_num
        )
        raw_maps = [g[1] for g in gradcams]
        stacked = torch.stack(raw_maps).reshape(image.size(0), -1)
        cam_feat = stacked.reshape(-1, )
        
        avg_gradcam = getAttMap(
            np.float32(raw_image.resize((target_size, target_size))) / 255,
            cam_feat.cpu().numpy().reshape(gradcams[0][1].shape),
            blur=True, overlap=False
        )
        normalized_cam = (avg_gradcam - avg_gradcam.min()) / (avg_gradcam.max() - avg_gradcam.min() + 1e-8)
        all_positive_cams.append(normalized_cam)

    fused_cam = np.minimum.reduce(all_positive_cams)
    fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)


    destruction_mask = (fused_cam >= threshold).astype(np.float32)


    resizer = transforms.Resize((target_size, target_size))
    resized_tensor_image = resizer(tensor_image)

    if method == 'gaussian':

        sigma = destruction_level
        img_np = resized_tensor_image.permute(1, 2, 0).cpu().numpy()
        img_np_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        damaged_img_np = gaussian_filter(img_np_uint8, sigma=(sigma, sigma, 0))

        damaged_tensor = transforms.ToTensor()(damaged_img_np)

    elif method == 'diffusion_ddpm':

        num_steps = 1000
        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        def q_x(x_0, t):

            t_int = t.item() if isinstance(t, torch.Tensor) else int(t)
            noise = torch.randn_like(x_0)
            alphas_t = alphas_bar_sqrt[t_int]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t_int]
            return (alphas_t * x_0 + alphas_1_m_t * noise)


        noise_step = int(destruction_level)
        if not (0 <= noise_step < num_steps):
            raise ValueError(f"noise_step (destruction_level) must be between 0 and {num_steps-1}")


        damaged_tensor = q_x(resized_tensor_image, torch.tensor(noise_step))
        
    else:
        raise ValueError("Method must be 'gaussian' or 'diffusion_ddpm'")


    mask_tensor = torch.from_numpy(destruction_mask).to(resized_tensor_image.device).float()

    mask_tensor = mask_tensor.unsqueeze(0).expand_as(resized_tensor_image)

    final_tensor = resized_tensor_image * (1 - mask_tensor) + damaged_tensor * mask_tensor
    final_tensor = torch.clamp(final_tensor, 0, 1) 
    unloader = transforms.ToPILImage()
    return unloader(final_tensor.cpu())

def attention_reversal_augmentation1(image, question, tensor_image, model, tokenized_text, raw_image):

    from lavis.common.gradcam import getAttMap
    from lavis.models.blip_models.blip_image_text_matching import compute_gradcam

    block_nums = (7, 8, 9, 10)
    target_size = 384

    all_reversed_cams = []
    for block_num in block_nums:
        gradcams, _ = compute_gradcam(
            model=model,
            visual_input=image,
            text_input=question,
            tokenized_text=tokenized_text,
            block_num=block_num
        )
        raw_maps = [g[1] for g in gradcams]
        stacked = torch.stack(raw_maps).reshape(image.size(0), -1)
        
        cam_feat = stacked.reshape(-1, )
        avg_gradcam = getAttMap(
            np.float32(raw_image.resize((target_size, target_size))) / 255,
            cam_feat.cpu().numpy().reshape(gradcams[0][1].shape),
            blur=True,
            overlap=False
        )
        
        normalized_cam = (avg_gradcam - avg_gradcam.min()) / (avg_gradcam.max() - avg_gradcam.min() + 1e-8)
        

        reversed_cam = 1.0 - normalized_cam
        all_reversed_cams.append(reversed_cam)

    reversed_mask = np.minimum.reduce(all_reversed_cams)

    reversed_mask = np.power(reversed_mask, 1.5)  
    

    reversed_mask = (reversed_mask - reversed_mask.min()) / (reversed_mask.max() - reversed_mask.min() + 1e-8)
    
  
    from scipy.ndimage import gaussian_filter
    reversed_mask = gaussian_filter(reversed_mask, sigma=1.0)
    

    reversed_mask = (reversed_mask - reversed_mask.min()) / (reversed_mask.max() - reversed_mask.min() + 1e-8)
    
 
    visualize_reversed_mask_heatmap(reversed_mask, raw_image, target_size)
 
    mask_3ch = np.stack([reversed_mask] * 3, axis=2)
    img_np = tensor_image.permute(1, 2, 0).cpu().numpy()
    
  
    enhancement_factor = 0.4 + 0.6 * reversed_mask  
    suppression_factor = 1.0 - 0.3 * (1.0 - reversed_mask) 
    
 
    enhanced_regions = img_np * np.expand_dims(enhancement_factor, axis=2) 
    suppressed_regions = img_np * np.expand_dims(suppression_factor, axis=2) 
    

    final_image = (enhanced_regions * mask_3ch + 
                  suppressed_regions * (1 - mask_3ch))
    

    final_image = np.clip(final_image, 0, 1)
    

    new_tensor = torch.from_numpy(final_image).permute(2, 0, 1)
    unloader = transforms.ToPILImage()
    
    return unloader(new_tensor)


def noise_injection_augmentation(image, question, tensor_image, model, tokenized_text, raw_image):
 
    from lavis.common.gradcam import getAttMap
    from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
    

    block_nums = (7, 8, 9, 10)
    weights = [1.0 / len(block_nums)] * len(block_nums)
    target_size = 384
    
    cams = []
    for block_num, w in zip(block_nums, weights):
        gradcams, _ = compute_gradcam(
            model=model,
            visual_input=image,
            text_input=question,
            tokenized_text=tokenized_text,
            block_num=block_num
        )
        raw_maps = [g[1] for g in gradcams]
        stacked = torch.stack(raw_maps).reshape(image.size(0), -1)
        
        cam_feat = stacked.reshape(-1, )
        avg_gradcam = getAttMap(
            np.float32(raw_image.resize((target_size, target_size))) / 255,
            cam_feat.cpu().numpy().reshape(gradcams[0][1].shape),
            blur=True,
            overlap=False
        )
        cams.append((avg_gradcam * w))
    
    fused_cam = np.sum(cams, axis=0)
    fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)
    
  
    noise_strength = 0.3
    noise = np.random.normal(0, noise_strength, tensor_image.shape).astype(np.float32)
    noise_tensor = torch.from_numpy(noise)
    
  
    noise_mask = np.stack([fused_cam] * 3, axis=2)
    noise_mask_tensor = torch.from_numpy(noise_mask).permute(2, 0, 1)
    
  
    noisy_tensor = tensor_image + noise_tensor * noise_mask_tensor * 0.5
    noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
    
    unloader = transforms.ToPILImage()
    return unloader(noisy_tensor)


def region_removal_augmentation(image, question, tensor_image, model, tokenized_text, raw_image):

    from lavis.common.gradcam import getAttMap
    from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
    

    block_nums = (7, 8, 9, 10)
    target_size = 384
    
    gradcams, _ = compute_gradcam(
        model=model,
        visual_input=image,
        text_input=question,
        tokenized_text=tokenized_text,
        block_num=8  
    )
    
    raw_maps = [g[1] for g in gradcams]
    stacked = torch.stack(raw_maps).reshape(image.size(0), -1)
    cam_feat = stacked.reshape(-1, )
    
    fused_cam = getAttMap(
        np.float32(raw_image.resize((target_size, target_size))) / 255,
        cam_feat.cpu().numpy().reshape(gradcams[0][1].shape),
        blur=True,
        overlap=False
    )
    

    fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)
    
  
    removal_threshold = 0.7
    removal_mask = (fused_cam < removal_threshold).astype(np.float32)
    

    img_array = np.array(raw_image.resize((target_size, target_size)))
    blurred_img = cv2.GaussianBlur(img_array, (21, 21), 0)
    

    mask_3ch = np.stack([removal_mask] * 3, axis=2)
    result = img_array * mask_3ch + blurred_img * (1 - mask_3ch)
    result = result.astype(np.uint8)

    return Image.fromarray(result)


def semantic_corruption_augmentation(raw_image):

    enhancer = ImageEnhance.Color(raw_image)
    img = enhancer.enhance(np.random.uniform(0.2, 0.8)) 
    

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(np.random.uniform(0.3, 0.7))
    

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(np.random.uniform(0.4, 0.8))
    

    img_array = np.array(img)
    noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
    corrupted = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(corrupted)
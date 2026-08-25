import torch
import numpy as np
from lavis.common.gradcam import getAttMap
from torchvision import transforms
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from torchvision import transforms

# Generating augmentated images based on the input prompt with multi-layer CAM fusion
def augmentation(image, question, tensor_image, model, tokenized_text, raw_image, 
                 fusion_layers=[4, 6, 8], fusion_weights=None, fusion_method='average'):
    """

    """
    

    if fusion_weights is not None and len(fusion_weights) != len(fusion_layers):
        raise ValueError("fusion_weights length must match fusion_layers length")
    
 
    if fusion_weights is None:
        fusion_weights = [1.0 / len(fusion_layers)] * len(fusion_layers)
    
  
    fusion_weights = [w / sum(fusion_weights) for w in fusion_weights]
    
    with torch.set_grad_enabled(True):
        gradcams_dict = {}
        for layer_num in fusion_layers:
            gradcams, *_ = compute_gradcam(model=model,
                                visual_input=image,
                                text_input=question,
                                tokenized_text=tokenized_text,
                                block_num=layer_num)
            gradcams_dict[layer_num] = [gradcam_[1] for gradcam_ in gradcams]
    
    gradcam_stacked_list = []
    for layer_num in fusion_layers:
        gradcam_stacked = torch.stack(gradcams_dict[layer_num]).reshape(image.size(0), -1)
        gradcam_stacked_list.append(gradcam_stacked)
    

    if fusion_method == 'average':

        fused_gradcam = torch.stack(gradcam_stacked_list).mean(dim=0)
    elif fusion_method == 'weighted':

        fused_gradcam = torch.zeros_like(gradcam_stacked_list[0])
        for i, gradcam in enumerate(gradcam_stacked_list):
            fused_gradcam += fusion_weights[i] * gradcam
    else:
        raise ValueError("fusion_method must be 'average' or 'weighted'")
    

    itc_score = model({"image": image, "text_input": question}, match_head='itc')
    ratio = 1 - itc_score/2
    ratio = min(ratio, 1-10**(-5))
  
    resized_img = raw_image.resize((384, 384))
    norm_img = np.float32(resized_img) / 255
    

    gradcam = fused_gradcam.reshape(24, 24)
    avg_gradcam = getAttMap(norm_img, gradcam.cpu().numpy(), blur=True, overlap=False)
    

    temp, *_ = torch.sort(torch.tensor(avg_gradcam).reshape(-1), descending=True)
    cam1 = torch.tensor(avg_gradcam).unsqueeze(2)
    cam = torch.cat([cam1, cam1, cam1], dim=2)
    mask = torch.where(cam < temp[int(384 * 384 * ratio)], 0, 1)
    
  
    new_image = tensor_image.permute(1, 2, 0) * mask
    unloader = transforms.ToPILImage()
    imag = new_image.clone().permute(2, 0, 1) 
    imag = unloader(imag)
    
    return imag


import torch
import numpy as np
import os
from PIL import Image
from typing import List
import textwrap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Utils:
    @staticmethod
    def set_seed(seed: int = 42):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def load_image(img_path: str, target_size: int = 512) -> torch.Tensor:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        pil_img = Image.open(img_path).convert('RGB')
        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        # [0, 1] float
        tensor_img = torch.from_numpy(np.array(pil_img)).float() / 255.0
        # [H, W, C] -> [1, C, H, W]
        tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0)
        return tensor_img

    @staticmethod
    def latents_to_img(latents: torch.Tensor, vae) -> Image.Image:
        latents = latents / vae.config.scaling_factor
        with torch.no_grad():
            image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).astype(np.uint8)
        return Image.fromarray(image)
    
    @staticmethod
    def visualize_latents(vae, latents_list: List[torch.Tensor], titles: List[str] = None, source_prompt: str = "", target_prompt: str = "", save_path: str = "vis_output.png"):
        n_imgs = len(latents_list)
        if n_imgs == 0:
            print("Warning: No latents to visualize.")
            return
        
        device = vae.device
        dtype = vae.dtype

        # 1. 解码所有 latents
        decoded_imgs = []
        with torch.no_grad():
            for z in latents_list:
                if isinstance(z, torch.Tensor):
                    z = z.to(device=device, dtype=dtype)
                    img = Utils.latents_to_img(z, vae)
                    if not isinstance(img, np.ndarray):
                        img = np.array(img)
                    decoded_imgs.append(img)
                elif isinstance(z, Image.Image):
                    decoded_imgs.append(np.array(z))

        if not decoded_imgs:
            return

        img_h, img_w, _ = decoded_imgs[0].shape
        
        dpi = 100        
        # 给title和prompt留空间
        title_height_px = 40
        text_height_px = 200
        
        # 转换为英寸
        img_width_inch = img_w / dpi
        total_img_width_inch = img_width_inch * n_imgs
        
        img_height_inch = img_h / dpi
        text_height_inch = text_height_px / dpi
        title_height_inch = title_height_px / dpi
        
        # 总高度 = 标题 + 图片 + 文本
        total_fig_height = title_height_inch + img_height_inch + text_height_inch
        total_fig_width = total_img_width_inch

        fig = plt.figure(figsize=(total_fig_width, total_fig_height), dpi=dpi)
        

        top_fraction = 1.0 - (title_height_inch / total_fig_height)
        

        gs = gridspec.GridSpec(2, 1, 
                            height_ratios=[img_height_inch, text_height_inch],
                            hspace=0.02,
                            left=0, right=1, bottom=0, 
                            top=top_fraction)

        gs_imgs = gridspec.GridSpecFromSubplotSpec(1, n_imgs, subplot_spec=gs[0], wspace=0)

        if titles is None or len(titles) != n_imgs:
            titles = [f"Step {i}" for i in range(n_imgs)]

        for idx, (img, title) in enumerate(zip(decoded_imgs, titles)):
            ax = plt.subplot(gs_imgs[0, idx])
            ax.imshow(img)
            ax.set_title(title, fontsize=12, pad=10) 
            ax.axis('off')

        ax_text = plt.subplot(gs[1])
        ax_text.axis('off')

        char_limit = int((img_w * n_imgs) / 10) 
        s_wrapper = textwrap.fill(f"Source: {source_prompt}", width=char_limit)
        t_wrapper = textwrap.fill(f"Target: {target_prompt}", width=char_limit)
        
        ax_text.text(
            0.5, 0.6, 
            f"{s_wrapper}\n{t_wrapper}", 
            ha='center', va='center', 
            fontsize=12,
            transform=ax_text.transAxes
        )

        plt.savefig(save_path, dpi=dpi)
        plt.close()
        print(f"Visualization saved to {save_path}")

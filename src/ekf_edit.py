import argparse
import torch
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from extended_kalman_filter import ExtendedKalmanFilter
from utils import Utils
from sd3_attn_processor import SD3RFEditingAttnProcessor


class EditPipeline:
    def __init__(self, model_id: str, device='cuda', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading SD3 model from {model_id}...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        # === 替换 Attention Processors ===
        self.register_rf_processors()
        
    def register_rf_processors(self):
        """将 SD3 的 Attn Processor 替换为支持 RF-Edit 的版本"""
        attn_procs = {}
        # SD3 Transformer 中有很多块，名字通常包含 'attn'
        for name, proc in self.pipe.transformer.attn_processors.items():
            if "attn" in name: 
                attn_procs[name] = SD3RFEditingAttnProcessor()
        self.pipe.transformer.set_attn_processor(attn_procs)

    def clear_memory(self):
        """清空所有 Processor 的显存"""
        for proc in self.pipe.transformer.attn_processors.values():
            if hasattr(proc, "clear_memory"):
                proc.clear_memory()
                
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # 假设输入 image 已经是 [B, C, H, W] 且范围 [0, 1]
        image = 2. * image.to(device=self.device, dtype=self.dtype) - 1.
        posterior = self.pipe.vae.encode(image).latent_dist
        # SD3 VAE Scaling: (latents - shift) * scale
        latents = posterior.mean  * self.pipe.vae.config.scaling_factor
        return latents
                
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        # SD3 VAE Unscaling: (latents / scale) + shift
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        # 后处理到 [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def get_embeds(self, prompt: str, guidance_scale: float, do_classifier_free_guidance: bool = True):
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            device=self.device, do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=1
        )
        if do_classifier_free_guidance:
            return torch.cat([negative_prompt_embeds, prompt_embeds], dim=0), \
                   torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def run_ekf_edit_rf(self, 
                    source_image: torch.Tensor,
                    source_prompt: str,
                    target_prompt: str,
                    T_steps: int = 40,
                    T_start: int = 13,
                    eta: float = 0.8,
                    src_guidance_scale: float = 3.5, 
                    tar_guidance_scale: float = 5.5, 
                    lambda_1: float = 0.1,
                    lambda_2: float = 1.0,
                    inject_steps: int = 20,
                    save_vis_path: str = "vis_output.png"):

        self.clear_memory()
        print(f"Starting EKF-Edit: Steps={T_steps}, eta={eta}, Skip={T_start}")

        latents_src = self.encode_image(source_image)
        
        kalman_filter = ExtendedKalmanFilter(shape=latents_src.shape, device=self.device, q_scale=0.01, r_scale=0.05)

        # embedding
        inv_emb, inv_pool = self.get_embeds(source_prompt, 1.0, do_classifier_free_guidance=False)
        src_emb, src_pool = self.get_embeds(source_prompt, src_guidance_scale)
        tar_emb, tar_pool = self.get_embeds(target_prompt, tar_guidance_scale)

        # do cfg
        if src_guidance_scale > 1.0 or tar_guidance_scale > 1.0:
            combined_emb = torch.cat([src_emb[:1], src_emb[1:], tar_emb[:1], tar_emb[1:]], dim=0)
            combined_pool = torch.cat([src_pool[:1], src_pool[1:], tar_pool[:1], tar_pool[1:]], dim=0)
        else:
            combined_emb = torch.cat([src_emb, tar_emb], dim=0)
            combined_pool = torch.cat([src_pool, tar_pool], dim=0)

        self.pipe.scheduler.set_timesteps(T_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        sigmas = self.pipe.scheduler.sigmas


        def second_order_diffusion_step(
            z: torch.Tensor,
            t_model_input: torch.Tensor,
            t_physical: torch.Tensor,
            dt: float,
            latent_input: torch.Tensor,
            emb: torch.Tensor,
            pool: torch.Tensor,
            guidance_scale: float,
            joint_attn_kwargs: dict
        ) -> torch.Tensor:
            
            # 一阶
            t_long = (t_model_input * 1000).long().expand(latent_input.shape[0]).to(self.device)
            
            noise_pred = self.pipe.transformer(
                hidden_states=latent_input,
                timestep=t_long,
                encoder_hidden_states=emb,
                pooled_projections=pool,
                joint_attention_kwargs=joint_attn_kwargs, 
                return_dict=True
            ).sample
                        
            if guidance_scale > 1.0:
                uncond, text = noise_pred.chunk(2)
                v1 = uncond + guidance_scale * (text - uncond)
            else:
                v1 = noise_pred

            z_mid = z + (dt / 2.0) * v1
            t_mid = t_physical + dt / 2.0
            
            joint_attn_kwargs_mid = joint_attn_kwargs.copy()
            

            is_inverse = joint_attn_kwargs.get("inverse", False)
            if is_inverse:
                joint_attn_kwargs_mid["inject"] = False # 避免重复 Record
            
            # 二阶
            t_mid_long = (t_mid * 1000).long().expand(z_mid.shape[0]).to(self.device)
            
            latent_input_mid = torch.cat([z_mid]*2) if guidance_scale > 1.0 else z_mid

            noise_pred_mid = self.pipe.transformer(
                hidden_states=latent_input_mid,
                timestep=t_mid_long,
                encoder_hidden_states=emb,
                pooled_projections=pool,
                joint_attention_kwargs=joint_attn_kwargs_mid,
                return_dict=True
            ).sample
            
            if guidance_scale > 1.0:
                uncond_mid, text_mid = noise_pred_mid.chunk(2)
                v2 = uncond_mid + guidance_scale * (text_mid - uncond_mid)
            else:
                v2 = noise_pred_mid

            first_order = (v2 - v1) / (dt / 2)
            return v1 + 0.5 * dt * first_order

        
        inv_sigmas = sigmas.flip(0)
        z_rf = latents_src.clone()
        rf_invs_dict = {}
        print(f"Phase 1: RF-Solver Inversion...")
        for t_curr, t_next in zip(tqdm(inv_sigmas[:-1], desc="Inverting"), inv_sigmas[1:]):

            # SD3 的 timestep 是 0-1000 的 float
            t_key_val = int(t_curr.item() * 1000) 
            
            joint_attn_kwargs = {
                "inverse": True,
                "inject": True,
                "timestep": t_key_val,
                "editing_strategy": "replace_v"
            }

            dt = t_next - t_curr 
            
            # inversion无cfg
            latent_input = z_rf 
                        
            v = second_order_diffusion_step(
                z=z_rf, 
                t_model_input=t_next, # 使用 t_next
                t_physical=t_curr,    # 物理时间仍是 t_curr
                dt=dt, 
                latent_input=latent_input, 
                emb=inv_emb, 
                pool=inv_pool, 
                guidance_scale=1.0, 
                joint_attn_kwargs=joint_attn_kwargs
            )
            
            z_rf = z_rf + dt * v

            rf_invs_dict[t_key_val] = v.clone().cpu()

        # 防止key error
        rf_invs_dict[1000] = torch.zeros_like(v)


        # 状态初始化
        zt_edit = latents_src.clone() 
        inject_mask = [True] * inject_steps + [False] * (len(timesteps) - inject_steps)
        for i, _ in enumerate(tqdm(timesteps, desc="EKF Loop")):

            t_curr_sigma = sigmas[i]
            t_prev_sigma = sigmas[i+1]
            t_key_val = int(t_curr_sigma.item() * 1000)

            t_curr_float = t_curr_sigma.item()
            dt = t_prev_sigma.item() - t_curr_float


            # 准备Z^{tar}_t
            fwd_noise = torch.randn_like(latents_src).to(self.device)
            zt_src = (1 - t_curr_float) * latents_src + t_curr_float * fwd_noise
            zt_tar = zt_edit + zt_src - latents_src


            # Q/R 动态调整
            q_scale = lambda_1 * t_curr_float
            r_scale = lambda_2 * (1 - t_curr_float)
            kalman_filter.set_q_scale(q_scale)
            kalman_filter.set_r_scale(r_scale)


            def flow_func(x, _zt_tar, t):
                zt_tar = x + zt_src - latents_src
                do_cfg_full = src_guidance_scale > 1.0 or tar_guidance_scale > 1.0
                if do_cfg_full:
                    batch_latents = torch.cat([zt_src, zt_src, zt_tar, zt_tar], dim=0)
                else:
                    batch_latents = torch.cat([zt_src, zt_tar], dim=0)

                t_expand = (t * 1000).long().expand(batch_latents.shape[0]).to(self.device)

                noise_pred = self.pipe.transformer(
                    hidden_states=batch_latents.to(dtype=self.dtype),
                    timestep=t_expand,
                    encoder_hidden_states=combined_emb,
                    pooled_projections=combined_pool,
                    return_dict=False
                )[0]

                if do_cfg_full:
                    src_u, src_t, tar_u, tar_t = noise_pred.chunk(4)
                    vt_src = src_u + src_guidance_scale * (src_t - src_u)
                    vt_tar = tar_u + tar_guidance_scale * (tar_t - tar_u)
                else:
                    vt_src, vt_tar = noise_pred.chunk(2)
                return vt_tar, vt_src

            # 求Jacobian
            def dvdx_func(x, v_curr, t_curr, _t_prev):
                epsilon = 1e-2
                dx = v_curr * epsilon
                x = x + dx
                batch_latents = torch.cat([x] * 2) if tar_guidance_scale > 1.0 else x
                t_expand = (t_curr * 1000).long().expand(batch_latents.shape[0]).to(self.device)

                noise_pred = self.pipe.transformer(
                    hidden_states=batch_latents.to(dtype=self.dtype),
                    timestep=t_expand,
                    encoder_hidden_states=tar_emb,
                    pooled_projections=tar_pool,
                    return_dict=False
                )[0]
                
                if tar_guidance_scale > 1.0:
                    v_next_u, v_next_t = noise_pred.chunk(2)
                    v_next = v_next_u + tar_guidance_scale * (v_next_t - v_next_u)
                else:
                    v_next = noise_pred
                
                return (v_next - v_curr) / (dx + 1e-6)
                

            # 获取观测
            should_inject = inject_mask[i]
            joint_attn_kwargs = {
                "inverse": False,
                "inject": should_inject,
                "timestep": t_key_val,
                "editing_strategy": "replace_v"
            }
            rf_latent_input = torch.cat([z_rf] * 2) if tar_guidance_scale > 1.0 else z_rf
            
            v_rf = second_order_diffusion_step(
                z=z_rf,
                t_model_input=t_curr_sigma,
                t_physical=t_curr_sigma,
                dt=dt,
                latent_input=rf_latent_input,
                emb=tar_emb,
                pool=tar_pool,
                guidance_scale=tar_guidance_scale,
                joint_attn_kwargs=joint_attn_kwargs
            )
            v_rf_inv = rf_invs_dict[t_key_val].to(self.device)
            z_obs = zt_edit + (v_rf - v_rf_inv) * dt


            # Kalman Update
            if i >= T_start:
                zt_edit, _ = kalman_filter.step(
                    flow_func=flow_func,
                    dydx_func=dvdx_func,
                    x_curr=zt_edit,
                    zt_tar=zt_tar,
                    t_curr=t_curr_sigma,
                    t_prev=t_prev_sigma,
                    z_obs=z_obs
                )
            
            if i > T_start:
                # Alignment
                v_mvg = (z_rf - zt_edit) / t_curr_float
                v_rf = eta * v_rf + (1-eta) * v_mvg.to(self.dtype)
            
            z_rf = z_rf + dt * v_rf
            

        self.clear_memory()
        
        # 可视化
        if save_vis_path is not None:
            vis_latents = [zt_edit, latents_src]
            vis_titles = ["EKF-Edit", "Source"]

            Utils.visualize_latents(self.pipe.vae, vis_latents, vis_titles, source_prompt, target_prompt, save_vis_path)
        
        return Utils.latents_to_img(zt_edit.to(dtype=self.dtype), self.pipe.vae)

    @torch.no_grad()
    def run_ekf_edit_dna(self, 
                    source_image: torch.Tensor,
                    source_prompt: str,
                    target_prompt: str,
                    T_steps: int = 50,
                    T_start: int = 17,
                    eta: float = 0.8,        
                    src_guidance_scale: float = 3.5, 
                    tar_guidance_scale: float = 5.5, 
                    lambda_1: float = 0.1,
                    lambda_2: float = 1.0,
                    save_vis_path: str = "vis_output.png"):

        self.clear_memory()
        print(f"Starting EKF-DNA-Edit: Steps={T_steps}, eta={eta}, Skip={T_start}")

        latents_src = self.encode_image(source_image)
        
        kalman_filter = ExtendedKalmanFilter(shape=latents_src.shape, device=self.device, q_scale=0.01, r_scale=0.05)

        inv_emb, inv_pool = self.get_embeds(source_prompt, 1.0, do_classifier_free_guidance=False)
        src_emb, src_pool = self.get_embeds(source_prompt, src_guidance_scale)
        tar_emb, tar_pool = self.get_embeds(target_prompt, tar_guidance_scale)

        if src_guidance_scale > 1.0 or tar_guidance_scale > 1.0:
            combined_emb = torch.cat([src_emb[:1], src_emb[1:], tar_emb[:1], tar_emb[1:]], dim=0)
            combined_pool = torch.cat([src_pool[:1], src_pool[1:], tar_pool[:1], tar_pool[1:]], dim=0)
        else:
            combined_emb = torch.cat([src_emb, tar_emb], dim=0)
            combined_pool = torch.cat([src_pool, tar_pool], dim=0)

        self.pipe.scheduler.set_timesteps(T_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        sigmas = self.pipe.scheduler.sigmas
        
        print("Running Phase 1: DNA Inversion...")
        skip_steps = T_start
        zt_inv = latents_src.clone()
        noise_inv = torch.randn_like(latents_src)
        
        delta_z_dict = {} 
        v_inv_dict = {}   
        inv_sigmas = torch.cat([sigmas, torch.tensor([0], device=self.device)]).flip(0)
        for i, (t_curr, t_next) in enumerate(zip(tqdm(inv_sigmas[:-1], desc="Inverting"), inv_sigmas[1:])):
            if len(inv_sigmas) - 1 - i == skip_steps:
                break

            t_key_val = int(t_next.item() * 1000)
            
            zt_curr_inv = zt_inv 
            
            # Bridge 公式
            zt_next_inv = (t_next - t_curr) / (1 - t_curr) * (noise_inv - zt_inv) + zt_inv
            
            t_expand = t_next.expand(zt_next_inv.shape[0])
            v_pred_inv = self.pipe.transformer(
                hidden_states=zt_next_inv,
                timestep=t_expand,
                encoder_hidden_states=inv_emb,
                pooled_projections=inv_pool,
                return_dict=False,
            )[0]

            # 计算 Delta Velocity
            v_delta_inv = ((zt_next_inv - zt_curr_inv) / (t_next - t_curr) - v_pred_inv)
            
            # 更新状态
            zt_next_inv = zt_next_inv.to(torch.float32)
            zt_inv = zt_next_inv - v_delta_inv * (t_next - t_curr)
            
            delta_z = v_delta_inv * (t_next - t_curr) 
            
            zt_inv = zt_inv.to(delta_z.dtype)
            noise_inv = noise_inv.to(torch.float32)
            noise_inv -= v_delta_inv * (1 - t_curr)
            noise_inv = noise_inv.to(delta_z.dtype)

            # cache
            delta_z_dict[t_key_val] = delta_z
            v_inv_dict[t_key_val] = v_pred_inv
        

        zt_dna = zt_inv.clone()
        zt_edit = latents_src.clone() 

        for i, t in enumerate(tqdm(timesteps, desc="EKF Loop")):
            if i < T_start:
                continue

            t_curr_sigma = sigmas[i]
            t_prev_sigma = sigmas[i+1]
            t_key_val = int(t_curr_sigma.item() * 1000)

            t_curr_float = t_curr_sigma.item()
            dt = t_prev_sigma.item() - t_curr_float


            # Z^{tar}_t
            fwd_noise = torch.randn_like(latents_src).to(self.device)
            zt_src = (1 - t_curr_float) * latents_src + t_curr_float * fwd_noise
            zt_tar = zt_edit + zt_src - latents_src
            
            # 获取观测
            z_obs = None
            if t_key_val in delta_z_dict:
                
                delta_z = delta_z_dict[t_key_val]
                vt_src_dna = v_inv_dict[t_key_val]
                
                zt_dna_tar = zt_dna + delta_z

                if tar_guidance_scale > 1.0:
                    obs_model_input = torch.cat([zt_dna_tar, zt_dna_tar], dim=0)
                else:
                    obs_model_input = zt_dna_tar

                # 计算v_tar_dna
                t_expand = t.expand(obs_model_input.shape[0])
                noise_pred_dna = self.pipe.transformer(
                    hidden_states=obs_model_input,
                    timestep=t_expand,
                    encoder_hidden_states=tar_emb,
                    pooled_projections=tar_pool,
                    return_dict=False,
                )[0]

                if tar_guidance_scale > 1.0:
                    obs_u, obs_t = noise_pred_dna.chunk(2)
                    v_tar_dna = obs_u + tar_guidance_scale * (obs_t - obs_u)
                else:
                    v_tar_dna = noise_pred_dna
                
                zt_ref = zt_edit.clone().to(torch.float32)
                # 更新zt_ref, zt_ref=Z^{MVG}
                zt_ref = zt_ref + dt * (v_tar_dna - vt_src_dna)
                zt_ref = zt_ref.to(v_tar_dna.dtype)
                
                z_obs = zt_ref.clone()


            # Q/R 动态调整
            q_scale = lambda_1 * t_curr_float
            r_scale = lambda_2 * (1 - t_curr_float)
            kalman_filter.set_q_scale(q_scale)
            kalman_filter.set_r_scale(r_scale)

            def flow_func(x, _zt_tar, t):
                zt_tar = x + zt_src - latents_src
                do_cfg_full = src_guidance_scale > 1.0 or tar_guidance_scale > 1.0
                if do_cfg_full:
                    batch_latents = torch.cat([zt_src, zt_src, zt_tar, zt_tar], dim=0)
                else:
                    batch_latents = torch.cat([zt_src, zt_tar], dim=0)

                t_expand = (t * 1000).long().expand(batch_latents.shape[0]).to(self.device)

                noise_pred = self.pipe.transformer(
                    hidden_states=batch_latents.to(dtype=self.dtype),
                    timestep=t_expand,
                    encoder_hidden_states=combined_emb,
                    pooled_projections=combined_pool,
                    return_dict=False
                )[0]

                if do_cfg_full:
                    src_u, src_t, tar_u, tar_t = noise_pred.chunk(4)
                    vt_src = src_u + src_guidance_scale * (src_t - src_u)
                    vt_tar = tar_u + tar_guidance_scale * (tar_t - tar_u)
                else:
                    vt_src, vt_tar = noise_pred.chunk(2)
                return vt_tar, vt_src
            
            # Jacobian
            def dvdx_func(x, v_curr, t_curr, _t_prev):
                epsilon = 1e-2
                dx = v_curr * epsilon
                x = x + dx
                batch_latents = torch.cat([x] * 2) if tar_guidance_scale > 1.0 else x
                t_expand = (t_curr * 1000).long().expand(batch_latents.shape[0]).to(self.device)

                noise_pred = self.pipe.transformer(
                    hidden_states=batch_latents.to(dtype=self.dtype),
                    timestep=t_expand,
                    encoder_hidden_states=tar_emb,
                    pooled_projections=tar_pool,
                    return_dict=False
                )[0]
                
                if tar_guidance_scale > 1.0:
                    v_next_u, v_next_t = noise_pred.chunk(2)
                    v_next = v_next_u + tar_guidance_scale * (v_next_t - v_next_u)
                else:
                    v_next = noise_pred
                
                return (v_next - v_curr) / (dx + 1e-6)
                

            # EKF update
            zt_edit, _ = kalman_filter.step(
                flow_func=flow_func,
                dydx_func=dvdx_func,
                x_curr=zt_edit,
                zt_tar=zt_tar,
                t_curr=t_curr_sigma,
                t_prev=t_prev_sigma,
                z_obs=z_obs
            )


            if t_key_val in delta_z_dict:
                v_mvg = (zt_dna - zt_edit) / t_curr_float
                v_tar_dna = eta * v_tar_dna + (1-eta) * v_mvg.to(self.dtype)
                zt_dna = zt_dna + dt * v_tar_dna
            

        self.clear_memory()
        

        # 可视化
        if save_vis_path is not None:
            vis_latents = [zt_edit, latents_src]
            vis_titles = ["EKF-Edit", "Source"]

            Utils.visualize_latents(self.pipe.vae, vis_latents, vis_titles, source_prompt, target_prompt, save_vis_path)

        return Utils.latents_to_img(zt_edit.to(dtype=self.dtype), self.pipe.vae)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKFEdit demo")
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--model', type=str, default='/data1/2025/zcf/stabilityai/stable-diffusion-3.5-medium',
        help='Base T2I model ID'
    )
    parser.add_argument(
        '--img_path', type=str, default='assets/brown_owl.png',
        help='Path to the source image'
    )
    parser.add_argument(
        '--source_prompt', type=str, default='A small, brown owl standing on a patch of grass.',
        help='Source prompt describing the source image'
    )
    parser.add_argument(
        '--target_prompt', type=str, default='A small glass sculpture of a brown owl standing on a patch of grass.',
        help='Target prompt for editing the image'
    )
    parser.add_argument(
        '--seed', type=int, default=2025,
        help='Random seed'
    )
    parser.add_argument(
        '--output', type=str, default='edited_image.png',
        help='Path to save the edited image'
    )
    parser.add_argument(
        '--tar_guidance_scale', type=float, default=7.5,
        help='Target guidance scale'
    )
    parser.add_argument(
        '--src_guidance_scale', type=float, default=3.5,
        help='Source guidance scale'
    )
    parser.add_argument(
        '--eta', type=float, default=0.8,
        help='Alignment eta value'
    )
    parser.add_argument(
        '--lambda_1', type=float, default=0.1,
        help='EKF Q scale factor'
    )
    parser.add_argument(
        '--lambda_2', type=float, default=1.0,
        help='EKF R scale factor'
    )
    parser.add_argument(
        '--T_start', type=int, default=17,
        help='Start timestep'
    )
    parser.add_argument(
        '--T_steps', type=int, default=50,
        help='Total timesteps'
    )
    parser.add_argument(
        '--method', type=str, default='ekf_edit_dna', choices=['ekf_edit_dna', 'ekf_edit_rf'],
        help='Editing method to use, either "ekf_edit_dna" or "ekf_edit_rf"'
    )

    args = parser.parse_args()

    Utils.set_seed(args.seed)

    try:
        editor = EditPipeline(args.model, device=args.device, dtype=torch.bfloat16)

        src_img_tensor = Utils.load_image(args.img_path, 512)
        src_prompt = args.source_prompt
        tar_prompt = args.target_prompt

        if args.method == 'ekf_edit_rf':
            result = editor.run_ekf_edit_rf(
                source_image=src_img_tensor,
                source_prompt=src_prompt,
                target_prompt=tar_prompt,
                T_steps=args.T_steps,
                T_start=args.T_start,
                src_guidance_scale=args.src_guidance_scale,
                tar_guidance_scale=args.tar_guidance_scale,
                lambda_1=args.lambda_1,
                lambda_2=args.lambda_2,
                eta=args.eta,
                save_vis_path=f"{args.output.split('.')[0]}_vis.png"
            )
        else:
            result = editor.run_ekf_edit_dna(
                source_image=src_img_tensor,
                source_prompt=src_prompt,
                target_prompt=tar_prompt,
                T_steps=args.T_steps,
                T_start=args.T_start,
                src_guidance_scale=args.src_guidance_scale,
                tar_guidance_scale=args.tar_guidance_scale,
                lambda_1=args.lambda_1,
                lambda_2=args.lambda_2,
                eta=args.eta,
                save_vis_path=f"{args.output.split('.')[0]}_vis.png"
            )
        result.save(args.output)


    except OSError as e:
        print(f"Error loading model: {e}")

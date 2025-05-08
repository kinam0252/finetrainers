import torch
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Video generation validation script")
    parser.add_argument(
        "--model_type",
        type=str,
        default="wan",
        required=False,
        help="Type of model to use for validation"
    )
    parser.add_argument(
        "--lora_weight_path",
        type=str,
        required=False,
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="/data/kinamkim/finetrainers/laboratory/dataset/480x480",
        help="Path to dataset to use for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Width of the video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the video"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=30,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--apply_target_noise_only",
        type=str,
        default=None,
        # required=True,
        help="Apply noise only to target frame"
    )
    return parser.parse_args()


args = parse_args()

@torch.no_grad()
def process_video(pipe, video_path, dtype, generator, height, width, apply_target_noise_only):
    if apply_target_noise_only == None:
        return None
    from diffusers.utils import load_video
    from diffusers.pipelines.wan.pipeline_wan_video2video import retrieve_latents
    from diffusers.utils.torch_utils import randn_tensor
    video = load_video(video_path)
    video = pipe.video_processor.preprocess_video(video, height=height, width=width)
    video = video.to("cuda", dtype=torch.float32)
    

    video_latents = retrieve_latents(pipe.vae.encode(video))
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(pipe.device, dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        pipe.device, dtype
    )

    init_latents = (video_latents - latents_mean) * latents_std

    init_latents = init_latents.to(pipe.device)

    noise = randn_tensor(init_latents.shape, generator=generator, device=pipe.device, dtype=dtype)
    if apply_target_noise_only == "back":
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
        init_latents[:, :, :-1] = noise[:, :, :-1]
    elif apply_target_noise_only == "front" or apply_target_noise_only == "front-none":
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
        init_latents[:, :, 1:] = noise[:, :, 1:]
    elif apply_target_noise_only == "front-long" or apply_target_noise_only == "front-long-none":
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
        init_latents[:, :, 6:] = noise[:, :, 6:]
    elif apply_target_noise_only == "front-4-none":
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
        init_latents[:, :, 4:] = noise[:, :, 4:]
    elif apply_target_noise_only == "front-4-noise-none":
        timesteps = pipe.scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
        scheduler = pipe.scheduler
        n_timesteps = timesteps.shape[0]
        #t_100 = timesteps[0]
        t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
        t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
        t_75 = timesteps[int(n_timesteps * (1 - 0.75))]
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
        #init_latents[:, :, 0] = scheduler.add_noise(init_latents[:, :, 0], noise[:, :, 0], torch.tensor([t_100]))
        init_latents[:, :, 1] = scheduler.add_noise(init_latents[:, :, 1], noise[:, :, 1], torch.tensor([t_25]))
        init_latents[:, :, 2] = scheduler.add_noise(init_latents[:, :, 2], noise[:, :, 2], torch.tensor([t_50]))
        init_latents[:, :, 3] = scheduler.add_noise(init_latents[:, :, 3], noise[:, :, 3], torch.tensor([t_75]))
        init_latents[:, :, 4:] = noise[:, :, 4:]
    elif apply_target_noise_only == "front-4-sdedit-none":
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
    
    else:
        raise ValueError(f"apply_target_noise_only must be either 'back' or 'front', but got {apply_target_noise_only}")
    init_latents = init_latents.to(pipe.device)
    return init_latents

@torch.no_grad()
def retrieve_video(pipe, init_latents):
    latents = init_latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    return video



if __name__ == "__main__":
    with torch.no_grad():
        if args.model_type == "ltxvideo":
            from diffusers import LTXPipeline
            from diffusers.utils import export_to_video
            pipe = LTXPipeline.from_pretrained(
                "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
            ).to("cuda")
            pipe.load_lora_weights(args.lora_weight_path, adapter_name="ltxv-lora")
            pipe.set_adapters(["ltxv-lora"], [0.75])

        elif args.model_type == "cogvideo":
            from pipeline import CogVideoXPipeline
            from finetrainers.models.cogvideox.model import CogVideoXTransformer3DModel
            from diffusers.utils import export_to_video, load_video
            model_id = "/home/nas4_user/kinamkim/checkpoint/cogvideox-5b"
            pipe = CogVideoXPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            ).to("cuda")
            pipe.transformer.to("cpu")
            pipe.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer", torch_dtype=torch.bfloat16
            ).to("cuda")
            pipe.enable_model_cpu_offload(device="cuda")
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            if args.lora_weight_path:
                pipe.load_lora_weights(args.lora_weight_path, adapter_name="cogvideox-lora")
                pipe.set_adapters(["cogvideox-lora"], [0.75])
            else:
                args.lora_weight_path = "output/base/dummy.safetensors"
        elif args.model_type == "wan":
            import torch
            from diffusers import AutoencoderKLWan
            from pipeline import WanPipeline
            from diffusers.utils import export_to_video
            from finetrainers.models.wan.model import WanTransformer3DModel
            model_id = "/data/kinamkim/checkpoint/Wan2.1-T2V-14B-Diffusers"
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            pipe.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
            pipe.to("cuda")

            if args.lora_weight_path:
                pipe.load_lora_weights(args.lora_weight_path, adapter_name="wan-lora")
                pipe.set_adapters(["wan-lora"], [0.75])
            else:
                args.lora_weight_path = "output/base/dummy.safetensors"

        # Create validation_videos directory in the same folder as lora_weight_path
        lora_dir = args.lora_weight_path
        savedir = os.path.join(lora_dir, "validation_videos")
        if args.apply_target_noise_only:
            savedir = os.path.join(savedir, args.apply_target_noise_only)
        else:
            savedir = os.path.join(savedir, "None")
        dataset_name = "/".join(args.dataset_dir.split("/")[-2:])
        # savedir = os.path.join(savedir, dataset_name)
        savedir = f"laboratory/outputs/{args.apply_target_noise_only}"
        os.makedirs(savedir, exist_ok=True)
        
        video_dir = os.path.join(args.dataset_dir, "videos")
        prompt_path = os.path.join(args.dataset_dir, "prompt.txt")
        with open(prompt_path, "r") as f:
            prompts = f.readlines()
        
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        neg_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        for i, prompt in enumerate(prompts[:args.num_videos]):
            predicted_num_elements = 10 * 1024**3 // 4
            null_prompt_tensor = torch.empty(predicted_num_elements, dtype=torch.float32, device='cuda')
            pipe.to("cuda")
            print(f"Generating video {i+1}: {prompt[:100]}...")
            video_path = os.path.join(video_dir, f"{i+1}.mp4")
            print(f"video_path: {video_path}")
            if args.apply_target_noise_only:
                init_latents = process_video(pipe, 
                                            video_path, 
                                            torch.bfloat16, 
                                            generator, 
                                            args.height, 
                                            args.width,
                                            args.apply_target_noise_only)
            else:
                init_latents = None
            #video = retrieve_video(pipe, init_latents)
            #export_to_video(video, "test.mp4")
            #assert False, "stop here"
            # front-long 구현해야됨됨
            # pipe.enable_model_cpu_offload()
            if args.apply_target_noise_only == "front-4-sdedit-none":
                input_latents = None
            else:
                input_latents = init_latents
                
            video = pipe(prompt, 
                         negative_prompt=neg_prompt,
                         generator=generator, 
                         width=args.width, 
                         height=args.height, 
                         num_frames=49,
                         guidance_scale=5,
                         latents=input_latents,
                         apply_target_noise_only=args.apply_target_noise_only,
                         init_latents=init_latents).frames[0]
            export_to_video(video, os.path.join(savedir, f"output_{i}.mp4"))
            print(f"saved at {os.path.join(savedir, f'output_{i}.mp4')}")
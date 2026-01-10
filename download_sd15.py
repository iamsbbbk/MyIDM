import os
from diffusers import StableDiffusionPipeline
import torch

def download_sd15(save_dir="./sd15"):
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading Stable Diffusion v1.5 to:", os.path.abspath(save_dir))

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,   # 可选：不需要生成图片时可以关掉
    )

    pipe.save_pretrained(save_dir)

    print("✅ Download finished!")
    print("You should see model_index.json in:", save_dir)


if __name__ == "__main__":
    download_sd15("./sd15")

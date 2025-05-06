import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_decode
from torchvision.io import write_video

import torch
import torchvision
from torchvision import transforms
import pathlib
import os
import random

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ckpt_path = "models/ltxv-2b-0.9.6-dev-04-25.safetensors"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_FOLDER = "/home/twtomtwcc00/VSPW"
OUT_FOLDER = "VSPW_latent"
OUT_LATENT_FOLDER = os.path.join(OUT_FOLDER, "origin")
os.makedirs(OUT_LATENT_FOLDER, exist_ok=True)


vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
vae.to(device, dtype=torch.bfloat16)
with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
    resizer = transforms.Resize([864, 1536])
    for vid_folder in (pathlib.Path(IN_FOLDER) / "data").iterdir():
        if f"{vid_folder.name}.pt" in pathlib.Path(OUT_LATENT_FOLDER).iterdir():
            print(vid_folder.name, "already exists in output folder, skipping")
            continue
        images = []
        flist = sorted(i for i in (vid_folder/"origin").iterdir() if not i.name.startswith("._"))
        if len(flist) < 41:
            print(vid_folder.name, "do not have enough frames.")
            continue
        if (pathlib.Path(OUT_FOLDER) / f"{vid_folder.name}.STARTFRAME").exists():
            with open(pathlib.Path(OUT_FOLDER) / f"{vid_folder.name}.STARTFRAME") as f:
                start_frame = int(f.read())
        else:
            start_frame = random.randint(0, len(flist)-41)
            with open(pathlib.Path(OUT_FOLDER) / f"{vid_folder.name}.STARTFRAME", "w") as f:
                f.write(str(start_frame))
        flist = flist[start_frame:start_frame + 41]

        for frame_img in flist:
            image = torchvision.io.decode_image(frame_img, mode="RGB") / 127.5 - 1.0
            image = image.bfloat16()
            image = resizer(image)
            images.append(image)

        images = torch.stack(images, dim=0).permute(1, 0, 2, 3).unsqueeze(0).bfloat16().cuda()
        print(images.shape)
        latent = vae_encode(images, vae, vae_per_channel_normalize=False)
        out_images = vae_decode(latent, vae, vae_per_channel_normalize=False, timestep=0)

        images = images.permute(0, 2, 3, 4, 1).squeeze(0)
        out_images = out_images.permute(0, 2, 3, 4, 1).squeeze(0)

        images = (images + 1.0) * 127.5
        out_images = (out_images + 1.0) * 127.5
        images = torch.clamp(images, 0, 255)
        out_images = torch.clamp(out_images, 0, 255)
        write_video("decode_test_in.mp4", images, fps=24, video_codec="h264")
        write_video("decode_test_out.mp4", out_images, fps=24, video_codec="h264")
        break

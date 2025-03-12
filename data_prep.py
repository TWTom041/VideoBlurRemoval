import torch
import torchvision
from torchvision import transforms
import pathlib
import os
import random

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode

ckpt_path = "models/ltx-video-2b-v0.9.5.safetensors"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_FOLDER = "/home/twtomtwcc00/VSPW"
OUT_FOLDER = "VSPW_latent"
OUT_LATENT_FOLDER = os.path.join(OUT_FOLDER, "origin")
os.makedirs(OUT_LATENT_FOLDER, exist_ok=True)


vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
vae.to(device, dtype=torch.bfloat16)
with torch.autocast("cuda", torch.bfloat16):
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
        flist = flist[start_frame:]

        for frame_img in flist:
            image = torchvision.io.decode_image(frame_img, mode="RGB")
            image = image
            image = resizer(image)
            images.append(image)
        images = torch.stack(images, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        latent = vae_encode(images.to(device), vae, vae_per_channel_normalize=True)
        torch.save(latent, pathlib.Path(OUT_LATENT_FOLDER) / f"{vid_folder.name}.pt")
        print(vid_folder.name, "done successfully.")




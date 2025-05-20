import os
import random
from collections import Counter
import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import read_image, write_png, ImageReadMode
import torchvision.transforms as T
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "/home/twtomtwcc00/VideoBlurRemoval/models/ltxv-2b-0.9.6-dev-04-25.safetensors"
vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
vae.to(device, dtype=torch.bfloat16)
resizer = T.Resize([864, 1536])
blur_transform = T.GaussianBlur(kernel_size=9, sigma=4)

DATADIR = "/home/twtomtwcc00/VSPW/data/"
OUT_DIR = "/home/twtomtwcc00/VideoBlurRemoval/VSPW_latent"
OUT_LATENT_DIR = os.path.join(OUT_DIR, "blurred")
os.makedirs(OUT_LATENT_DIR, exist_ok=True)

for vid in os.listdir(DATADIR):
    origin_dir = os.path.join(DATADIR, vid, "origin")
    mask_dir = os.path.join(DATADIR, vid, "mask")
    if not os.path.exists(origin_dir) or not os.path.exists(mask_dir):
        continue
    print("starting vid:", vid)

    origin = {}
    mask = {}

    # Read origin images as RGB tensors and send them to the GPU.
    for filename in os.listdir(origin_dir):
        if filename.startswith("._"):
            continue
        path = os.path.join(origin_dir, filename)
        # read_image returns a [C, H, W] tensor in RGB with dtype uint8.
        img = read_image(path).to(device)
        origin[filename] = img

    # Read mask images in grayscale ([1, H, W]) and send to GPU.
    for filename in os.listdir(mask_dir):
        if filename.startswith("._"):
            continue
        path = os.path.join(mask_dir, filename)
        m_img = read_image(path, mode=ImageReadMode.GRAY).to(device)
        mask[filename] = m_img

    # Build a counter of unique pixel values from each mask image.
    pix_counter = Counter()
    for key in origin.keys():
        # Assuming mask filename corresponds to origin filename with .png extension.
        num = key.split(".")[0]
        mask_key = num + ".png"
        unique_vals = torch.unique(mask[mask_key]).tolist()
        for val in unique_vals:
            pix_counter[val] += 1

    # Select a random best_value from the mask pixel values.
    best_value = list(pix_counter.keys())[random.randint(0, len(pix_counter)-1)]
    option = random.randint(0, 2)

    frame_list = []

    if os.path.exists(os.path.join(OUT_DIR, f"{vid}.STARTFRAME")):
        with open(os.path.join(OUT_DIR, f"{vid}.STARTFRAME")) as f:
            start_frame = int(f.read())
    else:
        print("STARTFRAME not found, skipping")
        continue

    for key in sorted(origin.keys())[start_frame:start_frame+41]:
        num = key.split(".")[0]
        origin_img = origin[key].float()  # convert to float for processing
        mask_tensor = (mask[num + ".png"] == best_value).float()
        C, H, W = origin_img.shape
        if option == 0:
            blurred = blur_transform(origin_img)
        elif option == 1:
            noise = torch.randint(0, 255, origin_img.shape, device=device, dtype=torch.float32)
            blurred = noise * (7/16) + origin_img * (9/16)
        elif option == 2:
            blurred = F.interpolate(origin_img.unsqueeze(0), size=(H // 6, W // 6), mode='bilinear', align_corners=False)
            blurred = F.interpolate(blurred, size=(H, W), mode='nearest')
        masked_blur = blurred * mask_tensor + origin_img * (1 - mask_tensor)
        if H != 864 or W != 1536:
            masked_blur = resizer(masked_blur)
        frame_list.append(masked_blur.squeeze(0))
    
    frame_list = torch.stack(frame_list, dim=0).permute(1, 0, 2, 3).unsqueeze(0).bfloat16() / 127.5 - 1.0
    with torch.no_grad():
        latent = vae_encode(frame_list, vae, vae_per_channel_normalize=True)
    torch.save(latent, os.path.join(OUT_LATENT_DIR, f"{vid}.pt"))
    print(vid, "done successfully.")

    # Save the tuple with torch.save into a .pt file.
    # torch.save((origin, mask, option, best_value), f"vid-{vid}.pt")
    print("finished vid:", vid)

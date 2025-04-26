import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
from pathlib import Path, PurePath
from PIL import Image

# Import the VAE encoder function and models from ltx_video
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import retrieve_timesteps

# -----------------------------------------------------------------------------
# Paired Video Dataset
# -----------------------------------------------------------------------------
class PairedVideoDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        """
        Args:
            input_dir (str): Folder containing input videos' latents (e.g. input_vids/<id>.pt).
            target_dir (str): Folder containing target videos' latents (e.g. target_vids/<id>.pt).
            num_frames (int, optional): Maximum number of frames to sample from each video.
                If None, all frames will be used.
        """
        # Assumes target videos have the same filenames as input videos.
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        
        self.input_files = sorted(list(self.input_dir.glob("*.pt")))
        self.target_files = [self.target_dir / f.name for f in self.input_files]

        self.input_data = [self.process_pt_file(f) for f in self.input_files]
        self.target_data = [self.process_pt_file(f) for f in self.target_files]

    def process_pt_file(self, filename):
        tensor = torch.load(filename, map_location=torch.device("cpu"))
        if len(tensor.shape) == 4:
            return tensor
        elif len(tensor.shape) == 5:
            return tensor.squeeze(0)  # tensor has batch dimension
        else:
            raise ValueError(f"The shape of {filename} is stupid")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        with torch.no_grad():
            return self.input_data[idx], self.target_data[idx]

# -----------------------------------------------------------------------------
# Training Script
# -----------------------------------------------------------------------------
def main():
    # Paths to the folders with input and target videos.
    input_vids_dir = "/home/twtomtwcc00/VideoBlurRemoval/VSPW_latent/blurred/"
    target_vids_dir = "/home/twtomtwcc00/VideoBlurRemoval/VSPW_latent/origin/"

    # Video processing hyperparameters.
    target_height = 864   # original video height
    target_width = 1536   # original video width
    num_frames = 41      # or set to a lower number to sample a subset

    # Training hyperparameters.
    batch_size = 1        # adjust based on available GPU memory
    num_epochs = 1000
    learning_rate = 1e-4
    num_timesteps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained VAE.
    # Adjust ckpt_path to point to your pretrained checkpoint directory.
    ckpt_path = "models/ltx-video-2b-v0.9.5.safetensors"

    # Create the paired video dataset.
    dataset = PairedVideoDataset(
        input_dir=input_vids_dir,
        target_dir=target_vids_dir
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    latent_height = target_height // 8
    latent_width = target_width // 8
    latent_frames = (num_frames + 7) // 8
    latent_channels = 128

    transformer = Transformer3DModel(
        in_channels=latent_channels,
        positional_embedding_theta=10000,
        positional_embedding_max_pos=[latent_height, latent_width, num_frames // 8],  # adjust if needed
    )
    transformer.to(device)

    patchifier = SymmetricPatchifier(patch_size=1)

    scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)

    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        for epoch in range(num_epochs):
            for step, (input_latents, target_latents) in enumerate(dataloader):
                input_latents = input_latents.to(device)   # (B, latent_channels, F_latent, H_latent, W_latent)
                target_latents = target_latents.to(device)

                noise = target_latents - input_latents

                # Patchify the noisy target and the noise (prediction target).
                noise_target_patches, _ = patchifier.patchify(noise)
                # Patchify the input latents to use as conditioning.
                input_patches, indices_grid = patchifier.patchify(input_latents)

                # Sample a random timestep (for diffusion conditioning).
                t = torch.randint(0, num_timesteps, (batch_size, 1), device=device, dtype=torch.bfloat16)
                t = t / num_timesteps

                noisy_input_patches = scheduler.add_noise(input_patches, noise_target_patches, t)
                v_target = noisy_input_patches - input_patches
                
                # Forward pass: predict noise conditioned on the input video.
                predicted_noise = transformer(
                    hidden_states=input_patches,
                    indices_grid=indices_grid,
                    timestep=t,
                    attention_mask=None,
                    encoder_attention_mask=None,
                    skip_layer_mask=None,
                    skip_layer_strategy=None,
                    return_dict=False,
                )[0]

                loss = mse_loss(predicted_noise, v_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()

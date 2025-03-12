import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
from pathlib import Path
from PIL import Image

# Import the VAE encoder function and models from ltx_video
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier

# -----------------------------------------------------------------------------
# Paired Video Dataset
# -----------------------------------------------------------------------------
class PairedVideoDataset(Dataset):
    def __init__(self, input_dir, target_dir, vae, target_height, target_width, num_frames=None, sample_rate=1):
        """
        Args:
            input_dir (str): Folder containing input videos (e.g. input_vids/<id>.mp4).
            target_dir (str): Folder containing target videos (e.g. target_vids/<id>.mp4).
            vae: A pretrained VAE used to encode videos into latent representations.
            target_height (int): Height (in pixels) to which each frame will be resized.
            target_width (int): Width (in pixels) to which each frame will be resized.
            num_frames (int, optional): Maximum number of frames to sample from each video.
                If None, all frames will be used.
            sample_rate (int): Use every sample_rate–th frame.
        """
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.input_files = sorted(list(self.input_dir.glob("*.mp4")))
        # Assumes target videos have the same filenames as input videos.
        self.target_files = [self.target_dir / f.name for f in self.input_files]
        self.vae = vae
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames
        self.sample_rate = sample_rate

    def __len__(self):
        return 1
        return len(self.input_files)

    def load_video(self, video_path):
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        frames = []
        for i, frame in enumerate(reader):
            # If num_frames is specified, sample until that many frames are collected.
            if self.num_frames is not None and len(frames) >= self.num_frames:
                break
            # Optionally, skip frames according to the sample rate.
            if i % self.sample_rate != 0:
                continue
            pil_img = Image.fromarray(frame).convert("RGB")
            # Resize to target resolution.
            pil_img = pil_img.resize((self.target_width, self.target_height))
            frame_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float()  # (C, H, W)
            # Normalize to [-1, 1]
            frame_tensor = (frame_tensor / 127.5) - 1.0
            frames.append(frame_tensor)
        reader.close()
        # Stack frames into a tensor of shape (num_frames, C, H, W) then permute to (C, num_frames, H, W)
        video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        return video_tensor

    def __getitem__(self, idx):
        # input_path = self.input_files[idx]
        # target_path = self.target_files[idx]
        # input_video = self.load_video(input_path)   # shape: (3, F, H, W)  # F should be 8n+1
        # target_video = self.load_video(target_path)   # same shape
        with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
            input_video = torch.rand(3, 41, 864, 1536)
            target_video = torch.rand(3, 41, 864, 1536)
            
            # Add batch dimension for VAE encoding: (1, C, F, H, W)
            input_video = input_video.unsqueeze(0)
            target_video = target_video.unsqueeze(0)
            
            # Encode videos into latents using the VAE.
            # The vae_encode function returns a tensor of shape (B, latent_channels, F_latent, H_latent, W_latent)
            input_latents = vae_encode(input_video.to(self.vae.device), self.vae, vae_per_channel_normalize=True)
            target_latents = vae_encode(target_video.to(self.vae.device), self.vae, vae_per_channel_normalize=True)
            
            # Remove the batch dimension.
            input_latents = input_latents.squeeze(0)
            target_latents = target_latents.squeeze(0)    
            
            return input_latents, target_latents

# -----------------------------------------------------------------------------
# Training Script
# -----------------------------------------------------------------------------
def main():
    # Paths to the folders with input and target videos.
    input_vids_dir = "input_vids"
    target_vids_dir = "target_vids"

    # Video processing hyperparameters.
    # (These should match the resolution your VAE expects.)
    target_height = 864   # original height
    target_width = 1536   # original width
    num_frames = 41      # or set to a lower number to sample a subset
    sample_rate = 1       # use every frame

    # Training hyperparameters.
    batch_size = 1        # adjust based on available GPU memory
    num_epochs = 5
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained VAE.
    # Adjust ckpt_path to point to your pretrained checkpoint directory.
    ckpt_path = "models/ltx-video-2b-v0.9.5.safetensors"
    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
    vae.to(device, dtype=torch.bfloat16)

    # Create the paired video dataset.
    dataset = PairedVideoDataset(
        input_dir=input_vids_dir,
        target_dir=target_vids_dir,
        vae=vae,
        target_height=target_height,
        target_width=target_width,
        num_frames=num_frames,
        sample_rate=sample_rate,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Determine latent dimensions.
    # For example, if the VAE downsamples by a factor of 8, then:
    # latent_height = target_height // 8, latent_width = target_width // 8.
    latent_height = target_height // 8
    latent_width = target_width // 8
    latent_frames = (num_frames + 7) // 8
    # Assume the VAE produces 4 latent channels.
    latent_channels = 128

    # Initialize the Transformer3DModel.
    # Make sure to set in_channels to match the latent channels.
    transformer = Transformer3DModel(
        in_channels=latent_channels,
        positional_embedding_theta=10000,
        positional_embedding_max_pos=[latent_height, latent_width, num_frames // 8],  # adjust if needed
    )
    transformer.to(device)

    # Instantiate the patchifier.
    # Here, SymmetricPatchifier with patch_size=1 means no extra spatial patching.
    patchifier = SymmetricPatchifier(patch_size=1)

    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        for epoch in range(num_epochs):
            for step, (input_latents, target_latents) in enumerate(dataloader):
                # Move latents to the device.
                input_latents = input_latents.to(device)   # (B, latent_channels, F_latent, H_latent, W_latent)
                target_latents = target_latents.to(device)

                # In video-to-video diffusion, we condition on the input video latents.
                # Here we add noise to the target latents and train the transformer
                # to predict the added noise.
                noise = torch.randn_like(target_latents)
                noisy_target_latents = target_latents + noise

                # Patchify the noisy target and the noise (prediction target).
                noisy_target_patches, indices_grid = patchifier.patchify(noisy_target_latents)
                noise_target_patches, _ = patchifier.patchify(noise)
                # Patchify the input latents to use as conditioning.
                input_patches, _ = patchifier.patchify(input_latents)

                # Sample a random timestep (for diffusion conditioning).
                t = torch.randint(0, 1000, (batch_size, 1), device=device, dtype=torch.float32)

                # Forward pass: predict noise conditioned on the input video.
                predicted_noise = transformer(
                    hidden_states=noisy_target_patches,
                    indices_grid=indices_grid,
                    encoder_hidden_states=input_patches,
                    timestep=t,
                    attention_mask=None,
                    encoder_attention_mask=None,
                    skip_layer_mask=None,
                    skip_layer_strategy=None,
                    return_dict=False,
                )[0]

                loss = mse_loss(predicted_noise, noise_target_patches)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()

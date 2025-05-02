import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from diffusers.configuration_utils import FrozenDict
import numpy as np
import imageio
from pathlib import Path, PurePath
from PIL import Image
import random

# Import the VAE encoder function and models from ltx_video
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import retrieve_timesteps

torch.serialization.add_safe_globals([FrozenDict])


# -----------------------------------------------------------------------------
# Paired Video Dataset
# -----------------------------------------------------------------------------
class PairedVideoDataset(Dataset):
    def __init__(self, input_files, target_files):
        self.input_data = [self.process_pt_file(f) for f in input_files]
        self.target_data = [self.process_pt_file(f) for f in target_files]


    @classmethod
    def from_folder(cls, input_dir, target_dir):
        """
        Args:
            input_dir (str): Folder containing input videos' latents (e.g. input_vids/<id>.pt).
            target_dir (str): Folder containing target videos' latents (e.g. target_vids/<id>.pt).
            num_frames (int, optional): Maximum number of frames to sample from each video.
                If None, all frames will be used.
        """
        # Assumes target videos have the same filenames as input videos.
        input_dir = Path(input_dir)
        target_dir = Path(target_dir)
        
        input_files = sorted(list(input_dir.glob("*.pt")))
        target_files = [target_dir / f.name for f in input_files]

        return cls(input_files, target_files)

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


class Callback:
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_step_end(self, epoch, step, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_train_end(self, logs=None): pass
        

class EarlyStopping(Callback):
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = np.inf
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss") or logs.get("epoch_val_loss")
        if current_loss is None:
            return
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f"EarlyStopping triggered at epoch {epoch+1}")


class TensorBoardLogger(Callback):
    def __init__(self, log_dir="runs/exp"):
        self.writer = SummaryWriter(log_dir)

    def on_step_end(self, epoch, step, logs=None):
        if logs is None: return
        self.writer.add_scalar("train/step_loss", logs["step_train_loss"],
                               epoch * logs.get("steps_per_epoch", 0) + step)
    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return
        self.writer.add_scalar("val/epoch_loss", logs["epoch_val_loss"], epoch)
        self.writer.add_scalar("train/epoch_loss", logs["epoch_train_loss"], epoch)
    def on_train_end(self, logs=None):
        self.writer.close()


class ModelCheckpoint(Callback):
    def __init__(
        self,
        model,
        optimizer,
        scaler,
        save_dir="checkpoints",
        best_fname="best_model.pt",
        last_fname="last_model.pt",
        monitor="epoch_val_loss",
        mode="min",
        verbose=True,
        log_dir="runs/exp"
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_path = os.path.join(save_dir, best_fname)
        self.last_path = os.path.join(save_dir, last_fname)
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        # Comparison operator
        if mode == "min":
            self._is_improvement = lambda curr, best: curr < best
            self.best_score = self._load_initial_best(log_dir, minimize=True)
        elif mode == "max":
            self._is_improvement = lambda curr, best: curr > best
            self.best_score = self._load_initial_best(log_dir, minimize=False)
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def _load_initial_best(self, log_dir, minimize=True):
        if os.path.isdir(log_dir) and os.listdir(log_dir):
            ea = event_accumulator.EventAccumulator(log_dir)
            ea.Reload()
            if "val/epoch_loss" in ea.scalars.Keys():
                vals = [e.value for e in ea.Scalars("val/epoch_loss")]
                return min(vals) if minimize else max(vals)
        return float("inf") if minimize else -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Save last model + optimizer + scaler
        self._save_checkpoint(self.last_path, epoch)
        if self.verbose:
            print(f"[Epoch {epoch+1}] Saved last checkpoint → {self.last_path}")
        # Save best if improved
        current = logs.get(self.monitor)
        if current is None:
            return
        if self._is_improvement(current, self.best_score):
            old = self.best_score
            self.best_score = current
            self._save_checkpoint(self.best_path, epoch)
            if self.verbose:
                print(f"[Epoch {epoch+1}] {self.monitor} improved {old:.4f} → {current:.4f}. "
                      f"Saved best checkpoint → {self.best_path}")

    def on_train_end(self, logs=None):
        # final save
        self._save_checkpoint(self.last_path, None)
        if self.verbose:
            print(f"Training ended. Final checkpoint saved → {self.last_path}")
    
    def _save_checkpoint(self, path, epoch):
        data = {
            "transformer": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.model.config
        }
        if epoch is not None:
            data["epoch"] = epoch + 1
        torch.save(data, path)


def split_dataset(input_video_dir, target_video_dir, train_split_ratio=0.8):
    all_input_data = [i for i in Path(input_video_dir).glob("*.pt")]
    random.Random(1588).shuffle(all_input_data)
    all_target_data = [Path(target_video_dir) / i.name for i in all_input_data]
    train_split_input = all_input_data[:int(train_split_ratio * len(all_input_data))]
    train_split_target = all_target_data[:int(train_split_ratio * len(all_input_data))]

    test_split_input = all_input_data[int(train_split_ratio * len(all_input_data)):]
    test_split_target = all_target_data[int(train_split_ratio * len(all_input_data)):]

    return (train_split_input, train_split_target), (test_split_input, test_split_target)


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
    batch_size = 16        # adjust based on available GPU memory
    num_epochs = 1000
    learning_rate = 1e-4
    num_timesteps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained VAE.
    # Adjust ckpt_path to point to your pretrained checkpoint directory.
    ckpt_path = "models/ltxv-2b-0.9.6-dev-04-25.safetensors"

    train_split, test_split = split_dataset(input_vids_dir, target_vids_dir)

    train_dataset = PairedVideoDataset(*train_split)
    test_dataset = PairedVideoDataset(*test_split)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    latent_height = target_height // 8
    latent_width = target_width // 8
    latent_frames = (num_frames + 7) // 8
    latent_channels = 128

    if Path("checkpoints/best_model.pt").exists():
        state_dicts = torch.load("checkpoints/best_model.pt")
    else:
        state_dicts = None

    transformer = Transformer3DModel(
            in_channels=latent_channels,
            positional_embedding_theta=10000,
            positional_embedding_max_pos=[latent_height, latent_width, num_frames // 8],  # adjust if needed
        )
    transformer.to(device)

    patchifier = SymmetricPatchifier(patch_size=1)

    scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)

    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)

    scaler = torch.GradScaler("cuda")    

    if Path("checkpoints/best_model.pt").exists():
        state_dicts = torch.load("checkpoints/best_model.pt")
        transformer.load_state_dict(state_dict=state_dicts["transformer"])
        optimizer.load_state_dict(state_dict=state_dicts["optimizer"])
        scaler.load_state_dict(state_dict=state_dicts["scaler"])
   
    mse_loss = nn.MSELoss()

    callbacks = [
        TensorBoardLogger(log_dir="runs/vid_blur_removal"),
        EarlyStopping(patience=20, min_delta=1e-4),
        ModelCheckpoint(
            model=transformer,
            optimizer=optimizer,
            scaler=scaler,
            save_dir="checkpoints",
            best_fname="best_model.pt",
            last_fname="last_model.pt",
            monitor="epoch_val_loss",
            mode="min",
            verbose=True,
            log_dir="runs/vid_blur_removal"
        )
    ]

    global_step = 0

    
    for epoch in range(num_epochs):
        transformer.train()
        train_losses = []
        for step, (input_latents, target_latents) in enumerate(train_dataloader):
            with torch.autocast("cuda", torch.bfloat16):
                input_latents = input_latents.to(device)   # (B, latent_channels, F_latent, H_latent, W_latent)
                target_latents = target_latents.to(device)

                noise = target_latents - input_latents

                # Patchify the noisy target and the noise (prediction target).
                noise_target_patches, _ = patchifier.patchify(noise)
                # Patchify the input latents to use as conditioning.
                input_patches, indices_grid = patchifier.patchify(input_latents)

                # Sample a random timestep (for diffusion conditioning).
                t = torch.randint(0, num_timesteps, (input_latents.shape[0], 1), device=device, dtype=torch.bfloat16)
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

                loss = mse_loss(predicted_noise, noise_target_patches)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_val = loss.item()
                train_losses.append(loss_val)
                global_step += 1

                logs = {
                    "step_train_loss": loss_val,
                    "steps_per_epoch": len(train_dataloader)
                }

                for cb in callbacks:
                    cb.on_step_end(epoch, step, logs)

                if step % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(train_dataloader)}] Loss: {loss.item():.4f}")

        transformer.eval()
        val_losses = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for step, (input_latents, target_latents) in enumerate(test_dataloader):
                input_latents = input_latents.to(device)   # (B, latent_channels, F_latent, H_latent, W_latent)
                target_latents = target_latents.to(device)

                noise = target_latents - input_latents

                # Patchify the noisy target and the noise (prediction target).
                noise_target_patches, _ = patchifier.patchify(noise)
                # Patchify the input latents to use as conditioning.
                input_patches, indices_grid = patchifier.patchify(input_latents)

                # Sample a random timestep (for diffusion conditioning).
                t = torch.randint(0, num_timesteps, (input_latents.shape[0], 1), device=device, dtype=torch.bfloat16)
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

                val_losses.append(mse_loss(predicted_noise, noise_target_patches).item())
                if step % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(test_dataloader)}] Loss: {loss.item():.4f}")

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_loss = sum(train_losses) / len(train_losses)
        epoch_logs = {"epoch_val_loss": avg_val_loss, "epoch_train_loss": avg_train_loss}
        for cb in callbacks:
            cb.on_epoch_end(epoch, epoch_logs)
        # check for early stopping
        if any(getattr(cb, "stop_training", False) for cb in callbacks):
            break
    for cb in callbacks:
        cb.on_train_end()

    print("Training complete.")

if __name__ == "__main__":
    main()

from main2 import split_dataset, PairedVideoDataset
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers.configuration_utils import FrozenDict
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_encode
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import retrieve_timesteps
from ltx_video.models.autoencoders.vae_encode import vae_decode
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from torchvision.io import write_video

torch.serialization.add_safe_globals([FrozenDict])

def main():
    # Video processing hyperparameters.
    target_height = 864   # original video height
    target_width = 1536   # original video width
    num_frames = 41      # or set to a lower number to sample a subset
    num_timesteps = 1000

    latent_height = target_height // 8
    latent_width = target_width // 8
    latent_frames = (num_frames + 7) // 8
    latent_channels = 128

    # Paths to the folders with input and target videos.
    input_vids_dir = "/home/twtomtwcc00/VideoBlurRemoval/VSPW_latent/blurred/"
    target_vids_dir = "/home/twtomtwcc00/VideoBlurRemoval/VSPW_latent/origin/"
    
    train_split, test_split = split_dataset(input_vids_dir, target_vids_dir)
    test_dataset = PairedVideoDataset(*test_split)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    transformer = Transformer3DModel(
            in_channels=latent_channels,
            positional_embedding_theta=10000,
            positional_embedding_max_pos=[latent_height, latent_width, num_frames // 8],  # adjust if needed
        )
    transformer.load_state_dict(torch.load("checkpoints/best_model.pt")["transformer"])
    scheduler = RectifiedFlowScheduler.from_pretrained("models/ltxv-2b-0.9.6-dev-04-25.safetensors")
    
    transformer.eval()
    
    with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
        for input_latents, target_latents in test_dataloader:
            break
        patchifier = SymmetricPatchifier(patch_size=1)
        scale=0.1
        input_patches, indices_grid = patchifier.patchify(input_latents)
        for i in range(10):
            
            predicted_noise = transformer(
                        hidden_states=input_patches,
                        indices_grid=indices_grid,
                        timestep=torch.tensor(i/10, dtype=torch.bfloat16),
                        attention_mask=None,
                        encoder_attention_mask=None,
                        skip_layer_mask=None,
                        skip_layer_strategy=None,
                        return_dict=False,
                    )[0]
            input_patches=input_patches-predicted_noise*scale
        input_latents=patchifier.unpatchify(input_patches)
        vae = CausalVideoAutoencoder.from_pretrained("checkpoints/best_model.pt")
        vae.to("cuda", dtype=torch.bfloat16)
        out_images=vae_decode(input_latents, vae, vae_per_channel_normalize=False, timestep=0)
        out_images = (out_images + 1.0) * 127.5
        out_images = torch.clamp(out_images, 0, 255)
    write_video("decode_test_out.mp4", out_images, fps=24, video_codec="h264")

    
if __name__=="__main__":
    main()

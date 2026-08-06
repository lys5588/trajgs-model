from typing import Any, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import numpy as np
from PIL import Image
import os
from copy import deepcopy
from collections import OrderedDict
import random
from decord import VideoReader, cpu

##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              misc
##################################################

def _count_params(module):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in module.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def _freeze_params(model, frozen_params=None, exclude_params=None):
    """
    Freeze the specified parameters.

    Args:
        model: the model
        frozen_params: list of parameter names to freeze (matched with startswith)
        exclude_params: list of parameter-name prefixes to exclude (these parameters are not frozen)
    """
    if frozen_params is not None:
        exclude_params = exclude_params or []
        for n, p in model.named_parameters():
            # Check whether the parameter is in the exclusion list
            if any(n.startswith(ex) for ex in exclude_params):
                continue
            # Match with startswith to avoid accidental substring matches
            for name in frozen_params:
                if n.startswith(name):
                    p.requires_grad = False
                    break


def _freeze_original_embeddings(model, original_vocab_size, logger=None, strict=False):
    """
    Freeze the embeddings of the original vocabulary; only the newly added token embeddings stay trainable.

    A gradient hook is registered that zeroes the gradients of the original tokens during backprop.

    Args:
        model: the Showo2WithMotion model
        original_vocab_size: the original vocab size (before new tokens were added)
        logger: optional logger
        strict: if True, raise when embedding/lm_head cannot be found or there are no new tokens
    """
    # Handle tied embeddings:
    # in Qwen/HF, lm_head.weight may share the same Parameter as embed_tokens.weight;
    # named_parameters() de-duplicates by default, so only one of the names may be enumerable.
    embed_tokens = None
    lm_head = None

    # Prefer accessing the weights directly via module paths to avoid named_parameters de-duplication
    try:
        embed_tokens = model.showo.model.embed_tokens.weight
    except Exception:
        embed_tokens = None

    try:
        lm_head = model.showo.lm_head.weight
    except Exception:
        lm_head = None

    # Fallback: if module-path access fails, match by parameter name instead
    if embed_tokens is None or lm_head is None:
        for name, param in model.named_parameters():
            if embed_tokens is None and name.endswith('showo.model.embed_tokens.weight'):
                embed_tokens = param
            if lm_head is None and name.endswith('showo.lm_head.weight'):
                lm_head = param

    tracked_params = []
    if embed_tokens is not None:
        tracked_params.append(("embed_tokens", embed_tokens))
    if lm_head is not None and all(lm_head is not p for _, p in tracked_params):
        tracked_params.append(("lm_head", lm_head))

    if len(tracked_params) == 0:
        msg = "Could not find embed_tokens/lm_head parameters; skip embedding freeze hook"
        if strict:
            raise RuntimeError(msg)
        if logger:
            logger.warning(msg)
        return {
            "enabled": False,
            "new_token_count": 0,
            "tracked_params": [],
            "reason": "missing_parameters",
        }

    # Make trainable (even if frozen by _freeze_params in the previous step)
    for _, p in tracked_params:
        p.requires_grad = True

    vocab_size = tracked_params[0][1].shape[0]
    safe_original_vocab_size = min(max(int(original_vocab_size), 0), vocab_size)
    new_token_count = vocab_size - safe_original_vocab_size

    if new_token_count <= 0:
        msg = (
            f"No new tokens found for embedding hook "
            f"(original_vocab_size={safe_original_vocab_size}, vocab_size={vocab_size})"
        )
        if strict:
            raise RuntimeError(msg)
        if logger:
            logger.warning(msg)
        return {
            "enabled": False,
            "new_token_count": int(new_token_count),
            "tracked_params": [name for name, _ in tracked_params],
            "reason": "no_new_tokens",
        }

    if logger:
        logger.info(
            f"Enabling training for new tokens: {new_token_count} tokens "
            f"(indices {safe_original_vocab_size} to {vocab_size-1})"
        )
        logger.info(f"Original vocab size: {safe_original_vocab_size}, New vocab size: {vocab_size}")
        if embed_tokens is not None and lm_head is not None and embed_tokens is lm_head:
            logger.info("Detected tied embeddings: embed_tokens and lm_head share the same Parameter")
    
    # Register a gradient hook that freezes the gradients of the original tokens
    def freeze_original_embed_grad(grad):
        """Keep gradients only for the new tokens; zero the gradients of the original tokens."""
        if grad is None:
            return None
        # Build a mask: 0 for original token rows, 1 for new token rows
        mask = torch.zeros_like(grad)
        mask[safe_original_vocab_size:, :] = 1.0
        return grad * mask
    
    # Register hooks (with tied embeddings, the shared parameter is registered only once)
    for name, p in tracked_params:
        p.register_hook(freeze_original_embed_grad)
        if logger:
            logger.info(f"Registered gradient hook for {name}")

    return {
        "enabled": True,
        "new_token_count": int(new_token_count),
        "tracked_params": [name for name, _ in tracked_params],
        "reason": None,
    }


path_to_llm_name = {
    "Qwen/Qwen2.5-7B-Instruct": 'qwen2_5',
    "Qwen/Qwen2.5-1.5B-Instruct": 'qwen2_5',
    "meta-llama/Llama-3.2-1B-Instruct": 'llama3'
}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denorm(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images

def denorm_vid(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    # B, C, T, H, W --> B, T, C, H, W
    images = images.permute(0, 2, 1, 3, 4).cpu().numpy().astype(np.uint8)
    return images


def get_hyper_params(config, text_tokenizer, showo_token_ids, is_video=False, is_hq=False):
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    max_seq_len = config.dataset.preprocessing.max_seq_length
    num_video_tokens = config.dataset.preprocessing.num_video_tokens
    if is_video:
        max_text_len = max_seq_len - num_video_tokens - 4
        latent_width = config.dataset.preprocessing.video_latent_width
        latent_height = config.dataset.preprocessing.video_latent_height
        num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
        num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
    else:
        if is_hq:
            latent_width = config.dataset.preprocessing.hq_latent_width
            latent_height = config.dataset.preprocessing.hq_latent_height
            num_t2i_image_tokens = config.dataset.preprocessing.num_hq_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            max_seq_len = config.dataset.preprocessing.max_hq_seq_length
            max_text_len = max_seq_len - num_t2i_image_tokens - 4
        else:
            num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            latent_width = config.dataset.preprocessing.latent_width
            latent_height = config.dataset.preprocessing.latent_height
            max_text_len = max_seq_len - num_t2i_image_tokens - 4

    image_latent_dim = config.model.showo.image_latent_dim
    patch_size = config.model.showo.patch_size

    pad_id = text_tokenizer.pad_token_id
    bos_id = showo_token_ids['bos_id']
    eos_id = showo_token_ids['eos_id']
    boi_id = showo_token_ids['boi_id']
    eoi_id = showo_token_ids['eoi_id']
    bov_id = showo_token_ids['bov_id']
    eov_id = showo_token_ids['eov_id']
    img_pad_id = showo_token_ids['img_pad_id']
    vid_pad_id = showo_token_ids['vid_pad_id']

    guidance_scale = config.transport.guidance_scale

    return num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, \
           latent_width, latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, \
           vid_pad_id, guidance_scale


# these save and recover functions are based on our internal packages
# please modified them when necessary
def save_dataloader_state(rank, loader, ckpt_path="./"):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    saved_state = deepcopy(loader.__getstate__())
    torch.save(saved_state, ckpt_path)


def _safe_torch_load(path_or_file, map_location='cpu'):
    """
    Prefer weights_only=True to avoid FutureWarning.
    Fall back to the legacy behavior when needed.
    """
    def _rewind():
        if hasattr(path_or_file, "seek"):
            try:
                path_or_file.seek(0)
            except Exception:
                pass

    try:
        return torch.load(path_or_file, map_location=map_location, weights_only=True)
    except TypeError:
        _rewind()
        return torch.load(path_or_file, map_location=map_location)
    except Exception:
        _rewind()
        return torch.load(path_or_file, map_location=map_location)


def recover_dataloader_state(rank, loader, ckpt_path='./'):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            loader_state_dict = _safe_torch_load(f, map_location='cpu')
            loader.__setstate__(loader_state_dict)
        print(f"rank {rank} loader state dict loaded successfully!")


def save_images_as_grid(pil_images, fn, path, grid_size=(2, 2)):

    os.makedirs(path, exist_ok=True)

    rows, cols = grid_size

    num_images = len(pil_images)
    if num_images > rows * cols:
        raise ValueError(f"Number of images ({num_images}) exceeds grid capacity ({rows * cols}).")

    img_width, img_height = pil_images[0].size

    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))  # white background

    for idx, image in enumerate(pil_images):
        row = idx // cols
        col = idx % cols
        x_offset = col * img_width
        y_offset = row * img_height
        grid_image.paste(image, (x_offset, y_offset))

    grid_image.save(os.path.join(path, f"{fn}.png"))

    return grid_image


def load_state_dict(model_path):
    if model_path.endswith(".bin"):
        state_dict = _safe_torch_load(model_path, map_location='cpu')
    else:
        checkpoint_files = sorted(
            [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.bin')]
        )

        state_dict = OrderedDict()
        for checkpoint_file in checkpoint_files:
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = _safe_torch_load(checkpoint_file, map_location='cpu')
            state_dict.update(checkpoint)

    return state_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_video(video_path, max_frames_num, fps, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 432, 432, 3))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = [Image.fromarray(item) for item in vr.get_batch(frame_idx).asnumpy()]

    return spare_frames, frame_time, video_time


##################################################
#         Motion Visualization Utils
##################################################

def _qinv(q):
    """Inverse of quaternion."""
    assert q.shape[-1] == 4
    q_inv = q.clone()
    q_inv[..., 1:] = -q_inv[..., 1:]
    return q_inv


def _qrot(q, v):
    """Rotate vector by quaternion."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    
    qvec = q[..., 1:]  # [..., 3]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def _recover_root_rot_pos(data):
    """Recover root rotation and position from motion features."""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=22):
    """
    Recover joint positions from RIC features (263-dim or 269-dim for HumanML3D).
    
    Only the first 67 dims (root and ric_data) are used, so both 263-dim and 269-dim inputs are supported.
    
    Args:
        data: [bs, seq_len, nfeats] or [seq_len, nfeats] motion features
              supports 263-dim or 269-dim input
        joints_num: number of joints (22 for HumanML3D)
    Returns:
        positions: [..., seq_len, joints_num, 3] joint positions
    """
    r_rot_quat, r_pos = _recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    positions = _qrot(_qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def plot_skeleton_frame(joints, ax, title='', frame_idx=0):
    """
    Plot a single frame of skeleton with correct orientation.
    
    HumanML3D coordinate system:
    - X: forward/backward
    - Y: height (up)
    - Z: left/right
    
    Args:
        joints: [22, 3] joint positions
        ax: matplotlib 3D axis
        title: plot title
        frame_idx: frame index for display
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # SMPL kinematic chain for 22 joints
    kinematic_chain = [
        [0, 2, 5, 8, 11],       # right leg
        [0, 1, 4, 7, 10],       # left leg
        [0, 3, 6, 9, 12, 15],   # spine + head
        [9, 14, 17, 19, 21],    # right arm
        [9, 13, 16, 18, 20],    # left arm
    ]
    colors = ['red', 'blue', 'black', 'red', 'blue']
    
    ax.clear()
    
    data = joints.copy()
    
    # Compute bounds
    MINS = data.min(axis=0)
    MAXS = data.max(axis=0)
    
    # Center the skeleton at origin (X, Z plane)
    height_offset = MINS[1]
    data[:, 1] -= height_offset
    
    # Center X and Z
    center_x = (MINS[0] + MAXS[0]) / 2
    center_z = (MINS[2] + MAXS[2]) / 2
    data[:, 0] -= center_x
    data[:, 2] -= center_z
    
    # Set axis limits (X-Z is ground plane, Y is up)
    radius = 2.0
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])  
    ax.set_zlim3d([0, radius * 2])
    
    # Plot ground plane (X-Z plane at Y=0)
    xx, yy = np.meshgrid(
        np.linspace(-radius, radius, 2),
        np.linspace(-radius, radius, 2)
    )
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='gray')
    
    # Plot skeleton - Note: in 3D plot, we map (X, Z, Y) to (x, y, z) for proper view
    # X -> x (forward), Z -> y (right), Y -> z (up)
    for chain, color in zip(kinematic_chain, colors):
        ax.plot3D(
            data[chain, 0],   # X -> forward
            data[chain, 2],   # Z -> right  
            data[chain, 1],   # Y -> up
            linewidth=3.0,
            color=color
        )
    
    # Plot joints
    ax.scatter3D(data[:, 0], data[:, 2], data[:, 1], s=20, c='black')
    
    # Set labels
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Z (Right)')
    ax.set_zlabel('Y (Up)')
    
    # View from slightly above and behind
    ax.view_init(elev=20, azim=-45)
    
    if title:
        ax.set_title(f'{title}\nFrame {frame_idx}', fontsize=9)
    else:
        ax.set_title(f'Frame {frame_idx}', fontsize=9)
    
    ax.grid(True, alpha=0.3)


def visualize_motion_to_png(
    motion_features,
    save_dir: str,
    prompt: str = '',
    sample_idx: int = 0,
    mean=None,
    std=None,
    fps: int = 20,
    max_frames: int = None,
):
    """
    Visualize motion and save as PNG sequence.
    
    Args:
        motion_features: [seq_len, 263] normalized motion features (numpy or tensor)
        save_dir: directory to save PNG files
        prompt: text prompt for title
        sample_idx: sample index for naming
        mean: mean for denormalization (numpy array)
        std: std for denormalization (numpy array)
        fps: frames per second (for info)
        max_frames: maximum frames to save (None = save all frames)
        
    Returns:
        saved_frames: list of saved frame paths
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to tensor if needed
    if isinstance(motion_features, np.ndarray):
        motion_features = torch.from_numpy(motion_features).float()
    
    # Denormalize if mean/std provided
    if mean is not None and std is not None:
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
            std = torch.from_numpy(std).float()
        motion_features = motion_features * std + mean
    
    # Recover joint positions
    motion_features = motion_features.unsqueeze(0)  # [1, seq_len, 263]
    joints = recover_from_ric(motion_features, joints_num=22)  # [1, seq_len, 22, 3]
    joints = joints.squeeze(0).numpy()  # [seq_len, 22, 3]
    
    # Determine number of frames to render
    seq_len = joints.shape[0]
    if max_frames is not None:
        seq_len = min(seq_len, max_frames)
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Shorten prompt for title
    short_prompt = prompt[:50] + '...' if len(prompt) > 50 else prompt
    
    # Save each frame
    saved_frames = []
    for frame_idx in range(seq_len):
        plot_skeleton_frame(joints[frame_idx], ax, title=short_prompt, frame_idx=frame_idx)
        
        frame_path = os.path.join(save_dir, f'sample_{sample_idx:02d}_frame_{frame_idx:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        saved_frames.append(frame_path)
    
    plt.close(fig)
    
    return saved_frames


def create_motion_grid(
    all_joints: list,
    prompts: list,
    save_path: str,
    frame_indices: list = None,
):
    """
    Create a grid visualization of multiple motions at selected frames.
    
    Args:
        all_joints: list of [seq_len, 22, 3] joint arrays
        prompts: list of text prompts
        save_path: path to save the grid image
        frame_indices: which frames to show (default: [0, 1/4, 1/2, 3/4] of seq_len)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_samples = len(all_joints)
    
    # Determine frame indices based on motion length if not specified
    if frame_indices is None:
        # Use relative positions: 0%, 25%, 50%, 75%
        max_len = max(len(j) for j in all_joints)
        frame_indices = [0, max_len // 4, max_len // 2, 3 * max_len // 4]
    
    n_frames = len(frame_indices)
    
    fig = plt.figure(figsize=(4 * n_frames, 4 * n_samples))
    
    for sample_idx, (joints, prompt) in enumerate(zip(all_joints, prompts)):
        seq_len = len(joints)
        short_prompt = prompt[:40] + '...' if len(prompt) > 40 else prompt
        
        for frame_plot_idx, frame_idx in enumerate(frame_indices):
            actual_frame = min(frame_idx, seq_len - 1)
            
            ax = fig.add_subplot(
                n_samples, n_frames, 
                sample_idx * n_frames + frame_plot_idx + 1,
                projection='3d'
            )
            plot_skeleton_frame(
                joints[actual_frame], ax, 
                title=short_prompt if frame_plot_idx == 0 else '',
                frame_idx=actual_frame
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')  # Ensure all figures are closed
    
    # Ensure file is fully written to disk before returning
    # This prevents race conditions when wandb tries to read the file
    import time
    for _ in range(10):  # Try up to 10 times with small delays
        try:
            from PIL import Image
            with Image.open(save_path) as img:
                img.verify()  # Verify the image is complete
            break
        except Exception:
            time.sleep(0.1)  # Wait a bit for file to be flushed


def load_motion_stats(data_root: str, logger=None):
    """
    Load mean and std for motion denormalization.
    
    Args:
        data_root: HumanML3D data root directory
        logger: optional logger
        
    Returns:
        mean, std: numpy arrays or (None, None) if not found
    """
    # Try multiple possible locations (HumanML3D uses capital Mean.npy and Std.npy)
    possible_mean_paths = [
        os.path.join(data_root, 'Mean.npy'),           # HumanML3D default
        os.path.join(data_root, 'mean.npy'),           # lowercase version
        'assets/meta/mean.npy',                         # local assets
    ]
    
    possible_std_paths = [
        os.path.join(data_root, 'Std.npy'),            # HumanML3D default
        os.path.join(data_root, 'std.npy'),            # lowercase version
        'assets/meta/std.npy',                          # local assets
    ]
    
    mean, std = None, None
    
    # Find mean file
    for mean_path in possible_mean_paths:
        if os.path.exists(mean_path):
            mean = np.load(mean_path)
            if logger:
                logger.info(f"Loaded motion mean from {mean_path}")
            break
    
    # Find std file
    for std_path in possible_std_paths:
        if os.path.exists(std_path):
            std = np.load(std_path)
            if logger:
                logger.info(f"Loaded motion std from {std_path}")
            break
    
    if mean is None or std is None:
        if logger:
            logger.warning(f"Could not find mean/std files in {data_root}")
    
    return mean, std

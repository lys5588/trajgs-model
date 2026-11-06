import torch


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def mld_collate_with_source(batch):
    """Collate function for combined datasets (e.g. unimotion) that carries
    a per-sample dataset source label and optional image data.

    Expected sample tuples:
    - (motion, length, dataset_source)
    - (motion, length, dataset_source, image, has_video, joints_2d_normalized)
    """
    notnone_batches = [b for b in batch if b is not None]
    # Sort by motion length (descending) so padded batches stay well ordered
    notnone_batches.sort(key=lambda x: x[1], reverse=True)

    # Tuple length >= 5 means image data is included
    has_image_data = len(notnone_batches[0]) >= 5
    # Tuple length >= 6 means precomputed 2D joints are included
    has_joints_data = len(notnone_batches[0]) >= 6

    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "length": [b[1] for b in notnone_batches],
        # Dataset source: 0=humanml3d, 1=motionfix, 2=h36m
        "dataset_source": torch.tensor([b[2] for b in notnone_batches], dtype=torch.long),
    }

    if has_image_data:
        images = []
        has_video_flags = []

        # Infer the image size from the first valid image
        img_h, img_w = 256, 192  # default HRNet input size
        for b in notnone_batches:
            if b[3] is not None:
                img_h, img_w = b[3].shape[:2]
                break

        for b in notnone_batches:
            img = b[3]  # image: [H, W, 3] uint8 numpy array or None
            has_vid = b[4]  # has_video: bool

            if img is not None:
                # [H, W, 3] uint8 -> [1, H, W, 3] float32 in [0, 1]
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # [1, H, W, 3]
            else:
                # Zero placeholder (fixed to 1 frame, independent of motion length)
                img_tensor = torch.zeros(1, img_h, img_w, 3)

            images.append(img_tensor)
            has_video_flags.append(has_vid)

        # Stack images: [B, 1, H, W, 3]
        # (stack instead of collate_tensors to avoid padding the time dimension)
        adapted_batch["video_rgb"] = torch.stack(images, dim=0)
        adapted_batch["has_video"] = torch.tensor(has_video_flags, dtype=torch.bool)

    if has_joints_data:
        joints_2d_list = []
        num_joints = 22

        for b in notnone_batches:
            joints_2d = b[5]  # joints_2d_normalized: (22, 2) float32 or None

            if joints_2d is not None:
                # Add a time dimension: (22, 2) -> (1, 22, 2)
                joints_tensor = torch.from_numpy(joints_2d).float().unsqueeze(0)
            else:
                joints_tensor = torch.zeros(1, num_joints, 2)

            joints_2d_list.append(joints_tensor)

        # Stack joints: [B, 1, 22, 2]
        adapted_batch["joints_2d_normalized"] = torch.stack(joints_2d_list, dim=0)

    return adapted_batch

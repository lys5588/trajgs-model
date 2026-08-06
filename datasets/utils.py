import copy
import math
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode


def image_transform(image, resolution=256, normalize=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                    y0_centercrop=False):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    if y0_centercrop:
        width, height = image.size
        left = (width - resolution) / 2
        top = 0
        right = (width + resolution) / 2
        bottom = resolution
        image = image.crop((left, top, right, bottom))
    else:
        image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=mean, std=std, inplace=True)(image)
    return image


def to_tensor_and_normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=mean, std=std, inplace=True)(image)
    return image


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '').replace("This image showcases ", '').replace("This suggests ", '').replace(
        "In the photo, we see ", '').replace("This is ", '').replace("This image is ", '').replace(
        "In the photo, we have ", '').replace("The photo features ", '').replace("The photo depicts ", '').replace(
        "The photo appears to be ", '')

    return caption


# At this time, we do not model the text in image-text pairs for t2i
def format_sequence_gen_qwen2_5(text_tokens, system_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                                num_image_tokens, max_seq_len, system_token_len):
    if system_token_len == 0:
        modality_positions = torch.tensor([[len(text_tokens) + 1 + 1, num_image_tokens]])
        # text_labels = [bos_id] + [-100] * len(text_tokens) + [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        # text_labels = [bos_id] + text_tokens + [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        text_labels = [-100] + [-100] * len(text_tokens) + [-100] + [-100] * num_image_tokens + [-100] + [-100]
        text_tokens = [bos_id] + text_tokens + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]
    else:
        # Note: this branch is not exercised by the released configs.
        modality_positions = torch.tensor([[1 + system_token_len + len(text_tokens) + 1 + 1, num_image_tokens]])
        text_labels = [bos_id] + [-100] * len(system_tokens[0] + system_tokens[1] + text_tokens) + [eos_id] + \
                      [-100] * len(system_tokens[2]) + \
                      [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        text_tokens = [bos_id] + system_tokens[0] + system_tokens[1] + text_tokens + [eos_id] + system_tokens[2] + \
                      [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask

def format_sequence_und(text_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                        num_image_tokens, max_seq_len):
    modality_positions = torch.tensor([[1 + 1, num_image_tokens]])

    text_labels = [bos_id] + [boi_id] + [-100] * num_image_tokens + [eoi_id] + \
                  text_tokens + [eos_id]

    text_tokens = [bos_id] + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + \
                  text_tokens + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def format_interleaved_sequence(image_list, text_token_list, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                                num_image_tokens, max_seq_len, max_num_images, system_tokens=None, system_token_len=0):
    """
    # generation
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    # eg. 0        1-9           10          11-15        16         17
    # understanding
    # [bos_id, im_start, image_tokens, im_end, text_tokens, eos_id, pad_id]
    # eg. 0        1            2-6           7           8-16       17
    """

    text_tokens = []
    text_labels = []
    modality_positions = []

    cur_len = 1 + system_token_len # bos token
    for txt_token, image in zip(text_token_list, image_list):
        if txt_token is not None:
            text_tokens.extend(txt_token)
            text_labels.extend(copy.deepcopy(txt_token))
            cur_len += len(txt_token)

        if image is not None:
            text_tokens.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
            text_labels.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
            # +1 for one <|img_start|> token
            modality_positions.append((cur_len + 1, num_image_tokens))
            cur_len = cur_len + 1 + num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>

    if system_token_len == 0:
        text_labels = [bos_id] + text_labels + [eos_id]
        text_tokens = [bos_id] + text_tokens + [eos_id]
    else:
        # Note: this branch is not exercised by the released configs.
        text_labels = [bos_id] + [-100] * system_token_len + text_labels + [eos_id]
        text_tokens = [bos_id] + system_tokens[0] + system_tokens[1] + system_tokens[2] + text_tokens + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    if len(modality_positions) < max_num_images:
        modality_positions += [(0, 0) for _ in range(max_num_images - len(modality_positions))]

    modality_positions = torch.tensor(modality_positions)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def resize_crop(image, image_height, image_width):
    aspect_ratio = image_width / image_height
    if isinstance(image, torch.Tensor) and image.ndim == 4:
        frame_height, frame_width = image[0].size(1), image[0].size(2)
        original_size_as_tuple = torch.tensor([frame_height, frame_width])
        image_aspect_ratio = frame_width / frame_height
        if image_aspect_ratio >= aspect_ratio:
            image_resize_h = image_height
            image_resize_w = int(round(image_height * (frame_width / frame_height)))
            crop_top_coord = 0
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            image_resize_w = image_width
            image_resize_h = int(round(image_width * (frame_height / frame_width)))
            crop_top_coord = random.randint(0, image_resize_h - image_height)
            crop_left_coord = 0
        image = TVF.resize(image, size=[image_resize_h, image_resize_w],
                           interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TVF.crop(image, crop_top_coord, crop_left_coord, image_height,
                         image_width)
    else:
        frame_height, frame_width = image.size(1), image.size(2)
        image_aspect_ratio = frame_width / frame_height
        original_size_as_tuple = torch.tensor([frame_height, frame_width])
        if image_aspect_ratio >= aspect_ratio:
            image_resize_h = image_height
            image_resize_w = int(round(image_height * (frame_width / frame_height)))
            crop_top_coord = 0
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            image_resize_w = image_width
            image_resize_h = int(round(image_width * (frame_height / frame_width)))
            crop_top_coord = random.randint(0, image_resize_h - image_height)
            crop_left_coord = 0
        image = TVF.resize(image, size=[image_resize_h, image_resize_w],
                           interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TVF.crop(image, crop_top_coord, crop_left_coord, image_height,
                         image_width)
    crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
    return image, original_size_as_tuple, crop_coords_top_left


# ==================== Motion Sequence Formatting ====================

def format_sequence_motion_gen(text_tokens, bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                                num_motion_tokens, max_seq_len):
    """
    Format sequence for Motion Generation (Text-to-Motion).
    
    Sequence structure: [BOS] text [BOM][MOT_PAD*N][EOM][EOS][PAD...]
    
    Args:
        text_tokens: list of text token ids
        bos_id: begin of sequence token id
        eos_id: end of sequence token id
        bom_id: begin of motion token id
        eom_id: end of motion token id
        pad_id: padding token id
        mot_pad_id: motion padding token id
        num_motion_tokens: number of motion tokens (latent_len)
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss (-100 for masked)
        modality_positions: [1, 2] tensor of (offset, length) for motion
        text_mask: [max_seq_len] mask for text tokens
        motion_mask: [max_seq_len] mask for motion tokens
    """
    # Build sequence: [BOS] text [BOM][MOT_PAD*N][EOM][EOS]
    modality_positions = torch.tensor([[len(text_tokens) + 1 + 1, num_motion_tokens]])
    
    # Labels: mask everything except the structure tokens we want to predict
    # We don't predict motion tokens, they are generated via diffusion
    text_labels = [-100] + [-100] * len(text_tokens) + [-100] + [-100] * num_motion_tokens + [-100] + [-100]
    text_tokens = [bos_id] + text_tokens + [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + [eos_id]
    
    # Pad to max_seq_len
    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)
    
    # Create masks
    text_mask = torch.where(
        (text_tokens != mot_pad_id) & (text_tokens != pad_id),
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    motion_mask = torch.where(
        text_tokens == mot_pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    
    return text_tokens, text_labels, modality_positions, text_mask, motion_mask


def format_sequence_motion_und(text_tokens, bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                                num_motion_tokens, max_seq_len):
    """
    Format sequence for Motion Understanding (Motion-to-Text).
    
    Sequence structure: [BOS][BOM][MOT_PAD*N][EOM] text [EOS][PAD...]
    
    Args:
        text_tokens: list of text token ids
        bos_id: begin of sequence token id
        eos_id: end of sequence token id
        bom_id: begin of motion token id
        eom_id: end of motion token id
        pad_id: padding token id
        mot_pad_id: motion padding token id
        num_motion_tokens: number of motion tokens (latent_len)
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss
        modality_positions: [1, 2] tensor of (offset, length) for motion
        text_mask: [max_seq_len] mask for text tokens
        motion_mask: [max_seq_len] mask for motion tokens
    """
    # Motion position: right after [BOS][BOM]
    modality_positions = torch.tensor([[1 + 1, num_motion_tokens]])
    
    # Build sequence: [BOS][BOM][MOT_PAD*N][EOM] text [EOS]
    # Labels: predict text tokens after motion
    text_labels = [bos_id] + [bom_id] + [-100] * num_motion_tokens + [eom_id] + \
                  text_tokens + [eos_id]
    text_tokens = [bos_id] + [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + \
                  text_tokens + [eos_id]
    
    # Pad to max_seq_len
    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)
    
    # Create masks
    text_mask = torch.where(
        (text_tokens != mot_pad_id) & (text_tokens != pad_id),
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    motion_mask = torch.where(
        text_tokens == mot_pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    
    return text_tokens, text_labels, modality_positions, text_mask, motion_mask


def format_sequence_motion_und_instruct(prompt_tokens, caption_tokens, bos_id, eos_id, bom_id, eom_id, 
                                         pad_id, mot_pad_id, num_motion_tokens, max_seq_len):
    """
    Format sequence for Motion Understanding (Motion-to-Text) with instruction.
    
    Used for the M2T task in the Stage 2b instruction-tuning phase. Consistent with MotionGPT3:
    - Input: motion tokens + instruction prompt
    - Output: caption
    
    Sequence structure: [BOS][BOM][MOT_PAD*N][EOM] prompt caption [EOS][PAD...]
    Labels: [-100]*bos [-100]*bom [-100]*motion [-100]*eom [-100]*prompt caption [EOS][-100]*pad
    
    No loss is computed on the prompt part; only the caption part is predicted.
    
    Args:
        prompt_tokens: list of prompt token ids (e.g., "Describe the motion using plain English.")
        caption_tokens: list of caption token ids (target output)
        bos_id: begin of sequence token id
        eos_id: end of sequence token id
        bom_id: begin of motion token id
        eom_id: end of motion token id
        pad_id: padding token id
        mot_pad_id: motion padding token id
        num_motion_tokens: number of motion tokens (latent_len)
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss
        modality_positions: [1, 2] tensor of (offset, length) for motion
        text_mask: [max_seq_len] mask for text tokens
        motion_mask: [max_seq_len] mask for motion tokens
    """
    # Motion position: right after [BOS][BOM] -> offset = 2
    modality_positions = torch.tensor([[1 + 1, num_motion_tokens]])
    
    # Build sequence: [BOS][BOM][MOT_PAD*N][EOM] prompt caption [EOS]
    # Labels: the prompt part is -100; only the caption and eos are predicted
    text_tokens_list = [bos_id] + [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + \
                       prompt_tokens + caption_tokens + [eos_id]
    text_labels_list = [-100] + [-100] + [-100] * num_motion_tokens + [-100] + \
                       [-100] * len(prompt_tokens) + caption_tokens + [eos_id]
    
    # Pad to max_seq_len
    text_labels_list = text_labels_list + [-100] * (max_seq_len - len(text_labels_list))
    text_tokens_list = text_tokens_list + [pad_id] * (max_seq_len - len(text_tokens_list))
    
    # Truncate if too long
    text_tokens_list = text_tokens_list[:max_seq_len]
    text_labels_list = text_labels_list[:max_seq_len]
    
    text_tokens = torch.tensor(text_tokens_list)
    text_labels = torch.tensor(text_labels_list)
    
    # Create masks
    text_mask = torch.where(
        (text_tokens != mot_pad_id) & (text_tokens != pad_id),
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    motion_mask = torch.where(
        text_tokens == mot_pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    
    return text_tokens, text_labels, modality_positions, text_mask, motion_mask


def format_sequence_motion_pred(text_tokens, bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                                 num_source_tokens, num_target_tokens, max_seq_len):
    """
    Format sequence for Motion Prediction (Motion-to-Motion).
    
    Given the first part of a motion, predict the rest.
    Source motion (s1) is the conditioning input; target motion (s2) is the generation target.
    
    Sequence structure: [BOS] "Predict motion:" [BOM][MOT_PAD*N1][EOM] [BOM][MOT_PAD*N2][EOM][EOS][PAD...]
                        |<-- text prefix -->| |<-- source motion -->| |<-- target motion -->|
    
    Args:
        text_tokens: list of text token ids (e.g., "Predict motion:")
        bos_id: begin of sequence token id
        eos_id: end of sequence token id
        bom_id: begin of motion token id
        eom_id: end of motion token id
        pad_id: padding token id
        mot_pad_id: motion padding token id
        num_source_tokens: number of source motion tokens (input condition)
        num_target_tokens: number of target motion tokens (generation target)
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss (-100 for masked)
        source_modality_positions: [1, 2] tensor of (offset, length) for source motion
        target_modality_positions: [1, 2] tensor of (offset, length) for target motion
        text_mask: [max_seq_len] mask for text tokens
        source_motion_mask: [max_seq_len] mask for source motion tokens
        target_motion_mask: [max_seq_len] mask for target motion tokens
    """
    # Calculate positions
    # [BOS] text [BOM][SOURCE_MOT*N1][EOM] [BOM][TARGET_MOT*N2][EOM][EOS]
    source_offset = len(text_tokens) + 1 + 1  # after [BOS] + text + [BOM]
    target_offset = source_offset + num_source_tokens + 1 + 1  # after source motion + [EOM] + [BOM]
    
    source_modality_positions = torch.tensor([[source_offset, num_source_tokens]])
    target_modality_positions = torch.tensor([[target_offset, num_target_tokens]])
    
    # Build sequence
    # Source motion: conditioning input, no loss computed
    # Target motion: generation target, diffusion loss computed (not NTP loss)
    text_labels = [-100] + [-100] * len(text_tokens) + \
                  [-100] + [-100] * num_source_tokens + [-100] + \
                  [-100] + [-100] * num_target_tokens + [-100] + [-100]
    text_tokens_seq = [bos_id] + text_tokens + \
                      [bom_id] + [mot_pad_id] * num_source_tokens + [eom_id] + \
                      [bom_id] + [mot_pad_id] * num_target_tokens + [eom_id] + [eos_id]
    
    # Pad to max_seq_len
    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens_seq = text_tokens_seq + [pad_id] * (max_seq_len - len(text_tokens_seq))
    text_tokens_tensor = torch.tensor(text_tokens_seq)
    text_labels_tensor = torch.tensor(text_labels)
    
    # Create masks
    text_mask = torch.where(
        (text_tokens_tensor != mot_pad_id) & (text_tokens_tensor != pad_id),
        torch.ones_like(text_tokens_tensor),
        torch.zeros_like(text_tokens_tensor)
    )
    
    # Source motion mask: num_source_tokens positions starting at source_offset
    source_motion_mask = torch.zeros_like(text_tokens_tensor)
    source_motion_mask[source_offset:source_offset + num_source_tokens] = 1
    
    # Target motion mask: num_target_tokens positions starting at target_offset
    target_motion_mask = torch.zeros_like(text_tokens_tensor)
    target_motion_mask[target_offset:target_offset + num_target_tokens] = 1
    
    return (text_tokens_tensor, text_labels_tensor, 
            source_modality_positions, target_modality_positions,
            text_mask, source_motion_mask, target_motion_mask)


def format_sequence_motion_edit(text_tokens, bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                                num_source_tokens, num_target_tokens, max_seq_len):
    """
    Format sequence for Motion Edit (Motion-to-Motion with text instruction).
    
    Given a source motion and an edit instruction, generate the edited motion.
    Source motion (s1) is the conditioning input; target motion (s2) is the generation target.
    
    Sequence structure: [BOS] <edit_instruction> [BOM][MOT_PAD*N1][EOM] [BOM][MOT_PAD*N2][EOM][EOS][PAD...]
                        |<-- edit text -------->| |<-- source motion -->| |<-- target motion -->|
    
    The only difference from Motion Prediction is that the text part is an edit instruction instead of a simple "Predict motion:"
    
    Args:
        text_tokens: list of text token ids (edit instruction)
        bos_id: begin of sequence token id
        eos_id: end of sequence token id
        bom_id: begin of motion token id
        eom_id: end of motion token id
        pad_id: padding token id
        mot_pad_id: motion padding token id
        num_source_tokens: number of source motion tokens (input condition)
        num_target_tokens: number of target motion tokens (generation target)
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss (-100 for masked)
        source_modality_positions: [1, 2] tensor of (offset, length) for source motion
        target_modality_positions: [1, 2] tensor of (offset, length) for target motion
        text_mask: [max_seq_len] mask for text tokens
        source_motion_mask: [max_seq_len] mask for source motion tokens
        target_motion_mask: [max_seq_len] mask for target motion tokens
    """
    # Reuse the format_sequence_motion_pred logic since the sequence structure is identical
    return format_sequence_motion_pred(
        text_tokens, bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
        num_source_tokens, num_target_tokens, max_seq_len
    )


def format_sequence_text_to_text(input_tokens, output_tokens, bos_id, eos_id, pad_id, max_seq_len):
    """
    Format sequence for Text-to-Text (T2T) tasks.
    
    Used for the Stage 3 T2T tasks, e.g.:
    - caption-to-framelen: predict the motion frame count from the caption
    - framelen-to-caption: generate a plausible motion description from the frame count
    - second-to-caption: generate a description from the duration in seconds
    - random-caption (n2t): generate a motion description unconditionally

    This is a text-only task without the motion modality.
    
    Sequence structure: [BOS] input output [EOS][PAD...]
    Labels: [-100] [-100]*input_len output [EOS][-100]*pad
    
    Only the output part is predicted; no loss is computed on the input part.
    
    Args:
        input_tokens: list of input text token ids (prompt/query)
        output_tokens: list of output text token ids (target response)
        bos_id: begin of sequence token id
        eos_id: end of sequence token id
        pad_id: padding token id
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss
        text_mask: [max_seq_len] mask for valid tokens (not padding)
    """
    # Build sequence: [BOS] input output [EOS]
    # Labels: the input part is -100; only the output and eos are predicted
    text_tokens_list = [bos_id] + input_tokens + output_tokens + [eos_id]
    text_labels_list = [-100] + [-100] * len(input_tokens) + output_tokens + [eos_id]
    
    # Pad to max_seq_len
    text_labels_list = text_labels_list + [-100] * (max_seq_len - len(text_labels_list))
    text_tokens_list = text_tokens_list + [pad_id] * (max_seq_len - len(text_tokens_list))
    
    # Truncate if too long
    text_tokens_list = text_tokens_list[:max_seq_len]
    text_labels_list = text_labels_list[:max_seq_len]
    
    text_tokens = torch.tensor(text_tokens_list)
    text_labels = torch.tensor(text_labels_list)
    
    # Create mask for valid tokens (not padding)
    text_mask = torch.where(
        text_tokens != pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    
    return text_tokens, text_labels, text_mask


# ==================== Image + Motion Sequence Formatting (Stage 3) ====================

def format_sequence_img2pose(instr_tokens, bos_id, eos_id, boi_id, eoi_id, bom_id, eom_id,
                              pad_id, img_pad_id, mot_pad_id,
                              num_image_tokens, num_motion_tokens, max_seq_len):
    """
    Format sequence for Image-to-Pose (img2pose) task.
    
    The image is the conditioning input (no time embed); the motion is the generation target (with time embed).
    
    Sequence structure:
        [BOS] instruction [BOI][IMG_PAD*N_img][EOI] [BOM][MOT_PAD*N_mot][EOM][EOS][PAD...]
        
    Labels: all -100 (img2pose only has the diffusion loss, no NTP loss)
    
    Args:
        instr_tokens: list of instruction token ids
        bos_id, eos_id: sequence boundary tokens
        boi_id, eoi_id: image boundary tokens
        bom_id, eom_id: motion boundary tokens
        pad_id: padding token id
        img_pad_id: image padding token id
        mot_pad_id: motion padding token id
        num_image_tokens: number of image tokens (PatchEmbed output)
        num_motion_tokens: number of motion tokens (time_embed + latent)
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss (-100 for all)
        image_modality_positions: [1, 2] (offset, length) for image
        motion_modality_positions: [1, 2] (offset, length) for motion
        text_mask: [max_seq_len] text/special token mask
        image_mask: [max_seq_len] image token mask
        motion_mask: [max_seq_len] motion token mask
    """
    # Calculate positions
    # [BOS] instr [BOI][IMG*N][EOI] [BOM][MOT*M][EOM][EOS]
    img_offset = len(instr_tokens) + 1 + 1  # after [BOS] + instr + [BOI]
    mot_offset = img_offset + num_image_tokens + 1 + 1  # after img + [EOI] + [BOM]
    
    image_modality_positions = torch.tensor([[img_offset, num_image_tokens]])
    motion_modality_positions = torch.tensor([[mot_offset, num_motion_tokens]])
    
    # Build sequence
    text_tokens_seq = ([bos_id] + instr_tokens + 
                       [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] +
                       [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + [eos_id])
    
    # Labels: all -100 (the motion is trained via the diffusion loss, no NTP loss needed)
    text_labels_seq = [-100] * len(text_tokens_seq)
    
    # Pad to max_seq_len
    text_labels_seq = text_labels_seq + [-100] * (max_seq_len - len(text_labels_seq))
    text_tokens_seq = text_tokens_seq + [pad_id] * (max_seq_len - len(text_tokens_seq))
    
    # Truncate
    text_tokens_seq = text_tokens_seq[:max_seq_len]
    text_labels_seq = text_labels_seq[:max_seq_len]
    
    text_tokens = torch.tensor(text_tokens_seq)
    text_labels = torch.tensor(text_labels_seq)
    
    # Create masks
    text_mask = torch.where(
        (text_tokens != img_pad_id) & (text_tokens != mot_pad_id) & (text_tokens != pad_id),
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    image_mask = torch.where(
        text_tokens == img_pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    motion_mask = torch.where(
        text_tokens == mot_pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    
    return (text_tokens, text_labels,
            image_modality_positions, motion_modality_positions,
            text_mask, image_mask, motion_mask)


def format_sequence_img2text(instr_tokens, caption_tokens, bos_id, eos_id, boi_id, eoi_id,
                              pad_id, img_pad_id, num_image_tokens, max_seq_len):
    """
    Format sequence for Image-to-Text (img2text) task.
    
    The image is the conditioning input; the text description is the generation target (NTP loss).
    
    Sequence structure:
        [BOS][BOI][IMG_PAD*N][EOI] instruction caption [EOS][PAD...]
        
    Labels: [-100]*bos [-100]*boi [-100]*img [-100]*eoi [-100]*instr caption [EOS][-100]*pad
    
    Only the caption part is predicted.
    
    Args:
        instr_tokens: list of instruction token ids
        caption_tokens: list of caption token ids (target output)
        bos_id, eos_id: sequence boundary tokens
        boi_id, eoi_id: image boundary tokens
        pad_id: padding token id
        img_pad_id: image padding token id
        num_image_tokens: number of image tokens
        max_seq_len: maximum sequence length
        
    Returns:
        text_tokens: [max_seq_len] padded token sequence
        text_labels: [max_seq_len] labels for NTP loss
        image_modality_positions: [1, 2] (offset, length) for image
        text_mask: [max_seq_len] text/special token mask
        image_mask: [max_seq_len] image token mask
    """
    # Image position: after [BOS][BOI] -> offset = 2
    image_modality_positions = torch.tensor([[1 + 1, num_image_tokens]])
    
    # Build sequence: [BOS][BOI][IMG*N][EOI] instruction caption [EOS]
    text_tokens_list = ([bos_id] + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] +
                        instr_tokens + caption_tokens + [eos_id])
    
    # Labels: only the caption and eos are predicted
    text_labels_list = ([-100] + [-100] + [-100] * num_image_tokens + [-100] +
                        [-100] * len(instr_tokens) + caption_tokens + [eos_id])
    
    # Pad to max_seq_len
    text_labels_list = text_labels_list + [-100] * (max_seq_len - len(text_labels_list))
    text_tokens_list = text_tokens_list + [pad_id] * (max_seq_len - len(text_tokens_list))
    
    # Truncate
    text_tokens_list = text_tokens_list[:max_seq_len]
    text_labels_list = text_labels_list[:max_seq_len]
    
    text_tokens = torch.tensor(text_tokens_list)
    text_labels = torch.tensor(text_labels_list)
    
    # Create masks
    text_mask = torch.where(
        (text_tokens != img_pad_id) & (text_tokens != pad_id),
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    image_mask = torch.where(
        text_tokens == img_pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    
    return text_tokens, text_labels, image_modality_positions, text_mask, image_mask


def format_sequence_mgie(
    instr_tokens,
    bos_id,
    eos_id,
    boi_id,
    eoi_id,
    bom_id,
    eom_id,
    pad_id,
    img_pad_id,
    mot_pad_id,
    num_source_image_tokens,
    num_ref_motion_tokens,
    num_target_image_tokens,
    max_seq_len,
):
    """
    Format sequence for Motion-Guided Image Editing (MGIE).

    Sequence structure:
        [BOS] instruction
        [BOI][SRC_IMG_PAD * N_src][EOI]
        [BOM][REF_MOT_PAD * N_mot][EOM]
        [BOI][TGT_IMG_PAD * N_tgt][EOI]
        [EOS][PAD...]

    Loss:
      - NTP labels are all -100 (no text generation objective).
      - Flow loss is applied only on target image token positions.
    """
    src_img_offset = len(instr_tokens) + 1 + 1
    ref_mot_offset = src_img_offset + num_source_image_tokens + 1 + 1
    tgt_img_offset = ref_mot_offset + num_ref_motion_tokens + 1 + 1

    source_image_modality_positions = torch.tensor([[src_img_offset, num_source_image_tokens]])
    ref_motion_modality_positions = torch.tensor([[ref_mot_offset, num_ref_motion_tokens]])
    target_image_modality_positions = torch.tensor([[tgt_img_offset, num_target_image_tokens]])

    text_tokens_seq = (
        [bos_id] + instr_tokens +
        [boi_id] + [img_pad_id] * num_source_image_tokens + [eoi_id] +
        [bom_id] + [mot_pad_id] * num_ref_motion_tokens + [eom_id] +
        [boi_id] + [img_pad_id] * num_target_image_tokens + [eoi_id] +
        [eos_id]
    )
    text_labels_seq = [-100] * len(text_tokens_seq)

    text_labels_seq = text_labels_seq + [-100] * (max_seq_len - len(text_labels_seq))
    text_tokens_seq = text_tokens_seq + [pad_id] * (max_seq_len - len(text_tokens_seq))
    text_tokens_seq = text_tokens_seq[:max_seq_len]
    text_labels_seq = text_labels_seq[:max_seq_len]

    text_tokens = torch.tensor(text_tokens_seq)
    text_labels = torch.tensor(text_labels_seq)

    text_mask = torch.where(
        (text_tokens != img_pad_id) & (text_tokens != mot_pad_id) & (text_tokens != pad_id),
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens)
    )
    source_image_mask = torch.zeros_like(text_tokens)
    source_image_mask[src_img_offset:src_img_offset + num_source_image_tokens] = 1

    ref_motion_mask = torch.zeros_like(text_tokens)
    ref_motion_mask[ref_mot_offset:ref_mot_offset + num_ref_motion_tokens] = 1

    target_image_mask = torch.zeros_like(text_tokens)
    target_image_mask[tgt_img_offset:tgt_img_offset + num_target_image_tokens] = 1

    return (
        text_tokens,
        text_labels,
        source_image_modality_positions,
        ref_motion_modality_positions,
        target_image_modality_positions,
        text_mask,
        source_image_mask,
        ref_motion_mask,
        target_image_mask,
    )


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

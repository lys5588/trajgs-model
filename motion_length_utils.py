from typing import List, Sequence


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be > 0, got {divisor}")
    value = int(value)
    if value <= 0:
        return 0
    return (value + divisor - 1) // divisor


def latent_len_from_motion_length(
    length: int,
    downsample_rate: int,
    min_latent_len: int = 1,
) -> int:
    length = int(length)
    if length <= 0:
        return 0
    latent_len = ceil_div(length, downsample_rate)
    return max(int(min_latent_len), latent_len)


def latent_lens_from_motion_lengths(
    lengths: Sequence[int],
    downsample_rate: int,
    min_latent_len: int = 1,
) -> List[int]:
    return [
        latent_len_from_motion_length(length, downsample_rate, min_latent_len=min_latent_len)
        for length in lengths
    ]


def motion_token_slots(
    length: int,
    downsample_rate: int,
    add_time_embeds: bool,
    min_latent_len: int = 1,
) -> int:
    latent_len = latent_len_from_motion_length(
        length,
        downsample_rate,
        min_latent_len=min_latent_len,
    )
    if latent_len <= 0:
        return 0
    return latent_len + int(bool(add_time_embeds))

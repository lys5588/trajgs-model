"""
UniMotion Dataset for VAE training.
Jointly trains on the HumanML3D, MotionFix, and H36M datasets.

Data format:
- HumanML3D: each file is a motion sequence of shape (seq_len, 269);
  IDs look like "000001".
- MotionFix: each file is a motion sequence of shape (seq_len, 269);
  IDs look like "source/000362".
- H36M: single-frame data; all frames are merged into one large file
  of shape (n_frames, 269).
"""
import codecs as cs
import random
from os.path import join as pjoin

import numpy as np
from rich.progress import track
from torch.utils import data


class UniMotionDataset(data.Dataset):
    """
    UniMotion Dataset for VAE training.

    Supports two data types:
    - sequence: HumanML3D/MotionFix-style motion sequences
    - single_frame: H36M-style single-frame data

    Each sample is a tuple: (motion, length, dataset_source)
    """

    def __init__(
        self,
        mean,
        std,
        split_file,
        max_motion_length,
        min_motion_length,
        unit_length,
        motion_dir,
        data_type="sequence",  # "sequence" or "single_frame"
        fps=20,
        tiny=False,
        debug=False,
        progress_bar=True,
        dataset_source=0,  # dataset source label: 0=humanml3d, 1=motionfix, 2=h36m
        **kwargs,
    ):
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.fps = fps
        self.dataset_source = dataset_source
        self.data_type = data_type
        self.motion_dir = motion_dir

        if data_type == "single_frame":
            self._load_single_frame_data(split_file, motion_dir, tiny, debug, progress_bar)
        else:
            self._load_sequence_data(split_file, motion_dir, tiny, debug, progress_bar)

        self.mean = mean
        # Replace zeros in std with 1 to avoid division by zero
        self.std = np.where(std == 0, 1.0, std)

    def _load_sequence_data(self, split_file, motion_dir, tiny, debug, progress_bar):
        """Load sequence data (HumanML3D/MotionFix format)."""
        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading UniMotion {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []

        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion_path = pjoin(motion_dir, name + ".npy")
                motion = np.load(motion_path)

                # Filter motions that are too short or too long
                if len(motion) < self.min_motion_length or len(motion) >= self.max_motion_length:
                    bad_count += 1
                    continue

                data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                }
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1

            except Exception:
                bad_count += 1

        print(f"Loaded {count} motions, skipped {bad_count} motions")

        if len(new_name_list) == 0:
            raise ValueError("No valid motions found!")

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def _load_single_frame_data(self, split_file, motion_dir, tiny, debug, progress_bar):
        """Load single-frame data (H36M format)."""
        # The split file contains the name of the data file
        with cs.open(split_file, "r") as f:
            data_file_name = f.readline().strip()

        data_path = pjoin(motion_dir, data_file_name + ".npy")
        all_data = np.load(data_path)
        print(f"Loaded H36M data from {data_path}, shape: {all_data.shape}")

        if tiny or debug:
            maxdata = 1000 if tiny else 10000
            all_data = all_data[:maxdata]

        # Each frame is treated as one sample
        self.all_frames = all_data
        self.nfeats = all_data.shape[1]

        # Single-frame data does not need name_list or data_dict
        self.data_dict = None
        self.name_list = list(range(len(all_data)))
        self.length_arr = np.ones(len(all_data))  # every sample has length 1
        self.pointer = 0

        print(f"Loaded {len(all_data)} single frames")

    def reset_max_len(self, length):
        if self.data_type == "single_frame":
            return  # not applicable to single-frame data
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        if self.data_type == "single_frame":
            return len(self.all_frames)
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        if self.data_type == "single_frame":
            return self._get_single_frame_item(item)
        return self._get_sequence_item(item)

    def _get_sequence_item(self, item):
        """Fetch a sequence sample."""
        # The DDP DistributedSampler may pad indices; take the modulo to stay in range
        max_idx = len(self.name_list) - self.pointer
        item = item % max_idx if max_idx > 0 else 0
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data["motion"], data["length"]

        # Crop the motions to times of unit_length
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return motion, m_length, self.dataset_source

    def _get_single_frame_item(self, item):
        """Fetch a single-frame sample."""
        # The DDP DistributedSampler may pad indices; take the modulo to stay in range
        item = item % len(self.all_frames)
        motion = self.all_frames[item]  # (269,)
        motion = motion[np.newaxis, :]  # (1, 269)

        # Z Normalization
        motion = (motion - self.mean) / self.std

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return motion, 1, self.dataset_source

"""
UniMotion Dataset with image support for VAE training.
Supports image-motion pairs for H36M while remaining compatible with
motion-only datasets.

Data format:
- HumanML3D: each file is a motion sequence of shape (seq_len, 269); no images.
- MotionFix: each file is a motion sequence of shape (seq_len, 269); no images.
- H36M: single-frame data with paired images; all frames are merged into
  one large file of shape (n_frames, 269).
"""
import codecs as cs
import os
import random
from os.path import join as pjoin

import cv2
import numpy as np
from rich.progress import track
from torch.utils import data


class UniMotionDatasetWithImage(data.Dataset):
    """
    UniMotion Dataset with image support.

    Supports two data types:
    - sequence: HumanML3D/MotionFix-style motion sequences (no images)
    - single_frame: H36M-style single-frame data (with images)

    Each sample is a tuple:
    (motion, length, dataset_source, image, has_video, joints_2d_normalized)
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
        # Image parameters (H36M only)
        image_root=None,  # root directory of H36M images
        image_paths_file=None,  # index file of H36M image paths (.npy)
        img_size=(256, 192),  # image size (H, W)
        load_images=True,  # if False, images are returned as None
        # Precomputed 2D joint parameters (H36M only)
        joints2d_file=None,  # precomputed 2D joints file (.pkl)
        use_precomputed_joints=False,  # whether to use precomputed joints
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

        self.image_root = image_root
        self.image_paths_file = image_paths_file
        self.img_h, self.img_w = img_size
        self.load_images = load_images
        self.has_images = (data_type == "single_frame" and 
                           image_root is not None and 
                           image_paths_file is not None and
                           load_images)

        self.joints2d_file = joints2d_file
        self.use_precomputed_joints = use_precomputed_joints
        self.joints2d_array = None

        # Explicitly disable joints2d when images are not loaded
        if not self.has_images:
            self.use_precomputed_joints = False
            self.joints2d_file = None

        if data_type == "single_frame":
            self._load_single_frame_data(split_file, motion_dir, tiny, debug, progress_bar)
        else:
            self._load_sequence_data(split_file, motion_dir, tiny, debug, progress_bar)

        self.mean = mean
        # Replace zeros in std with 1 to avoid division by zero
        self.std = np.where(std == 0, 1.0, std)

        # Precomputed 2D joints are only loaded for H36M single-frame data with images enabled
        if data_type == "single_frame" and self.has_images and use_precomputed_joints and joints2d_file is not None:
            self._load_precomputed_joints(split_file)
        elif data_type == "single_frame" and not self.has_images and joints2d_file is not None:
            print(f"  Skipping joints2d loading (images disabled, joints2d not needed)")

    def _load_sequence_data(self, split_file, motion_dir, tiny, debug, progress_bar):
        """Load sequence data (HumanML3D/MotionFix format); no images."""
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
        """Load single-frame data (H36M format), with images."""
        # The split file contains the name of the data file
        with cs.open(split_file, "r") as f:
            data_file_name = f.readline().strip()

        data_path = pjoin(motion_dir, data_file_name + ".npy")
        all_data = np.load(data_path)
        print(f"Loaded H36M data from {data_path}, shape: {all_data.shape}")

        if self.has_images:
            # Infer the image paths file from the motion file name so that
            # train/test splits pick up their matching index files.
            if self.image_paths_file is not None:
                image_paths_dir = os.path.dirname(self.image_paths_file)
                expected_image_file = pjoin(image_paths_dir, data_file_name + ".npy")

                if os.path.exists(expected_image_file):
                    image_paths_file = expected_image_file
                else:
                    # Fall back to the configured file (may not match the split)
                    image_paths_file = self.image_paths_file
                    print(f"Warning: Expected image paths file not found: {expected_image_file}")
                    print(f"  Using configured file: {image_paths_file}")
            else:
                image_paths_file = self.image_paths_file
            
            image_paths_array = np.load(image_paths_file, allow_pickle=True)
            print(f"Loaded H36M image paths from {image_paths_file}, shape: {image_paths_array.shape}")
            
            # Ensure motion and image paths have the same length
            if len(all_data) != len(image_paths_array):
                print(f"Warning: motion frames ({len(all_data)}) != image paths ({len(image_paths_array)})")
                print(f"  This usually means train/test image paths mismatch.")
                print(f"  Taking minimum length: {min(len(all_data), len(image_paths_array))}")
                min_len = min(len(all_data), len(image_paths_array))
                all_data = all_data[:min_len]
                image_paths_array = image_paths_array[:min_len]
            
            self.image_paths_array = image_paths_array
        else:
            self.image_paths_array = None

        if tiny or debug:
            maxdata = 1000 if tiny else 10000
            all_data = all_data[:maxdata]
            if self.image_paths_array is not None:
                self.image_paths_array = self.image_paths_array[:maxdata]

        # Each frame is treated as one sample
        self.all_frames = all_data
        self.nfeats = all_data.shape[1]

        # Single-frame data does not need name_list or data_dict
        self.data_dict = None
        self.name_list = list(range(len(all_data)))
        self.length_arr = np.ones(len(all_data))  # every sample has length 1
        self.pointer = 0

        print(f"Loaded {len(all_data)} single frames (with_images={self.has_images})")

    def _load_precomputed_joints(self, split_file):
        """
        Load precomputed 2D joint coordinates from a pkl file
        (generated by generate_2d_skeleton_from_gt.py).
        """
        import joblib
        
        try:
            joints2d_data = joblib.load(self.joints2d_file)
            split_name = 'train' if 'train' in split_file else 'test'
            
            if split_name not in joints2d_data:
                print(f"Warning: Split '{split_name}' not found in joints2d file, available: {joints2d_data.keys()}")
                # Fall back to 'train' joints when 'test' is unavailable
                if split_name == 'test' and 'train' in joints2d_data:
                    print(f"  Using 'train' split for '{split_name}' (test joints not precomputed yet)")
                    split_name = 'train'
                else:
                    self.joints2d_array = None
                    return
            
            self.joints2d_array = joints2d_data[split_name]  # (N, 22, 2)
            print(f"Loaded precomputed 2D joints ({split_name}): {self.joints2d_array.shape}")
            
            if len(self.joints2d_array) != len(self.all_frames):
                print(f"Warning: joints2d count ({len(self.joints2d_array)}) != motion count ({len(self.all_frames)})")
                print(f"  Will use min of the two.")
                min_count = min(len(self.joints2d_array), len(self.all_frames))
                self.joints2d_array = self.joints2d_array[:min_count]
                self.all_frames = self.all_frames[:min_count]
            
        except Exception as e:
            print(f"Error loading precomputed joints from {self.joints2d_file}: {e}")
            self.joints2d_array = None
    
    def _normalize_joints_to_grid_sample(self, joints_2d):
        """
        Normalize pixel coordinates to [-1, 1] for grid_sample.

        Args:
            joints_2d: (22, 2) pixel coordinates

        Returns:
            joints_normalized: (22, 2) normalized coordinates in [-1, 1]
        """
        joints_normalized = joints_2d.copy()
        joints_normalized[:, 0] = 2 * joints_2d[:, 0] / self.img_w - 1  # X
        joints_normalized[:, 1] = 2 * joints_2d[:, 1] / self.img_h - 1  # Y
        joints_normalized = np.clip(joints_normalized, -1, 1)
        return joints_normalized
    
    def _load_image(self, image_path):
        """Load an image and resize it if needed.

        Returns:
            image: [H, W, 3] uint8 numpy array
        """
        try:
            full_path = pjoin(self.image_root, image_path)
            
            img = cv2.imread(full_path)
            if img is None:
                raise ValueError(f"Failed to load image: {full_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img.shape[:2] != (self.img_h, self.img_w):
                img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
            
            return img  # [H, W, 3] uint8
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

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
        """Fetch a sequence sample (no image)."""
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

        return (
            motion,
            m_length,
            self.dataset_source,
            None,  # sequence data has no image
            False,  # has_video = False
            None,  # sequence data has no precomputed joints
        )

    def _get_single_frame_item(self, item):
        """Fetch a single-frame sample (with image if available)."""
        # The DDP DistributedSampler may pad indices; take the modulo to stay in range
        item = item % len(self.all_frames)
        motion = self.all_frames[item]  # (269,)
        motion = motion[np.newaxis, :]  # (1, 269)

        # Z Normalization
        motion = (motion - self.mean) / self.std

        image = None
        has_video = False
        if self.has_images:
            image_path = self.image_paths_array[item]
            image = self._load_image(image_path)  # [H, W, 3] uint8
            has_video = True
        
        joints_2d_normalized = None
        if self.joints2d_array is not None and item < len(self.joints2d_array):
            joints_2d = self.joints2d_array[item]  # (22, 2) pixel coordinates
            # Normalize to [-1, 1] for grid_sample
            joints_2d_normalized = self._normalize_joints_to_grid_sample(joints_2d)
            # joints_2d_normalized: (22, 2) float32

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            motion,
            1,  # length = 1 for single frame
            self.dataset_source,
            image,  # [H, W, 3] uint8 or None
            has_video,  # bool
            joints_2d_normalized,  # (22, 2) float32 or None
        )

"""
Scaffold Dataset for ReCon++ Fine-tuning
- Classification task: Predict scaffold characteristics (floors, bays, safety status)
- Adapts ReCon++ encoder to understand scaffold structures
"""

import os
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from .build import DATASETS
from ReConV2.utils.logger import print_log


def pc_normalize(pc):
    """Normalize point cloud to [-1, 1]"""
    pc = pc - np.mean(pc, axis=0)
    max_norm = np.max(np.linalg.norm(pc, axis=1))
    if max_norm < 1e-6:
        return np.zeros_like(pc)
    return pc / max_norm


def farthest_point_sample(point, npoint):
    """Farthest point sampling"""
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return point[centroids.astype(np.int32)]


@DATASETS.register_module()
class ScaffoldDataset(Dataset):
    """
    Scaffold Dataset for ReCon++ fine-tuning.

    Classification targets:
    - Class 0-11: Combined (floors-1) * 4 + (bays-3) for floors in [2,3,4] and bays in [3,4,5,6]
      This gives 3*4=12 classes representing scaffold configurations

    Alternative: Use multi-label classification
    - floors: 2, 3, 4, 5 (4 classes)
    - bays: 3, 4, 5, 6 (4 classes)
    - safety: safe, minor_defect, major_defect (3 classes)
    """

    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.num_category = config.NUM_CATEGORY
        self.subset = config.subset
        self.with_color = config.get('with_color', False)

        # Load split
        split_file = os.path.join(self.root, 'split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            self.scene_ids = splits.get(self.subset, [])
        else:
            # Fallback: list all files
            pcs_dir = os.path.join(self.root, 'pcs')
            self.scene_ids = [f.replace('.npy', '') for f in os.listdir(pcs_dir) if f.endswith('.npy')]
            if self.subset == 'train':
                self.scene_ids = self.scene_ids[:int(len(self.scene_ids) * 0.8)]
            elif self.subset == 'val':
                self.scene_ids = self.scene_ids[int(len(self.scene_ids) * 0.8):int(len(self.scene_ids) * 0.9)]
            else:
                self.scene_ids = self.scene_ids[int(len(self.scene_ids) * 0.9):]

        self.pcs_dir = os.path.join(self.root, 'pcs')
        self.meta_dir = os.path.join(self.root, 'meta')

        # Class definitions
        # Combined class: (floors-2)*4 + (bays-3) for floors in [2,3,4,5] and bays in [3,4,5,6]
        self.floor_range = [2, 3, 4, 5]
        self.bay_range = [3, 4, 5, 6]
        self.num_floor_classes = len(self.floor_range)
        self.num_bay_classes = len(self.bay_range)

        # Safety classes
        self.safety_classes = ['safe', 'minor_defect', 'major_defect']

        # Build dataset
        self._build_dataset()

        print_log(f'ScaffoldDataset [{self.subset}]: {len(self.data)} samples, '
                  f'{self.num_category} classes', logger='ScaffoldDataset')

    def _build_dataset(self):
        """Build dataset from scene files"""
        self.data = []
        self.labels = []

        cache_file = os.path.join(self.root, f'scaffold_{self.subset}_{self.npoints}pts.pkl')

        if os.path.exists(cache_file):
            print_log(f'Loading cached data from {cache_file}', logger='ScaffoldDataset')
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.data = cache['data']
                self.labels = cache['labels']
            return

        print_log(f'Building dataset for {self.subset}...', logger='ScaffoldDataset')

        for scene_id in tqdm(self.scene_ids):
            # Load point cloud
            pc_file = os.path.join(self.pcs_dir, f'{scene_id}.npy')
            if not os.path.exists(pc_file):
                continue

            points = np.load(pc_file)

            # Only use xyz (first 3 channels)
            if points.shape[1] > 3:
                xyz = points[:, :3]
            else:
                xyz = points

            # Sample points
            if len(xyz) > self.npoints:
                xyz = farthest_point_sample(xyz, self.npoints)
            elif len(xyz) < self.npoints:
                # Pad with random duplicates
                pad_idx = np.random.choice(len(xyz), self.npoints - len(xyz), replace=True)
                xyz = np.vstack([xyz, xyz[pad_idx]])

            # Normalize
            xyz = pc_normalize(xyz)

            # Load metadata for label
            meta_file = os.path.join(self.meta_dir, f'{scene_id}_meta.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                config = meta.get('config', {})
                num_floors = config.get('num_floors', 3)
                num_bays = config.get('num_bays', 3)
                safety = config.get('safety_status', 'safe')
            else:
                num_floors = 3
                num_bays = 3
                safety = 'safe'

            # Compute class label
            # Clamp to valid ranges
            floors_idx = max(0, min(num_floors - 2, self.num_floor_classes - 1))
            bays_idx = max(0, min(num_bays - 3, self.num_bay_classes - 1))

            # Combined label: floor_idx * num_bays + bay_idx
            label = floors_idx * self.num_bay_classes + bays_idx

            self.data.append(xyz.astype(np.float32))
            self.labels.append(label)

        # Cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'data': self.data, 'labels': self.labels}, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        points = self.data[index].copy()
        label = self.labels[index]

        # Random shuffle for training
        if self.subset == 'train':
            np.random.shuffle(points)

        # Add color if needed
        if self.with_color:
            color = np.ones_like(points) * 0.6
            points = np.concatenate([points, color], axis=-1)

        points = torch.from_numpy(points).float()
        return 'Scaffold', f'sample_{index}', (points, label)


@DATASETS.register_module()
class ScaffoldSafetyDataset(Dataset):
    """
    Scaffold Dataset with Safety Classification target.
    - 3 classes: safe, minor_defect, major_defect
    """

    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.num_category = 3  # safe, minor, major
        self.subset = config.subset
        self.with_color = config.get('with_color', False)

        # Load split
        split_file = os.path.join(self.root, 'split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            self.scene_ids = splits.get(self.subset, [])
        else:
            pcs_dir = os.path.join(self.root, 'pcs')
            self.scene_ids = [f.replace('.npy', '') for f in os.listdir(pcs_dir) if f.endswith('.npy')]

        self.pcs_dir = os.path.join(self.root, 'pcs')
        self.meta_dir = os.path.join(self.root, 'meta')

        self.safety_classes = {'safe': 0, 'minor_defect': 1, 'major_defect': 2}

        self._build_dataset()

        print_log(f'ScaffoldSafetyDataset [{self.subset}]: {len(self.data)} samples, '
                  f'3 classes (safe/minor/major)', logger='ScaffoldSafetyDataset')

    def _build_dataset(self):
        self.data = []
        self.labels = []

        cache_file = os.path.join(self.root, f'scaffold_safety_{self.subset}_{self.npoints}pts.pkl')

        if os.path.exists(cache_file):
            print_log(f'Loading cached data from {cache_file}', logger='ScaffoldSafetyDataset')
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.data = cache['data']
                self.labels = cache['labels']
            return

        print_log(f'Building safety dataset for {self.subset}...', logger='ScaffoldSafetyDataset')

        for scene_id in tqdm(self.scene_ids):
            pc_file = os.path.join(self.pcs_dir, f'{scene_id}.npy')
            if not os.path.exists(pc_file):
                continue

            points = np.load(pc_file)
            if points.shape[1] > 3:
                xyz = points[:, :3]
            else:
                xyz = points

            if len(xyz) > self.npoints:
                xyz = farthest_point_sample(xyz, self.npoints)
            elif len(xyz) < self.npoints:
                pad_idx = np.random.choice(len(xyz), self.npoints - len(xyz), replace=True)
                xyz = np.vstack([xyz, xyz[pad_idx]])

            xyz = pc_normalize(xyz)

            meta_file = os.path.join(self.meta_dir, f'{scene_id}_meta.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                safety = meta.get('config', {}).get('safety_status', 'safe')
            else:
                safety = 'safe'

            label = self.safety_classes.get(safety, 0)

            self.data.append(xyz.astype(np.float32))
            self.labels.append(label)

        with open(cache_file, 'wb') as f:
            pickle.dump({'data': self.data, 'labels': self.labels}, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        points = self.data[index].copy()
        label = self.labels[index]

        if self.subset == 'train':
            np.random.shuffle(points)

        if self.with_color:
            color = np.ones_like(points) * 0.6
            points = np.concatenate([points, color], axis=-1)

        points = torch.from_numpy(points).float()
        return 'ScaffoldSafety', f'sample_{index}', (points, label)

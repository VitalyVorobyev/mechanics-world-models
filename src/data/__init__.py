"""Dataset storage helpers."""

from importlib import import_module

from data.image_transforms import ImageCropConfig
from data.trajectory import Trajectory, load_trajectory_npz, save_trajectory_npz

__all__ = [
    "DatasetStats",
    "EpisodeInfo",
    "EpisodeSequenceDataset",
    "ImageCropConfig",
    "SequenceLocation",
    "Trajectory",
    "compute_dataset_stats",
    "index_episode_files",
    "load_trajectory_npz",
    "make_train_val_datasets",
    "read_episode_info",
    "save_trajectory_npz",
    "split_episode_files",
    "valid_sequence_starts",
]


def __getattr__(name: str) -> object:
    """Lazily expose torch-backed dataset utilities."""

    if name in {
        "EpisodeInfo",
        "EpisodeSequenceDataset",
        "SequenceLocation",
        "index_episode_files",
        "make_train_val_datasets",
        "read_episode_info",
        "split_episode_files",
        "valid_sequence_starts",
    }:
        sequence_dataset = import_module("data.sequence_dataset")
        return getattr(sequence_dataset, name)
    if name in {"DatasetStats", "compute_dataset_stats"}:
        dataset_stats = import_module("data.dataset_stats")
        return getattr(dataset_stats, name)
    raise AttributeError(f"module 'data' has no attribute {name!r}")

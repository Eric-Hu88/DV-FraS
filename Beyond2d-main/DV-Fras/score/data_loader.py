import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FHADataset(Dataset):
    """Paired AP/LAT global crops, two ROIs per view, and four mRUST labels."""

    VIEWS = ("ap", "lat")

    def __init__(self, data_json, global_transform=None, local_transform=None):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.root = os.path.dirname(os.path.abspath(data_json))

        with open(data_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "pairs" in payload:
            payload = payload["pairs"]
        self.data_list = payload
        if not isinstance(self.data_list, list):
            raise ValueError("The scoring JSON must contain a list of examinations.")

    def __len__(self):
        return len(self.data_list)

    @property
    def subject_ids(self):
        return {
            str(item.get("subject_id", item.get("examination_id", item.get("id", index))))
            for index, item in enumerate(self.data_list)
        }

    def _resolve(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

    @staticmethod
    def _first(item, keys):
        for key in keys:
            if key in item:
                return item[key]
        return None

    @classmethod
    def _view_record(cls, item, view):
        view_index = 0 if view == "ap" else 1
        if view in item and isinstance(item[view], dict):
            record = item[view]
            global_path = cls._first(record, ("global", "image", "path"))
            roi_paths = record.get("rois")
            scores = cls._first(record, ("scores", "labels"))
        else:
            global_path = cls._first(
                item, (view, f"global_{view}", f"{view}_global", f"{view}_path")
            )
            roi_paths = cls._first(item, (f"rois_{view}", f"{view}_rois"))
            scores = cls._first(item, (f"scores_{view}", f"{view}_scores"))

        if roi_paths is None and "rois" in item:
            all_rois = item["rois"]
            if isinstance(all_rois, dict):
                roi_paths = all_rois[view]
            elif len(all_rois) == 4:
                roi_paths = all_rois[view_index * 2 : view_index * 2 + 2]
        if scores is None:
            all_scores = cls._first(item, ("scores", "labels"))
            if isinstance(all_scores, dict):
                scores = all_scores[view]
            elif all_scores is not None and len(all_scores) == 4:
                scores = all_scores[view_index * 2 : view_index * 2 + 2]

        if global_path is None or roi_paths is None or scores is None:
            raise KeyError(
                f"Incomplete {view.upper()} record: expected a paired image, two ROIs, and two scores."
            )
        return global_path, roi_paths, scores

    def _load_image(self, path, transform):
        with Image.open(self._resolve(path)) as image:
            image = image.convert("L")
            return transform(image) if transform else image.copy()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        global_images = []
        local_images = []
        labels = []

        for view in self.VIEWS:
            global_path, roi_paths, scores = self._view_record(item, view)
            if len(roi_paths) != 2 or len(scores) != 2:
                raise ValueError(f"Examination {idx}, view {view} must contain two ROIs and two scores.")
            global_images.append(self._load_image(global_path, self.global_transform))
            local_images.append(
                torch.stack(
                    [self._load_image(path, self.local_transform) for path in roi_paths]
                )
            )
            labels.append(torch.as_tensor(scores, dtype=torch.long) - 1)

        globals_tensor = torch.stack(global_images)  # [2, 1, 224, 224]
        locals_tensor = torch.stack(local_images)  # [2, 2, 1, 224, 224]
        labels_tensor = torch.stack(labels)  # [2, 2]
        if torch.any((labels_tensor < 0) | (labels_tensor > 3)):
            raise ValueError(f"Examination {idx} contains an mRUST score outside 1--4.")

        sample_id = str(item.get("examination_id", item.get("id", idx)))
        subject_id = str(item.get("subject_id", sample_id))
        return globals_tensor, locals_tensor, labels_tensor, sample_id, subject_id

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class PairedXRayDataset(Dataset):

    def __init__(self, pair_json, transform=None, return_paths=False):
        self.transform = transform
        self.return_paths = return_paths

        self.root = os.path.dirname(os.path.abspath(pair_json))
        with open(pair_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "pairs" in payload:
            payload = payload["pairs"]
        self.pairs = payload

        if not isinstance(self.pairs, list):
            raise ValueError(
                "The paired-image JSON must be a list, or an object containing a 'pairs' list."
            )
        self.n = len(self.pairs)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        ap_path = self._resolve(pair["ap"])
        lat_path = self._resolve(pair["lat"])
        with Image.open(ap_path) as image:
            img_ap = image.convert("L")
        with Image.open(lat_path) as image:
            img_lat = image.convert("L")
        if self.transform:
            img_ap = self.transform(img_ap)
            img_lat = self.transform(img_lat)

        if self.return_paths:
            return img_ap, img_lat, os.path.basename(ap_path), os.path.basename(lat_path)

        return img_ap, img_lat

    def _resolve(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

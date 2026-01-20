import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class PillDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.image_paths = list(self.image_dir.glob("*.jpg")) # all jpgs
        self.classes = sorted(set(self._get_class_name(p) for p in self.image_paths))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)} # mapping class names to integer labels

        print(f"Found {len(self.image_paths)} images, {len(self.classes)} classes")

    def _get_class_name(self, path):
        # removing _s_XXX or _u_XXX suffix.
        # e.g., "acc_long_600_mg_s_020.jpg" -> "acc_long_600_mg"
        name = path.stem  # filename without extension

        return re.sub(r'_[su]_\d+$', '', name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        class_name = self._get_class_name(img_path)
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label

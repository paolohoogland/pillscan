import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DATA_DIR = Path('data.nosync/pills_raw/ogyeiv2') # has test, train and val folders
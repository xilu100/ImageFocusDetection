# Command

## Path/Item processing

```python
from pathlib import Path

# Current file path
current_file = Path(__file__).resolve()
# Parent folder path
current_dir = current_file.parent
parent_dir = current_dir.parent
# Root folder path
root_dir = parent_dir.parent
# Specify folder path
raw_image_dir = root_dir / "data/raw/train_img"
# Items in the folder
items = list(raw_image_dir.iterdir()) 
item_count = len(items)  # Number of items
```

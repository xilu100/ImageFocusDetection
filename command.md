# Command
## Path/Item processing
```python
import os
# Current file path
current_file = os.path.abspath(__file__)
# Parent folder path
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
# Root folder path
root_dir = os.path.dirname(parent_dir)
# Specify folder path
raw_image_dir = os.path.join(root_dir, 'data/raw/train_img')
# Items in the folder
item = os.listdir(raw_image_dir)
item_count = len(item) # Number of items
```

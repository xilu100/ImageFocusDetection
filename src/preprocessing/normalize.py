import os

def normalize():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    raw_image_dir = os.path.join(root_dir, 'data/raw/train_img')
    print("Processing : ", raw_image_dir)
    item = os.listdir(raw_image_dir)
    item_count = len(item)
    print("Number of images :",item_count)

if __name__ == "__main__":
    normalize()

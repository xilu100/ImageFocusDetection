from src.preprocessing import normalize_raw, segment_nor_img, label_patches, visualize_labels


def main():
    normalize_raw.normalize_images()
    segment_nor_img.segment_images(estimated_patches=5000)
    label_patches.label(thresholds=[200, 210, 220, 230])
    visualize_labels.visualize()


if __name__ == "__main__":
    main()

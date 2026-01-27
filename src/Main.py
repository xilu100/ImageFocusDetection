import standardize
import segmentation
import features_labels

def main():
    target_blocks=5000
    laplacian=230
    standardize.standardize()
    segmentation.segmentation(target_blocks)
    features_labels.features_labels(laplacian)

if __name__=="__main__":
    main()
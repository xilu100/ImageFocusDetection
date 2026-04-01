# PIPELINE

## 1. processing

* ROOT_PATH = /ImageFocusDetection
* DATA_PATH = /ImageFocusDetection/data
* TRAIN_IMAGE_PATH = /ImageFocusDetection/data/raw/train_img
* VALID_IMAGE_PATH = /ImageFocusDetection/data/raw/valid_img
* VARIABLE(DEFAULT) : path_size = 32

---
The training dataset and validation dataset are processed in the same way, but they are stored in different folders.  
Simply add "valid_" before the corresponding folder or file names.  
e.g.

* FOLDERS : DATA_PATH/normalized ==> DATA_PATH/valid_normalized
* FILES : DATA_PATH/samples/sampleX/sampleX_n_m.png ==> DATA_PATH/valid_samples/valid_sampleX/valid_sampleX_n_m.png

---

1. Original images (DATA_PATH/data) ==> grayscale images (DATA_PATH/normalized)
    1. Stretch the dimensions to integer multiples of patch_size.  
       e.g. path_size = 32  
       Original image size : $$1920(1920 / 32 = 60) * 1080(1080 / 32 = 33.75)
       ==> 1920 (32 * 60) * 1088 (32 * 34) \; or \; 1056 (32 * 33)$$
    2. SAVE : samples_info.csv (DATA_PATH/normalized)  
       e.g.  
       filename , original_size , current_size , aspect_ratio , original_filename  
       sample1.png , 2040x1404 , 2112x1408 , 3:2 , 0001.png  
       sample2.png , 2040x1848 , 1856x1856 , 1:1 , 0002.png  
       sample3.png , 2040x1356 , 2048x1344 , 3:2 , 0003.png
2. Each grayscale image (DATA_PATH/normalized/XXX.png) ==> segmented image (DATA_PATH/samples/sampleX)  
   e.g. 0001.png(DATA_PATH/normalized/0001.png) ==> /sample1/sample1_0_0.png , ... , /sample1/sample1_$n$_$m$.png  
   $n$ and $m$ are determined by the "patch_size",it means coordinates.
3. Record the score (clarity score) of each patch , decide on the threshold(e.g. thresholds=[0.3, 0.4, 0.5, 0.6]).  
   SAVE : sampleX_combined.csv (DATA_PATH/samples_labels/sampleX_lables)  
   e.g.  
   filename,combined_score,laplacian,fft,y_thresh1,y_thresh2,y_thresh3,y_thresh4
   sample1_0_0.png,0.22083768044829888,28.753851890563965,0.41292151313553643,0,0,0,0
   sample1_0_1.png,0.24008858549389528,32.64065170288086,0.44753652376060127,0,0,0,0
   sample1_0_10.png,0.3262259187198985,86.41039180755615,0.5660414512935193,1,0,0,0
   sample1_0_11.png,0.3329493909194818,105.02701473236084,0.5608717727163706,1,0,0,0
4. Recombine the patches according to coordinates, map them to a heatmap via the label (y), and overlay it with the
   original image.  
   SAVE : (DATA_PATH/samples_labels/sampleX_lables)  
   sampleX_thresh1_overlay.png  
   sampleX_thresh2_overlay.png  
   sampleX_thresh3_overlay.png  
   sampleX_thresh4_overlay.png

## 2. Model selection and parameter settings

### Machine Learning

* Support Vector Machine (SVM)
* Decision Tree
* Random Forest

### Deep Learning

* Convolutional Neural Network (CNN)
## 3. Model training
pass

        
   
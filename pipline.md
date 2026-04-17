# Preprocessing

## 0. Patch size selection

```yaml
- Title: "Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment"
  URL: "https://ieeexplore.ieee.org/abstract/document/8063957"
  BibTeX: |
    @ARTICLE{8063957,
    author={Bosse, Sebastian and Maniry, Dominique and Müller, Klaus-Robert and Wiegand, Thomas and Samek, Wojciech},
    journal={IEEE Transactions on Image Processing}, 
    title={Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment}, 
    year={2018},
    volume={27},
    number={1},
    pages={206-219},
    keywords={Feature extraction;Image quality;Distortion;Databases;Optimization;Computational modeling;Full-reference image quality assessment;no-reference image quality assessment;neural networks;quality pooling;deep learning;feature extraction;regression},
    doi={10.1109/TIP.2017.2760518}
    }

- Title: "An image is worth 16x16 words: Transformers for image recognition at scale"
  URL: "https://arxiv.org/pdf/2010.11929/100"
  BibTeX: |
    @article{dosovitskiy2020image,
    title={An image is worth 16x16 words: Transformers for image recognition at scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
```

## 1. Image normalization

```yaml
- Title: "Image quality assessment: from error visibility to structural similarity"
  Field: "grayscale image"
  URL: "https://ieeexplore.ieee.org/abstract/document/1284395"
  BibTeX: |
    @ARTICLE{1284395,
    author={Zhou Wang and Bovik, A.C. and Sheikh, H.R. and Simoncelli, E.P.},
    journal={IEEE Transactions on Image Processing},
    title={Image quality assessment: from error visibility to structural similarity},
    year={2004},
    volume={13},
    number={4},
    pages={600-612},
    keywords={Image quality;Humans;Transform coding;Visual system;Visual perception;Data mining;Layout;Quality assessment;Degradation;Indexes},
    doi={10.1109/TIP.2003.819861}
    }

- Title: "An image is worth 16x16 words: Transformers for image recognition at scale"
  Field: "H,W≡0(mod patch size)"
  URL: "https://arxiv.org/pdf/2010.11929/100"
  BibTeX: |
    @article{dosovitskiy2020image,
    title={An image is worth 16x16 words: Transformers for image recognition at scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
```

## 2. Image segmentation

```yaml
- Title: "Patch-based convolutional neural network for whole slide tissue image classification"
  Field: "Kernel of This Thesis"
  URL: "https://openaccess.thecvf.com/content_cvpr_2016/html/Hou_Patch-Based_Convolutional_Neural_CVPR_2016_paper.html"
  BibTeX: |
    @InProceedings{Hou_2016_CVPR,
    author = {Hou, Le and Samaras, Dimitris and Kurc, Tahsin M. and Gao, Yi and Davis, James E. and Saltz, Joel H.},
    title = {Patch-Based Convolutional Neural Network for Whole Slide Tissue Image Classification},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2016}
    }
```

## 3. Patch labeling

### 3.1 Image sharpness scoring strategy

```yaml
- Title: "A No-Reference Image Blur Metric Based on the Cumulative Probability of Blur Detection (CPBD)"
  Field: "Scoring without reference image"
  URL: "https://ieeexplore.ieee.org/abstract/document/5739529"
  BibTeX: |
    @ARTICLE{5739529,
    author={Narvekar, Niranjan D. and Karam, Lina J.},
    journal={IEEE Transactions on Image Processing}, 
    title={A No-Reference Image Blur Metric Based on the Cumulative Probability of Blur Detection (CPBD)}, 
    year={2011},
    volume={20},
    number={9},
    pages={2678-2683},
    keywords={Measurement;Image edge detection;Databases;Transform coding;Pixel;Image coding;Visualization;Blur detection;blur metric;no-reference;objective;perceptual;sharpness metric;visual quality},
    doi={10.1109/TIP.2011.2131660}
    }
- Title: "Image quality assessment: from error visibility to structural similarity"
  Field: "Integration Metrics -> Stable pseudo-labels"
  URL: "https://ieeexplore.ieee.org/abstract/document/1284395"
  BibTeX: |
    @ARTICLE{1284395,
    author={Zhou Wang and Bovik, A.C. and Sheikh, H.R. and Simoncelli, E.P.},
    journal={IEEE Transactions on Image Processing},
    title={Image quality assessment: from error visibility to structural similarity},
    year={2004},
    volume={13},
    number={4},
    pages={600-612},
    keywords={Image quality;Humans;Transform coding;Visual system;Visual perception;Data mining;Layout;Quality assessment;Degradation;Indexes},
    doi={10.1109/TIP.2003.819861}
    }
```

### 3.2

```yaml
- Title: "Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks"
  Field: "Pseudo-label"
  URL: "https://openreview.net/pdf?id=3iGjh_NmoG"
  BibTeX: |
    @inproceedings{lee2013pseudo,
    title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
    author={Lee, Dong-Hyun and others},
    booktitle={Workshop on challenges in representation learning, ICML},
    volume={3},
    number={2},
    pages={896},
    year={2013},
    organization={Atlanta}
    }
```

## 4. Original image reconstruction with labels

```yaml
- Title: "Fully Convolutional Networks for Semantic Segmentation"
  Field: "Reconstruction Strategy Guidance"
  URL: "https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html"
  BibTeX: |
    @InProceedings{Long_2015_CVPR,
    author = {Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
    title = {Fully Convolutional Networks for Semantic Segmentation},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2015}
    }
- Title: "Learning Deep Features for Discriminative Localization"
  Field: "Heatmap"
  URL: "https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html"
  BibTeX: |
    @InProceedings{Zhou_2016_CVPR,
    author = {Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Oliva, Aude and Torralba, Antonio},
    title = {Learning Deep Features for Discriminative Localization},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2016}
    }
```

# Model selection - Machine Learning

## 1. Decision Tree

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 2. Random Forest

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 3. Support Vector Machine (SVM)

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 4. Clustering

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

# Model selection - Deep Learning

## 1. Convolutional Neural Networks (CNN)

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 2. Transformer

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

# Evaluate

## 1. Accuracy, Precision, Recall, F1-score

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 2. Confusion matrix

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 3. Robustness across different image types and blur levels (Optional)

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 4. Training efficiency and computational cost (Time , Speed)

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

# Other metrics

## 1. Edge strength, gradient, Laplacian (Optional Reference)

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 2. GLCM, LBP

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

## 3. FFT, DCT (Optional Reference)

```yaml
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
- Title: ""
  Field: ""
  URL: ""
  BibTeX: |
```

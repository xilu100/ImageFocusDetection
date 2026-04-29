# Preprocessing

## 0. Patch size selection

Patch-based local classification is the core representation in this workflow [@Hou_2016_CVPR], and fixed-size image patches are also consistent with transformer-style patch tokenization [@dosovitskiy2020image].

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

Grayscale normalization and size alignment are used here to keep the focus-measure pipeline and downstream patch extraction comparable across images [@1284395; @dosovitskiy2020image].

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

This step follows a patch-based image analysis strategy, where the original image is decomposed into local regions before classification [@Hou_2016_CVPR].

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

The sharpness scoring stage combines no-reference blur assessment and structure-aware image quality cues to build rule-based labels [@5739529; @1284395].

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

This stage is related to pseudo-label style supervision in the broad weak-labeling sense, although the thesis itself distinguishes rule-based auto-labeling from model-generated pseudo-labels [@lee2013pseudo].

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

Projection of patch predictions back to image space is conceptually close to dense prediction and localization-style visualization workflows [@Long_2015_CVPR; @Zhou_2016_CVPR].

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

## 0. Principal Component Analysis

PCA is used here as a dimensionality-reduction baseline, while keeping in mind that explained variance alone does not guarantee better classification performance [@perry2025inference; @zheng2021application].

```yaml
- Title: "Inference on the Proportion of Variance Explained in Principal Component Analysis"
  Field: "explained variance"
  URL: "https://www.tandfonline.com/doi/abs/10.1080/01621459.2025.2538895"
  BibTeX: |
    @article{perry2025inference,
    title={Inference on the proportion of variance explained in principal component analysis},
    author={Perry, Ronan and Panigrahi, Snigdha and Bien, Jacob and Witten, Daniela},
    journal={Journal of the American Statistical Association},
    pages={1--11},
    year={2025},
    publisher={Taylor \& Francis}
    }
- Title: "On the Application of Principal Component Analysis to Classification Problems"
  Field: "explained variance is not always good"
  URL: "https://digitalcommons.chapman.edu/scs_articles/729/"
  BibTeX: |
    @article{zheng2021application,
    title={On the application of principal component analysis to classification problems},
    author={Zheng, Jianwei and Rakovski, Cyril},
    year={2021}
    }
```

## 1. Decision Tree

Decision trees provide an interpretable classical baseline for patch classification [@breiman2017classification].

```yaml
- Title: "Classification and Regression Trees"
  Field: "Decision Tree"
  URL: "https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-olshen-charles-stone"
  BibTeX: |
    @book{breiman2017classification,
    title={Classification and regression trees},
    author={Breiman, Leo and Friedman, Jerome and Olshen, Richard A and Stone, Charles J},
    year={2017},
    publisher={Chapman and Hall/CRC}
    }
```

## 2. Random Forest

Random forests extend single-tree decision rules through ensemble aggregation and usually improve robustness [@breiman2001random].

```yaml
- Title: "Random Forests"
  Field: "Random Forest"
  URL: "https://link.springer.com/article/10.1023/a:1010933404324"
  BibTeX: |
    @article{breiman2001random,
    title={Random forests},
    author={Breiman, Leo},
    journal={Machine learning},
    volume={45},
    number={1},
    pages={5--32},
    year={2001},
    publisher={Springer}
    }
```

## 3. Support Vector Machine (SVM)

SVM is included as a strong margin-based classifier for medium-scale pixel-vector features [@cortes1995support].

```yaml
- Title: "Support-Vector Networks"
  Field: "SVM"
  URL: "https://link.springer.com/article/10.1007/Bf00994018"
  BibTeX: |
    @article{cortes1995support,
    title   = {Support-Vector Networks},
    author  = {Cortes, Corinna and Vapnik, Vladimir},
    journal = {Machine Learning},
    volume  = {20},
    number  = {3},
    pages   = {273--297},
    year    = {1995},
    publisher = {Springer},
    doi     = {10.1007/BF00994018}
    }
```

## 4. Clustering(Optional)

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

CNNs serve as the deep learning baseline for directly learning patch-level discriminative patterns from pixels [@lecun2002gradient; @krizhevsky2012imagenet].

```yaml
- Title: "ImageNet Classification with Deep Convolutional Neural Networks"
  Field: "CNN Use"
  URL: "https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html"
  BibTeX: |
    @article{krizhevsky2012imagenet,
    title={Imagenet classification with deep convolutional neural networks},
    author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
    journal={Advances in neural information processing systems},
    volume={25},
    year={2012}
    }
- Title: "Gradient-based learning applied to document recognition"
  Field: "CNN LeNet"
  URL: "https://ieeexplore.ieee.org/abstract/document/726791/"
  BibTeX: |
    @article{lecun2002gradient,
    title={Gradient-based learning applied to document recognition},
    author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
    journal={Proceedings of the IEEE},
    volume={86},
    number={11},
    pages={2278--2324},
    year={2002},
    publisher={Ieee}
    }
```

## 2. Transformer (Optional)

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

These metrics are standard for classification evaluation and are especially important under class imbalance [@schutze2008introduction].

```yaml
- Title: "Introduction to information retrieval"
  Field: "Defined Precision, Recall, F1-score"
  URL: "https://web.cs.hacettepe.edu.tr/~pinar/courses/VBM681/lectures/Shutze-19web.pdf"
  BibTeX: |
    @book{schutze2008introduction,
    title={Introduction to information retrieval},
    author={Sch{\"u}tze, Hinrich and Manning, Christopher D and Raghavan, Prabhakar},
    volume={39},
    year={2008},
    publisher={Cambridge University Press Cambridge}
    }
```

## 2. Confusion matrix

The confusion matrix is used to inspect which focus states are most frequently confused with each other [@schutze2008introduction].

```yaml
- Title: "Introduction to information retrieval"
  Field: "Defined Confusion matrix"
  URL: "https://web.cs.hacettepe.edu.tr/~pinar/courses/VBM681/lectures/Shutze-19web.pdf"
  BibTeX: |
    @book{schutze2008introduction,
    title={Introduction to information retrieval},
    author={Sch{\"u}tze, Hinrich and Manning, Christopher D and Raghavan, Prabhakar},
    volume={39},
    year={2008},
    publisher={Cambridge University Press Cambridge}
    }
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

Optimization efficiency, mini-batch behavior, and learning-rate sensitivity are discussed here in line with standard SGD practice [@bottou2012stochastic].

```yaml
- Title: "Stochastic Gradient Descent Tricks"
  Field: "mini-batch,learning rate"
  URL: "https://link.springer.com/chapter/10.1007/978-3-642-35289-8_25"
  BibTeX: |
    @incollection{bottou2012stochastic,
    title={Stochastic gradient descent tricks},
    author={Bottou, L{\'e}on},
    booktitle={Neural networks: tricks of the trade: second edition},
    pages={421--436},
    year={2012},
    publisher={Springer}
    }
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

Texture descriptors such as GLCM and LBP are optional handcrafted references for describing local structure patterns [@haralick2007textural; @ojala2002multiresolution].

```yaml
- Title: "Textural Features for Image Classification"
  Field: "GLCM"
  URL: "https://ieeexplore.ieee.org/abstract/document/4309314"
  BibTeX: |
    @article{haralick2007textural,
    title={Textural features for image classification},
    author={Haralick, Robert M and Shanmugam, Karthikeyan and Dinstein, Its' Hak},
    journal={IEEE Transactions on systems, man, and cybernetics},
    number={6},
    pages={610--621},
    year={2007},
    publisher={Ieee}
    }
- Title: "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns"
  Field: "LBP"
  URL: "https://ieeexplore.ieee.org/abstract/document/1017623"
  BibTeX: |
    @article{ojala2002multiresolution,
    title={Multiresolution gray-scale and rotation invariant texture classification with local binary patterns},
    author={Ojala, Timo and Pietikainen, Matti and Maenpaa, Topi},
    journal={IEEE Transactions on pattern analysis and machine intelligence},
    volume={24},
    number={7},
    pages={971--987},
    year={2002},
    publisher={IEEE}
    }
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

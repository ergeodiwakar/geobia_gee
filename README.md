# Geographic Object-Based Image Analysis for Landslide Identification Using Machine Learning on Google Earth Engine  

## Authors  
- Diwakar Khadka (Corresponding Author: [diwakar@tongji.edu.cn](mailto:diwakar@tongji.edu.cn))  
- Jie Zhang ([cezhangjie@tongji.edu.cn](mailto:cezhangjie@tongji.edu.cn))  
- Atma Sharma ([atmasharma@tongji.edu.cn](mailto:atmasharma@tongji.edu.cn))  

## Abstract  
Landslides significantly threaten human life and infrastructure, requiring accurate and timely identification for effective hazard assessment and management. This study proposes a new approach combining Geographic Object-Based Image Analysis (GEOBIA) and machine learning on the Google Earth Engine (GEE) platform, utilizing high-resolution Sentinel-2 imagery and NASADEM data. Our methodology begins with Simple Non-iterative Clustering (SNIC) segmentation, which divides the images into homogeneous super-pixels. Following segmentation, Gray Level Co-occurrence Matrix (GLCM) feature extraction is employed to gather critical textural information. Various machine learning algorithms are utilized, including Support Vector Machine (SVM), Random Forest (RF), and Classification and Regression Trees (CART). The performance of these algorithms is evaluated for landslide detection, with RF demonstrating superior accuracy at 87.41%.  

## Introduction  
Landslides are natural disasters that occur globally, causing damage to infrastructure and loss of life. Identifying landslides and measuring their features is crucial but faces numerous technological challenges. This study aims to develop and evaluate a GEOBIA algorithm that integrates SNIC segmentation, GLCM feature extraction, and PCA for dimensionality reduction using the GEE platform to identify landslides effectively.  

## Methodology  

![Methodology Flowchart](https://raw.githubusercontent.com/ergeodiwakar/geobia_gee/main/Methodology.jpg) 

The methodology involves several key steps:  

1. **Data Collection**: High-resolution Sentinel-2 imagery and ALOS DSM data.  
   - Filter by Boundary (ROI)  
   - Filter by Date (2016-01-01 to 2022-12-31)  
   - Filter Cloud Cover and Cloud Band  
2. **Calculation of Spectral Indices**: NDVI, NDWI, ARVI, SAVI, NDMI, EVI, Slope, and Elevation.  
3. **Band Median Composition**: Statics calculation to concatenate band composition and indices.  
4. **SNIC Segmentation**: SNIC segmentation was applied to create homogeneous super-pixels.  
5. **GLCM Textural Analysis**: GLCM was used to extract textural features from the segmented images.  
6. **PCA (Principal Component Analysis)**: PCA was employed to reduce the dimensionality of the data.  
7. **Machine Learning**: SVM, RF, and CART algorithms were trained and evaluated for landslide detection.  

## Results  
The study identified 1,575 landslides within a 505 km² area, achieving an overall accuracy of 87.41% with the Random Forest algorithm, surpassing the other methods. The results indicate that the integrated GEOBIA approach significantly improves landslide detection capabilities.  

## Conclusion  
This research demonstrates the effectiveness of combining GEOBIA with machine learning on the GEE platform for landslide identification. The methodology developed can enhance disaster management responses and improve the understanding of landslide risks in various geographical settings.

## Published Article  
You can download the published research paper from the following link: [Published Research Paper](https://doi.org/10.1007/s12665-024-12045-8)  

# MST-NET: Multimodal SpatioTemporal Network for Pulmonary Nodule Segmentation and PrognosisMST-Net

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Awesome-FC60A8?style=flat-square&logo=Awesome&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)

## Proposed method
Overview of our MST-Net. It comprises three parts: (a) multimodal feature extraction module (visual and textual data), (b) the Vit-STUnet module for lung nodule detection and segmentation, and (c) a cross-modal spatiotemporal prediction module that fuses imaging and clinical data from different time points using CMSTF for final prediction.

![image](image/model.png)

## Experiment result
We compared our results with other state-of-the-art methods, and our results were better than any other comparison method. In summary, the MST-Net model, trained with a combination of multimodal data and spatiotemporal information, demonstrates superior performance in tasks of pulmonary nodule segmentation and prognosis compared to models trained using single-modality, single-time point, or planar data.

![image](image/result.png)

# Brain-Tumor-Segmentation-Using-3D-U-Net

This repo is about using a 3D U net model to segment brain tumor.
We have implemented unet approach to segment tumors by classifying them into four classes
  0: Background
  1: Enhancing Tumor
  2: Non Enhancing Tumor and
  3: Tumor Core
  
  Our project introduces a new approach called MDG (Modified Data Generator) 3D U-Net, where
  we created a custome datagenerator which augments images on the fly but making sure each augmentation is
  unique, so that we have a new data for training.

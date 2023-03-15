# 3D U-Net Based Brain Tumor Semantic Segmentation Using A Modified Data Generator

Introduction:

This is the Github repository for the research paper titled "3D U-Net Based Brain Tumor Semantic Segmentation Using A Modified Data Generator". The paper is currently under review for publication.

We have implemented unet approach to segment tumors by classifying them into four classes
  0: Background
  1: Enhancing Tumor
  2: Non Enhancing Tumor and
  3: Tumor Core

3D U-net Model Architecture:
![c-An-example-3D-U-Net-structure-with-3-encoding-and-3-decoding-blocks](https://user-images.githubusercontent.com/98879512/225287951-989295c9-d30b-46b7-a879-bee79651dc6a.png)

Flowchart:
![Flowchart](https://user-images.githubusercontent.com/98879512/225287559-17a5e1f7-851f-458f-8f5a-384e5adc4f69.jpeg)

Results:
![Dice to Specificity](https://user-images.githubusercontent.com/98879512/225287394-f95fe798-5f5e-4b52-b51a-54d85e557e99.png)
![HD95 copy](https://user-images.githubusercontent.com/98879512/225287483-4514bd66-7566-4158-b339-3be34ffc337a.png)
![seg vs gt](https://user-images.githubusercontent.com/98879512/225287603-3a655ebd-63ea-43b5-8eb0-77557123bd61.png)


Overview:

Brain tumor segmentation is a critical step in medical imaging analysis, as it allows physicians to identify the extent of the tumor and plan appropriate treatment. In recent years, deep learning techniques have shown great potential in medical image segmentation tasks. In this paper, we propose a Modified Data Generator Approach for Accurate 3D Brain Tumor Segmentation on the BraTS 2020 dataset.

Dataset:

We used the BraTS 2020 dataset, which contains multimodal MRI scans of 100 patients, each with four tumor subtypes (WT, TC, ET, and NCR/NET). The dataset was divided into 75% for training and 25% for validation.

Methodology:

We utilized a 3D U-Net architecture, which has been shown to be effective in segmentation tasks. The model consisted of multiple blocks of convolutional, max pooling, and transpose convolutional layers, as well as batch normalization, activation, and dropout layers. The encoder network used max-pooling layers to gradually reduce or down-sample the dimensions of the input, while the decoder network utilized the transpose operation on convolutional layers to increase the resolution of feature maps obtained from the encoder. Skip connections were employed after each transpose convolutional layer to concatenate the feature maps obtained from the decoder with their corresponding feature maps obtained from the encoder.

Training:

We trained our model on a 16GB NVIDIA T4 TENSOR CORE GPU on Google Colab, and the training process was completed over 59 epochs, taking around 88 hours. Adam with a starting learning rate of 0.0001 was used as an optimizer, which reduces the learning rate by 20% when the training curve does not improve after five epochs.

Evaluation:

We evaluated our model's performance using various evaluation metrics, including Mean Intersection over Union (MeanIoU), Mean Dice score, 95% Hausdorff distance in mm, sensitivity, and specificity. Our proposed approach outperformed previous works on the BraTS 2020 dataset, achieving mean dice scores of 77.8%, 82.2%, and 90.3% for ET, TC, and WT, respectively. The 95% Hausdorff distances obtained were 6.1, 7.4, and 7.5 mm for WT, TC, and ET, respectively. We presented the results of all metrics used in a box plot to provide a comprehensive evaluation of our model's performance.

Conclusion:

Our proposed Modified Data Generator Approach for Accurate Brain Tumor Segmentation on the BraTS 2020 dataset achieved robust and accurate brain tumor segmentation results. The proposed model outperformed previous works on the BraTS 2020 dataset, demonstrating the potential of deep learning techniques in medical image segmentation tasks. This Github repository provides the code and resources used in our study, which can be utilized by researchers and practitioners to develop and evaluate their own brain tumor segmentation models.

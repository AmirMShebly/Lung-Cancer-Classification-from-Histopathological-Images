## Lung Cancer Classification from Histopathological Images

Histopathological images, obtained from tissue samples, play a crucial role in diagnosing and classifying diseases. These images reveal the microscopic structure of cells and tissues, providing valuable insights into the nature of the illness. In the context of lung cancer, histopathological analysis helps to differentiate between different types of cancer and identify the presence of benign (non-cancerous) growths.

This project focuses on classifying three major types of lung cancer:

• Adenocarcinoma: The most common type of lung cancer, originating in the mucus-producing cells of the lung.

• Squamous cell carcinoma: Cancer that develops in the squamous cells that line the airways.

• Benign: Non-cancerous growths that do not spread to other tissues.

### Dataset

This project utilizes the publicly available Lung and Colon Cancer Histopathological Image Dataset, published in the National Library of Medicine in 2023 [Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset](https://arxiv.org/abs/1912.12142v1). This dataset provides a valuable resource for training lung cancer classification models. It includes:

• 5000 images for each of the three lung cancer types (adenocarcinoma, squamous cell carcinoma, and benign) 

• High-resolution images capturing detailed cellular structures

• Annotations for accurate classification

Here are some sample images from this dataset:

![Lung Cancers](https://github.com/user-attachments/assets/88de75bd-ffb0-47d7-8dcc-9e8c143c0ca9)
![Lung Cancers_2](https://github.com/user-attachments/assets/6aedd617-d4d3-472a-aab6-af8254c6dd81)


### Methodology

A Convolutional Neural Network (CNN) was employed for this task. CNNs are particularly well-suited for image classification due to their ability to automatically extract relevant features from the image data. The CNN architecture was carefully designed and trained using the provided dataset. 

### Results

The developed model achieved 99% F1 score, demonstrating high accuracy in classifying lung cancer types from histopathological images. This result indicates the model's ability to effectively distinguish between different types of lung cancer and identify benign cases.

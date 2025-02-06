A real-time system capable of detecting and recognizing vehicle license plates using computer vision and machine learning techniques. This system is designed for applications like automated parking, toll booths, and traffic monitoring.

**Technologies Used:**

- YOLOv8: For real-time license plate detection.

- VGG16: For feature extraction from detected license plates.

- OpenCV: For image preprocessing, bounding box detection, and Optical Character Recognition (OCR).

- PyTorch: For implementing and training the deep learning models.

- Roboflow: For sourcing and preprocessing a dataset of 31,250+ images.

**Key Steps Involved:**

- Data Preprocessing: Collected and annotated images, applied resizing, normalization, and augmentation to improve model robustness.

- Object Detection: Trained YOLOv8 to detect license plates in real-time with high accuracy.

- Feature Extraction: Used VGG16 to extract features from the detected license plates.

- OCR: Implemented OCR using OpenCV to recognize characters on the license plates.

- Validation: Added logic to validate detected license plates and correct common OCR errors.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Input image
def getcrop(img):
    inputImage = img
    inputImageCopy = inputImage.copy()

    # Convert BGR to grayscale:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Threshold via Otsu:
    threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.floodFill(binaryImage, None, (0, 0), 0)
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store bounding boxes of characters
    character_bboxes = []

    for _, c in enumerate(contours):
        # Get the bounding rectangle of the current contour:
        boundRect = cv2.boundingRect(c)

        # Get the bounding rectangle data:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Estimate the bounding rect area:
        rectArea = rectWidth * rectHeight

        # Set a min area threshold
        minArea = 500

        # Set a max area threshold
        maxArea = 5000

        # Filter blobs by area:
        if rectArea > minArea and maxArea > rectArea and rectHeight/rectWidth > 1:
            # Store the bounding box information
            character_bboxes.append(boundRect)

    # sort by bbox[0] (x value)
    character_bboxes = sorted(character_bboxes, key=lambda x: x[0])
    
    #Print
    print(character_bboxes)
    print([c[0] for c in character_bboxes])

    # Draw bounding boxes of characters
    for bbox in character_bboxes:
        rectX, rectY, rectWidth, rectHeight = bbox
        color = (0, 0, 0)
        cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                    (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
    cropped_characters = []
    # Display the image with bounding boxes
    for bbox in character_bboxes:
        rectX, rectY, rectWidth, rectHeight = bbox
        # Expand bounding box by 10 pixels on all sides
        expanded_rectX = max(0, rectX - 5)  
        expanded_rectY = max(0, rectY - 5)
        expanded_rectWidth = min(inputImage.shape[1], rectWidth + 10)
        expanded_rectHeight = min(inputImage.shape[0], rectHeight + 10)
        # Crop the expanded region of interest (ROI) from the original image
        character_image = inputImage[expanded_rectY:expanded_rectY + expanded_rectHeight, expanded_rectX:expanded_rectX + expanded_rectWidth]
        # Append the cropped character image to the list
        cropped_characters.append(character_image)
    return cropped_characters

import cv2
import numpy as np
import matplotlib.pyplot as plt
# Set image path
# path = "D://opencvImages//"
# fileName = "rhWM3.png"

# Read Input image
inputImage = cv2.imread('just_plate.jpg')

# Deep copy for results:
inputImageCopy = inputImage.copy()

# Convert BGR to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Threshold via Otsu:
threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.floodFill(binaryImage, None, (0, 0), 0)
contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    minArea = 10

    # Filter blobs by area:
    if rectArea > minArea:

        # Draw bounding box:
        color = (0, 255, 0)
        cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                      (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
        plt.imshow( inputImageCopy)
        plt.title('Bounding Boxes')
        plt.show()
        # Crop bounding box:
        currentCrop = inputImage[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
        plt.imshow(currentCrop)
        plt.title("Current Crop")
        plt.show()
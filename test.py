from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from final_model import create_model, inference
from PIL import Image

def extract_plate(img, bbox):
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def fix_dimension(img): 
    img = cv2.resize(img, (224, 224))    
    t = np.ones((224, 224, 3))
    for i in range(3):
        t[:, :, i] = img
    
    # Erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(t.astype(np.uint8), kernel, iterations=3)
    
    # Gaussian Blur
    blurred_img = cv2.GaussianBlur(eroded_img, (9, 9), 0)
    
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    character_bboxes = []

    for cntr in cntrs:
        x, y, w, h = cv2.boundingRect(cntr)
        if lower_width < w < upper_width and lower_height < h < upper_height:
            character_bboxes.append((x, y, w, h))

    return character_bboxes

def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]

    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


def loadmodel():
    modelYolo = YOLO(r"best.pt")
    modelVGG = create_model()
    return modelYolo, modelVGG

def predictimg(modelYolo, modelVGG, src):
    # source = [src]
    results = modelYolo.predict(source=src)
    for i, r in enumerate(results):
        plate = extract_plate(r.orig_img, r.boxes.xyxy.cpu().numpy().astype(int)[0])
        cv2.imwrite('just_plate.jpg', plate)
        plt.imshow(plate)
        plt.show()
        char_bboxes = segment_characters(plate)
        char_images = [plate[y:y+h, x:x+w] for x, y, w, h in char_bboxes]
        for i in range(len(char_images)):
            plt.subplot(1, len(char_images), i+1)
            plt.imshow(char_images[i], cmap='gray')
            plt.axis('on')
        plt.show()

    # output = []
    # mask = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    # for i in range(len(char_images)):
    #     c = inference(modelVGG, char_images[i], 'num' if mask[i] == 1 else 'alpha')
    #     output.append(c)
    # print(''.join(output))
    # return ''.join(output)
y,m=loadmodel()
print(predictimg(y,m,'car2.jpg'))
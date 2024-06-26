from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from final_model import create_model,inference
from PIL import Image
from sep import getcrop
def extract_plate(img,bbox):
    return img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
def fix_dimension(img): 
    img=cv2.resize(img,(224,224))    
    t=np.ones((224,224,3))
    for i in range(3):
        t[:,:,i]=img
    
    # Erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(t.astype(np.uint8), kernel, iterations=3)
    
    # Gaussian Blur
    blurred_img = cv2.GaussianBlur(eroded_img, (9, 9), 0)
    
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)


# def find_contours(dimensions, img) :

#     # Find all contours in the image
#     cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Retrieve potential dimensions
#     lower_width = dimensions[0]
#     upper_width = dimensions[1]
#     lower_height = dimensions[2]
#     upper_height = dimensions[3]
    
#     # Check largest 5 or  15 contours for license plate or character respectively
#     cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
#     ii = cv2.imread('contour.jpg')
    
#     x_cntr_list = []
#     target_contours = []
#     img_res = []
#     for cntr in cntrs :
#         # detects contour in binary image and returns the coordinates of rectangle enclosing it
#         intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
#         # checking the dimensions of the contour to filter out the characters by contour's size
#         if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
#             x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

#             char_copy = np.zeros((44,24))
#             # extracting each character using the enclosing rectangle's coordinates.
#             char = img[intY:intY+intHeight, intX:intX+intWidth]
#             char = cv2.resize(char, (20, 40))
            
#             cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
#             plt.imshow(ii, cmap='gray')
#             plt.title('Predict Segments')

#             # Make result formatted for classification: invert colors
#             char = cv2.subtract(255, char)

#             # Resize the image to 24x44 with black border
#             char_copy[2:42, 2:22] = char
#             char_copy[0:2, :] = 0
#             char_copy[:, 0:2] = 0
#             char_copy[42:44, :] = 0
#             char_copy[:, 22:24] = 0

#             img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
#     # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
#     plt.show()
#     # arbitrary function that stores sorted list of character indeces
#     indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
#     img_res_copy = []
#     for idx in indices:
#         img_res_copy.append(img_res[idx])# stores character images according to their index
#     img_res = np.array(img_res_copy)

#     return img_res
# def segment_characters(image) :

#     # Preprocess cropped license plate image
#     img_lp = cv2.resize(image, (333, 75))
#     img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
#     _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     img_binary_lp = cv2.erode(img_binary_lp, (3,3))
#     img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

#     LP_WIDTH = img_binary_lp.shape[0]
#     LP_HEIGHT = img_binary_lp.shape[1]

#     # Make borders white
#     img_binary_lp[0:3,:] = 255
#     img_binary_lp[:,0:3] = 255
#     img_binary_lp[72:75,:] = 255
#     img_binary_lp[:,330:333] = 255

#     # Estimations of character contours sizes of cropped license plates
#     dimensions = [LP_WIDTH/6,
#                        LP_WIDTH/2,
#                        LP_HEIGHT/10,
#                        2*LP_HEIGHT/3]
#     plt.imshow(img_binary_lp, cmap='gray')
#     plt.title('Contour')
#     plt.show()
#     cv2.imwrite('contour.jpg',img_binary_lp)

#     # Get contours within cropped license plate
#     char_list = find_contours(dimensions, img_binary_lp)

#     return char_list
def loadmodel():
    modelYolo=YOLO(r"best.pt")
    modelVGG=create_model()
    return modelYolo,modelVGG
def predictimg(modelYolo,modelVGG,src='tmp.jpg'):

    try:
        image = cv2.imread(src)
        img_gray_lp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

        finalimg = cv2.cvtColor(img_binary_lp, cv2.COLOR_GRAY2RGB)


        results=modelYolo.predict(source=src)
        print('i am printing')
        print(results[0], len(results), results[0].boxes)
        print('done printing')
        # if len(results.boxes)==0:
        #     return ''
        for i,r in enumerate(results):
            plate=extract_plate(r.orig_img,r.boxes.xyxy.cpu().numpy().astype(int)[0])
            cv2.imwrite('just_plate.jpg',plate)
            plt.imshow(plate)
            plt.show()
            char=getcrop(plate)
            # for i in range(len(char)):
            #     char_re.append(255-fix_dimension(char[i]).astype(np.uint8))
            for i in range(len(char)):
                plt.subplot(1, len(char), i+1)
                plt.imshow(char[i], cmap='gray')
                plt.axis('on')
            plt.show()
            break

        output=[]

        mask = [1,1,0,0,1,1,0,0,0,0]
        for i in range(len(char)):
            c=inference(modelVGG,char[i], 'num' if mask[i] == 1 else 'alpha')
            output.append(c)
        print(''.join(output))
        return ''.join(output)
    except IndexError:
        print('oopsie')
        return 'yikers'


yolo, vgg = loadmodel()
predictimg(yolo, vgg)

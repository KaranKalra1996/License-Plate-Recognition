from imageai.Detection import VideoObjectDetection
import os
import numpy as np
import cv2
import imutils
import pandas as pd
import time
import matplotlib.pyplot as plt
import pytesseract

def resize_image(img):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return img


def show_image(img):
    cv2.imshow("image window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 'Imaged displayed successfully'


def detecting_licenseplate(image):
    image = imutils.resize(image, width=500)
    # cv2.imshow("Original Image", image)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1 - Grayscale Conversion", gray)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # cv2.imshow("2 - Bilateral Filter", gray)

    edged = cv2.Canny(gray, 170, 200)
    # cv2.imshow("4 - Canny Edges", edged)
    # ret,thresh = cv2.threshold(edged.copy(),127,255,0)
    # print(len(cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)))

    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    count = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            # cv2.imshow('NumberPlateCnt',NumberPlateCnt)
            # cv2.waitKey(0)
            break

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    #     cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
    # croping numberplate
    #     show_image(new_image)
    #     new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    #     new_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_numberplate = np.where(new_image > 0.8)
    last_index, = img_numberplate[0].shape
    x1, y1 = img_numberplate[1].min(), img_numberplate[0].min()
    x2, y2 = img_numberplate[1].max(), img_numberplate[0].max()
    #     print("x1 = "+str(x1)+"x2 = "+str(x2)+"y1 = "+str(y1)+"y2 = "+str(y2))
    new_image = new_image[y1:y2, x1:x2]
    #     show_image(new_image)
    return new_image

def image_cleaning(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=500)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img


def reading_licenseplate(img):
    config = ('-l eng --oem 1 --psm 3')
    license_number = pytesseract.image_to_string(img, config=config)
    return license_number


def saving_image(img, name):
    cv2.imwrite(name, img)
    print('Imaged saved.')
    return True

#path_of_image = 'G:/LPR Project/Labled data/34.jpeg'
def working(something):
    license_number = "not able to detect."
    try:
        # img = cv2.imread(path_of_image)
        #         show_image(img)
        img = detecting_licenseplate(something)
        #     plt.imshow(img)
        #         print("shape : "+str(img.shape))
        show_image(img)
        cv2.imwrite('numberplate.jpg',img)
        img = resize_image(img)

        img = image_cleaning(img)
        print("Number from license plate is : - ")
        license_number = reading_licenseplate(img)
    #         arr[x] = license_number
    except Exception as err:
        print("Count Detect the number plate  ", err)

    print(license_number)


# cam = cv2.VideoCapture(0)
something = None
execution_path = os.getcwd()
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
custom = detector.CustomObjects(car=True)
detector.loadModel(detection_speed='normal')

def forFrame(frame_number, output_array, output_count,returned_frame):
     if output_array != []:
        # print("FOR FRAME " , frame_number)
        # print("Output for each object : ", output_array)
        # print("Output count for unique objects : ", output_count)
        print("test")
        cv2.line(returned_frame,(0,550),(1700,550),color=cv2.COLOR_BAYER_BG2RGB_VNG,thickness=4)

        #cv2.imshow("Any",returned_frame)
        print((output_array[0]['box_points'][3]+output_array[0]['box_points'][1])/2)
        if (output_array[0]['box_points'][3]+output_array[0]['box_points'][1])/2 in range(330,360):
            cv2.line(returned_frame, (0, 550), (1300, 550), color=cv2.COLOR_BAYER_GB2RGB_EA, thickness=4)
            cv2.imshow("line_img",returned_frame)
            y = output_array[0]['box_points'][1]
            x = output_array[0]['box_points'][0]
            w = output_array[0]['box_points'][2]
            h = output_array[0]['box_points'][3]
            crop = returned_frame[y:h, x:w]
            something = crop
            print('saving image')
            cv2.imwrite('croppedimage.jpg',something)
            working(something)

            cv2.imshow('Image', crop)
            cv2.imwrite('None.jpg',something)
            breakpoint()
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        plt.title("Frame : " + str(frame_number))
        plt.axis("off")
        plt.imshow(returned_frame, interpolation="none")

        # plt.subplot(1, 2, 2)
        # plt.title("Analysis: " + str(frame_number))
        # plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

        plt.pause(0.000001)

detector.detectCustomObjectsFromVideo(custom_objects=custom,
                                                    input_file_path=os.path.join(execution_path,"99.mp4"),
                                                    save_detected_video=True,
                                                    per_frame_function=forFrame,
                                                    output_file_path=os.path.join(execution_path, "samplevideo"),
                                                    minimum_percentage_probability=99,
                                                    frames_per_second=45,
                                                    return_detected_frame=True
                                                    )
    # for eachObject, eachObjectPath in zip(detections, object_path):
    #     box_points = eachObject['box_points']
    #     if (box_points[1]+box_points[3])/2 == 300:
    #         cv2.imshow('image',eachObjectPath)


# !/usr/bin/env python
# coding: utf-8

# In[1]:



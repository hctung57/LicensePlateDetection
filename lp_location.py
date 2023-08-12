from multiprocessing.sharedctypes import Value
from tkinter import Frame
import cv2
import torch
from PIL import Image
import function.helper as helper
import function.utils_rotate as utils_rotate

class LP_Detect_output():
    def __init__(self, image, x_location, y_location):
        self.image = image
        self.x = x_location
        self.y = y_location

class LicensePlateDetection:
    def __init__(self,path_image):
        self.path_to_image = path_image
        self.model, self.LP_ocr_model = self.load_model()
        self.classes = self.model.names

    def load_model(self):
        print("Loading model...")
        LP_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\document\Project 1\License_Plate/runs\exp3\weights/best.pt', force_reload=True)
        LP_ocr_model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\document\Project 1\License_Plate/runs\exp3\weights\LP_ocr.pt', force_reload=True)
        return LP_detect_model, LP_ocr_model

    def load_image(self):

        IMG = cv2.imread(self.path_to_image)
        x_shape = IMG.shape[1]
        y_shape = IMG.shape[0]
        imgs = [IMG]
        return imgs, x_shape, y_shape

    def LicensePlateDetect(self,image):
        results = self.model(image)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    def plot_boxes(self, results, frame, x_shape, y_shape):
        lp = []
        labels, cord = results
        n = len(labels)
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # cv2.putText(frame, str(self.classes[0]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                value = LP_Detect_output(frame[y1:y2, x1:x2],x1,y1)
                lp.append(value)
        return lp ,frame

    def __call__(self):
        image, x_shape, y_shape = self.load_image()
        results = self.LicensePlateDetect(image)
        list_lp_image, Frame = self.plot_boxes(results, image[0], x_shape, y_shape)
        lp = ""
        list_read_plates = set()
        for plate in list_lp_image:
            for cc in range(0,2):
                flag = 0
                for ct in range(0,2):
                    lp = helper.read_plate(self.LP_ocr_model, utils_rotate.deskew(plate.image, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(Frame, lp, (int(plate.x), int(plate.y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
                        flag = 1
                        break
                if flag == 1:
                    break
        Frame = cv2.resize(Frame, (int(x_shape/2), int(y_shape/2)),interpolation = cv2.INTER_NEAREST)
        cv2.imwrite('Test/result/6.2.png',Frame)
        cv2.imshow('frame', Frame)
        cv2.waitKey()
        cv2.destroyAllWindows()  
        print(list_read_plates)
        
detect = LicensePlateDetection('train_data/images/7.jpg')
detect()

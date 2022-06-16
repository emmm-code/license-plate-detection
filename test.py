# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html
import cv2 as cv
import cv2
import numpy as np
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class YOLO_License_plate_Detector(object):
    # 正规马赛克
    def do_mosaic(self, frame, x, y, w, h, neighbor=9):
        """
        :param rgb_img
        :param int x :  马赛克左顶点
        :param int y:  马赛克左顶点
        :param int w:  马赛克宽
        :param int h:  马赛克高
        :param int neighbor:  马赛克每一块的宽
        """
        for i in range(0, h, neighbor):
            for j in range(0, w, neighbor):
                rect = [j + x, i + y]
                color = frame[i + y][j + x].tolist()  # 关键点1 tolist
                left_up = (rect[0], rect[1])
                x2 = rect[0] + neighbor - 1  # 关键点2 减去一个像素
                y2 = rect[1] + neighbor - 1
                if x2 > x + w:
                    x2 = x + w
                if y2 > y + h:
                    y2 = y + h
                right_down = (x2, y2)
                cv2.rectangle(frame, left_up, right_down, color, -1)  # 替换为为一个颜值值

        return frame

    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        # cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        # image,矩形的起始坐标，矩形的结束目标，color,thickness矩形边框线的粗细像素
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        label = '%.2f' % conf  # 保留conf小数后两位

        # Load names of classes读取类别名
        classesFile = "classes.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        # with open(os.path.join(BASE_DIR, classesFile), 'rt') as f:
        #     classes = f.read().rstrip('\n').split('\n')
        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s: %s' % (classes[classId], label)
            # print(label) # License Plate: 0.98

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(
            1.5 * labelSize[0]), top + baseLine), (255, 0, 255), cv.FILLED)
        # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
        # 1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

    def rectangle(self, image, mosaic_level):
        success = False
        # Initialize the parameters
        confThreshold = 0.5  # Confidence threshold
        nmsThreshold = 0.4  # Non-maximum suppression threshold

        inpWidth = 416  # 608     # Width of network's input image
        inpHeight = 416  # 608     # Height of network's input image

        # Give the configuration and weight files for the model and load the network using them.模型参数文件
        model_configuration = "darknet-yolov3.cfg"
        model_weights = "weights/model.weights"
        modelConfiguration = os.path.join(BASE_DIR, model_configuration)
        modelWeights = os.path.join(BASE_DIR, model_weights)
        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        # 使用CPU计算
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # imgname = image.split('/')[-1]
        img = Image.open(image)  # 按照RGB方式读取图像
        # 利用img = Image.open(image_path)打开的图片类型是PIL类型，将PIL类型转换为numpy类型
        frame = np.array(img)
        # print(frame)
        # 对图像进行预处理，返回一个4通道的blob,用于神经网络的输入；[0, 0, 0]为各通道BGR减去的值
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(self.getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # print(frameWidth, frameHeight) # 为输入图片尺寸720 1160

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            # print("out.shape : ", out.shape)
            for detection in out:
                # if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)  # classId为scores最大值的索引
                # if scores[classId]>confThreshold:
                confidence = scores[classId]  # 置信度
                if detection[4] > confThreshold:
                    print(detection[4], " - ", scores[classId],
                          " - th : ", confThreshold)
                    print(detection)
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        # print('indices为', indices)
        for i in indices:
            # i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left,
                     top, left + width, top + height)
            self.do_mosaic(frame, left, top, width, height, int(mosaic_level * 0.2 * width))
            success = True
        # mosaic_image_name = image_output_path + str(imgname)
        # cv.imwrite(mosaic_image_name, frame.astype(np.uint8))  # cv.imwrite只能保存BGR图像
        return frame, success

    # def detector(self, image_input_path, image_output_path):
    #     for filename in os.listdir(image_input_path):
    #         image_path = os.path.join(image_input_path, filename)
    #         img = Image.open(image_path)  # 按照RGB方式读取图像
    #         # 利用img = Image.open(image_path)打开的图片类型是PIL类型，将PIL类型转换为numpy类型
    #         frame = np.array(img)
    #         # print(frame)
    #         # 对图像进行预处理，返回一个4通道的blob,用于神经网络的输入；[0, 0, 0]为各通道BGR减去的值
    #         inpWidth = 416  # 608     # Width of network's input image
    #         inpHeight = 416  # 608     # Height of network's input image
    #         blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    #         # Sets the input to the network
    #         net.setInput(blob)
    #         # Runs the forward pass to get output of the output layers
    #         outs = net.forward(getOutputsNames(net))
    #         # Remove the bounding boxes with low confidence
    #         postprocess(frame, outs)
    #         mosaic_image_name = './output_image/image_mosaic/' + str(filename)
    #         cv.imwrite(mosaic_image_name, frame.astype(np.uint8))  # cv.imwrite只能保存BGR图像






if __name__ == '__main__':
    detector = YOLO_License_plate_Detector()
    path = 'test_input/01-90_88-214&487_400&552-397&550_213&552_204&485_388&483-0_0_33_30_26_21_30-144-12.jpg'
    output_path = 'C:/Users\qtyxyg/caoweixin\python_project/repositry_git/yolo-license-plate-detection/test_output/'
    detector.rectangle(image=path, mosaic_level=0.6)

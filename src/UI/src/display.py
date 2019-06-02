#!/usr/bin/python
from PySide2 import QtCore, QtWidgets, QtGui
import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge
import sys
import numpy as np
import time
import message_filters

class RGBD_display(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.bridge = cv_bridge.CvBridge()
        self.refPt = []
        self.ROI_flag = 0
        self.UI_close_flag = 0

        self.color_sub = message_filters.Subscriber('camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('camera/aligned_depth_to_color/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.image_callback)

        self.setup_ui()


    def image_callback(self, msg_color, msg_depth):
        self.RGB_image = self.bridge.imgmsg_to_cv2(msg_color, desired_encoding='bgr8')
        
        self.depth_image = self.bridge.imgmsg_to_cv2(msg_depth)
        # self.depth_image = self.RGB_image.astype(np.float)
        # self.depth_image_max = np.amax(self.RGB_image)
        # self.RGB_image = 255*(self.RGB_image - 200) / (np.amax(self.RGB_image) - 200)
        # self.RGB_image = self.RGB_image.astype(np.uint8)
        
        if self.UI_close_flag == 0:
            if len(self.refPt) != 2:
                cv2.imshow("ROI", self.RGB_image)
                cv2.imshow("ROI_depth", self.depth_img_norm(self.depth_image))
            else:
                cv2.imshow("ROI", self.RGB_image[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]])
                cv2.imshow("ROI_depth", self.depth_img_norm(self.depth_image[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]))
            if self.ROI_flag:
                cv2.imshow("Select ROI", self.im_clone)
                if len(self.refPt) == 2:
                    cv2.destroyWindow("Select ROI")
                    print('select ROI finish')
                    self.ROI_flag = 0
            cv2.waitKey(3)    

    def depth_img_norm(self, depth_image):
        depth_image = depth_image.astype(np.float)
        print(np.amax(depth_image))
        depth_image = np.maximum(255*(depth_image - 300) / (np.amax(depth_image) - 300), 0)
        depth_image = depth_image.astype(np.uint8)
        return depth_image



    def setup_ui(self):
        # self.color_image_label = QtWidgets.QLabel()

        self.ROI_button = QtWidgets.QPushButton("Select ROI!")
        self.ROI_button.clicked.connect(self.btn_select_ROI)
        self.start_button = QtWidgets.QPushButton("Start!")
        self.start_button.clicked.connect(self.btn_start)
        self.stop_button = QtWidgets.QPushButton("Stop!")
        self.stop_button.clicked.connect(self.btn_stop)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.ROI_button)
        self.main_layout.addWidget(self.stop_button)

        self.setLayout(self.main_layout)
    
    def btn_select_ROI(self):
        print('please select ROI:')
        self.refPt = []
        self.im_clone = self.RGB_image.copy()
        cv2.namedWindow("Select ROI", 1)
        cv2.setMouseCallback("Select ROI", self.click_and_crop)
        self.ROI_flag = 1
        
    def btn_start(self):
        self.UI_close_flag = 0
        self.refPt = []
        
    
    def btn_stop(self):
        if self.ROI_flag:
            print("please finish ROI selection first")
        else:
            self.UI_close_flag = 1
            self.refPt = []
            cv2.destroyWindow("ROI")
            cv2.destroyWindow("ROI_depth")
            print('stop display')
        
    
    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt.append((x, y))

    def closeEvent(self, event):
        self.UI_close_flag = 1
        print('widge close')


if __name__ == "__main__":
    rospy.init_node('display')
    app = QtWidgets.QApplication(sys.argv)
    widget = RGBD_display()
    widget.show()
    sys.exit(app.exec_())
 
#!/usr/bin/python
from PySide2 import QtCore, QtWidgets, QtGui
import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge
import sys
import numpy as np
import time
import message_filters

from scipy.io import loadmat
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from mayavi import mlab
import math
import pickle, copy


class Crab_3D(object):
    def __init__(self, root_3D = '..//res//Crab_model_rotate.mat', root_keypoint='..//res//keypoints.pkl'):
        self.Crab_3D = loadmat(root_3D)
        self.Crab_3D = self.Crab_3D['Im_BW']
        self.Crab_2D_proj = self.Projection(self.Crab_3D)

        if root_keypoint == None:
            self.draw_keypoint()
        else:
            with open(root_keypoint, 'rb') as f:
                self.keypoints = pickle.load(f)
                # self.vis_keypoint(self.Crab_2D_proj, self.keypoints)

        

    # def _display_3D(self, crab_3D):
    #     mlab.figure('crab_shell')
    #     mlab.contour3d(crab_3D)
    #     mlab.show()

    def Projection(self, crab_3D):
        crab_2D = 255*np.sum(crab_3D, axis=0)
        crab_2D = crab_2D.astype('uint8')
        
        # (height, width) = self.Crab_2D_proj.shape
        # img_center = (width/2, width/2)
        # Rotate_base = cv2.getRotationMatrix2D(img_center, 90, 1) 
        # self.Crab_2D_proj = cv2.warpAffine(self.Crab_2D_proj, Rotate_base, (height, width))
        # img_center = (height/2, width/2)
        # Rotate_base = cv2.getRotationMatrix2D(img_center, 180, 1) 
        # self.Crab_2D_proj = cv2.warpAffine(self.Crab_2D_proj, Rotate_base, (height, width))
        
        # cv2.imshow('binary_2D', crab_2D)
        # cv2.waitKey(10000)
        return crab_2D

    def draw_keypoint(self):
        print('please select keypoints:')
        self.im_display = np.zeros((self.Crab_2D_proj.shape[0], self.Crab_2D_proj.shape[1], 3), dtype=np.uint8)
        self.im_display[:,:,0] = self.Crab_2D_proj.copy()
        self.im_display[:,:,1] = self.Crab_2D_proj.copy()
        self.im_display[:,:,2] = self.Crab_2D_proj.copy()
        im_clone = self.im_display.copy()
        cv2.namedWindow("Select keypoints", 1)
        cv2.setMouseCallback("Select keypoints", self.click_callback)
        self.color=(0, 0, 0)
        self.keypoints = {"knckle_position": [], "center_position": []}
        print "press 'r' for reset, 'k' for knuckle (left first), 'c' for center point, 'esc' to exit"
        while(1):
            cv2.imshow("Select keypoints", self.im_display)
            k = cv2.waitKey(33)
            if k == 27:
                break
            elif k == ord('r'):
                self.color=(0, 0, 0)
                self.im_display = im_clone.copy()
            elif k == ord('k'):
                self.color=(255,0,0)
            elif k == ord('c'):
                self.color=(0,0,255)
        with open('..//res//keypoints.pkl', 'wb') as f:
            pickle.dump(self.keypoints, f)

    def vis_keypoint(self, Crab_2D_proj, keypoints):
        self.im_display = np.zeros((Crab_2D_proj.shape[0], Crab_2D_proj.shape[1], 3), dtype=np.uint8)
        self.im_display[:,:,0] = Crab_2D_proj.copy()
        self.im_display[:,:,1] = Crab_2D_proj.copy()
        self.im_display[:,:,2] = Crab_2D_proj.copy()
        cv2.circle(self.im_display, (keypoints["center_position"][0][0], keypoints["center_position"][0][1]), 3, color=(0,0,255), thickness=-1)
        cv2.circle(self.im_display, (keypoints["knckle_position"][0][0], keypoints["knckle_position"][0][1]), 3, color=(255,0,0), thickness=-1)
        cv2.circle(self.im_display, (keypoints["knckle_position"][1][0], keypoints["knckle_position"][1][1]), 3, color=(255,0,0), thickness=-1)
        cv2.namedWindow("Visualize keypoints", 1)
        cv2.imshow("Visualize keypoints", self.im_display)
        cv2.waitKey()
        cv2.destroyWindow("Visualize keypoints")

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.color != (0, 0, 0):
                cv2.circle(self.im_display, (x, y), 3, color=self.color, thickness=-1)
                if self.color == (0, 0, 255):
                    if self.keypoints["center_position"].__len__() == 0:
                        self.keypoints["center_position"].append([x, y])
                    else:
                        print "center point has been selected, please press 'r' to reset the selection"
                if self.color == (255, 0, 0):
                    if self.keypoints["knckle_position"].__len__() <= 1:
                        self.keypoints["knckle_position"].append([x, y])
                    else:
                        print "two knuckle points has been selected, please press 'r' to reset the selection"

    def model_transform_para(self, test_keypoints):
        ## please follow the order of zoom -> rotate -> translate
        template_diff_x = self.keypoints["knckle_position"][1][0] - self.keypoints["knckle_position"][0][0]
        template_diff_y = self.keypoints["knckle_position"][1][1] - self.keypoints["knckle_position"][0][1]
        template_knuckle_angle = math.atan2(template_diff_y, template_diff_x)
        template_knuckle_distance = math.sqrt(template_diff_x*template_diff_x + template_diff_y*template_diff_y)

        test_diff_x = test_keypoints["knckle_position"][1][0] - test_keypoints["knckle_position"][0][0]
        test_diff_y = test_keypoints["knckle_position"][1][1] - test_keypoints["knckle_position"][0][1]
        test_knuckle_angle = math.atan2(test_diff_y, test_diff_x)
        test_knuckle_distance = math.sqrt(test_diff_x*test_diff_x + test_diff_y*test_diff_y)


        scale_ratio = test_knuckle_distance / template_knuckle_distance
        transformed_knuckle_left = self.point_scaling(self.keypoints["knckle_position"][0], scale_ratio)
        transformed_knuckle_right = self.point_scaling(self.keypoints["knckle_position"][1], scale_ratio)
        
        
        rotate_angle = -(test_knuckle_angle - template_knuckle_angle) * 180 / math.pi
        transformed_knuckle_left = self.point_rotation(transformed_knuckle_left, rotate_angle, scale_ratio)
        transformed_knuckle_right = self.point_rotation(transformed_knuckle_right, rotate_angle, scale_ratio)

        print transformed_knuckle_left, transformed_knuckle_right


        transformed_knuckle_center = [(transformed_knuckle_left[0] + transformed_knuckle_right[0]) / 2.0, (transformed_knuckle_left[1] + transformed_knuckle_right[1]) / 2.0]
        test_knuckle_center = [(test_keypoints["knckle_position"][1][0] + test_keypoints["knckle_position"][0][0]) / 2.0, (test_keypoints["knckle_position"][1][1] + test_keypoints["knckle_position"][0][1]) / 2.0]
        translation_shift = [int(test_knuckle_center[0] - transformed_knuckle_center[0]), int(test_knuckle_center[1] - transformed_knuckle_center[1])]
        print transformed_knuckle_center, test_knuckle_center
        return scale_ratio, rotate_angle, translation_shift
    
    def model3_zoom(self, scale_ratio, model_in):
        model_out = ndimage.zoom(model_in, scale_ratio, mode='nearest')
        return model_out
    
    def model3_rotate(self, rotate_angle, model_in):
        model_out = ndimage.rotate(model_in, rotate_angle, (1, 2)) 
        return model_out

    def model3_translate(self, translation_shift, model_in):
        new_shape = (model_in.shape[0], model_in.shape[1] + max(translation_shift[1], 0), model_in.shape[2] + max(translation_shift[0], 0))
        model_out = np.zeros(new_shape, dtype=np.uint8)
        model_out[0:model_in.shape[0], 0:model_in.shape[1], 0:model_in.shape[2]] = model_in
        model_out = ndimage.shift(model_out, [0, translation_shift[1], translation_shift[0]], mode='nearest')
        return model_out
    
    def model2_zoom(self, scale_ratio, image_in):
        image_out = ndimage.zoom(image_in, scale_ratio, mode='nearest')
        return image_out
    
    def model2_rotate(self, rotate_angle, image_in):
        image_out = ndimage.rotate(image_in, rotate_angle) 
        return image_out
    
    def model2_translate(self, translation_shift, image_in):
        new_shape = (image_in.shape[0] + max(translation_shift[1], 0), image_in.shape[1] + max(translation_shift[0], 0))
        image_out = np.zeros(new_shape, dtype=np.uint8)
        image_out[0:image_in.shape[0], 0:image_in.shape[1]] = image_in
        image_out = ndimage.shift(image_out, [translation_shift[1], translation_shift[0]], mode='nearest')
        return image_out
    
    def point_scaling(self, point_in, scaling_ratio):
        point_out = copy.deepcopy(point_in) 
        point_out[0] = int(point_in[0] * scaling_ratio)
        point_out[1] = int(point_in[1] * scaling_ratio)
        return point_out
    
    def point_rotation(self, point_in, rotation_angle_D, scaling_ratio=1):
    ## rotation center is always the center of self.Crab_2D_proj
        point_in[0] = float(point_in[0])
        point_in[1] = float(point_in[1])
        rotation_angle = rotation_angle_D * math.pi / 180

        transform_center = [self.Crab_2D_proj.shape[1]/2 * scaling_ratio, self.Crab_2D_proj.shape[0]/2 * scaling_ratio]
        new_transform_center = copy.deepcopy(transform_center)
        new_transform_center[0] = transform_center[0] * abs(math.cos(rotation_angle)) + (transform_center[1]) * abs(math.sin(rotation_angle))
        new_transform_center[1] = transform_center[0] * abs(math.sin(rotation_angle)) + (transform_center[1]) * abs(math.cos(rotation_angle))

        point_out = copy.deepcopy(point_in) 
        point_out_temp = copy.deepcopy(point_in) 
        point_out_temp[0] = point_out[0] - transform_center[0] 
        point_out_temp[1] = point_out[1] - transform_center[1] 
        ## notice that the y value needs to '-' to satisfy the right hand rule
        point_out[0] = point_out_temp[0] * math.cos(rotation_angle) - (-point_out_temp[1])*math.sin(rotation_angle)
        point_out[1] = point_out_temp[0] * math.sin(rotation_angle) + (-point_out_temp[1])*math.cos(rotation_angle)

        point_out[0] = int(point_out[0] + new_transform_center[0])
        point_out[1] = int(-point_out[1] + new_transform_center[1])

        return point_out

    def point_translation(self, point_in, translation_shift):
        point_out = copy.deepcopy(point_in) 
        point_out[0] = point_out[0] + translation_shift[0]
        point_out[1] = point_out[1] + translation_shift[1]
        return point_out

    def model3D2depth(self, model3D, maximum_height=[], minimum_height=[]):
        if maximum_height == []:
            maximum_height = model3D.shape[0]
        if minimum_height == []:
            minimum_height = 0

        Height_map = np.zeros((model3D.shape[1], model3D.shape[2]), dtype=float)
        for i in range(model3D.shape[1]):
            for j in range(model3D.shape[2]):
                Height_vector = model3D[:, i, j]
                Height_idx = 0
                if np.sum(Height_vector) > 0:
                    Height_idx = np.argwhere(Height_vector)
                    Height_idx = model3D.shape[0] - np.min(Height_idx)
                if Height_idx >= maximum_height:
                    Height_map[i, j] = 255
                elif Height_idx <= minimum_height:
                    Height_map[i, j] = 0
                else:
                    Height_map[i, j] = (255*(Height_idx - minimum_height)) / (1.0*(maximum_height - minimum_height))

        Height_map = Height_map.astype(np.uint8)
        return Height_map


        

class RGBD_display(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.bridge = cv_bridge.CvBridge()
        self.refPt = []
        self.knuckle_position = []
        self.ROI_flag = 0
        self.UI_close_flag = 0
        self.generate_flag = 0
        self.knuckle_select_flag = 0

        self.Crab_model = Crab_3D()
        self.color_sub = message_filters.Subscriber('camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('camera/aligned_depth_to_color/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.image_callback)

        self.setup_ui()


    def image_callback(self, msg_color, msg_depth):
        self.RGB_image = self.bridge.imgmsg_to_cv2(msg_color, desired_encoding='bgr8')
        self.depth_image = self.bridge.imgmsg_to_cv2(msg_depth)
        
        if self.UI_close_flag == 0:
            if len(self.refPt) != 2:
                cv2.imshow("ROI", self.RGB_image)
                cv2.imshow("ROI_depth", self.depth_img_norm(self.depth_image))
            else:
                cv2.imshow("ROI", self.RGB_image[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]])
                cv2.imshow("ROI_depth", self.depth_img_norm(self.depth_image[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]))
            if self.ROI_flag:
                cv2.imshow("Select ROI", self.im_ROI_select)
                if len(self.refPt) == 2:
                    cv2.waitKey()
                    cv2.destroyWindow("Select ROI")
                    print('select ROI finish')
                    self.ROI_flag = 0
            
            if self.knuckle_select_flag:
                cv2.imshow("Select Knuckles", self.im_knuckle_select)
                if len(self.knuckle_position) != 0:
                    for i in range(self.knuckle_position.__len__()):
                        cv2.circle(self.im_knuckle_select, (self.knuckle_position[i][0], self.knuckle_position[i][1]), 3, color=(255,0,0), thickness=-1)
                if len(self.knuckle_position) == 2:
                    cv2.waitKey(3) 
                    cv2.circle(self.im_knuckle_select, (self.knuckle_position[0][0], self.knuckle_position[0][1]), 3, color=(255,0,0), thickness=-1)
                    cv2.circle(self.im_knuckle_select, (self.knuckle_position[1][0], self.knuckle_position[1][1]), 3, color=(255,0,0), thickness=-1)
                    cv2.imshow("Select Knuckles", self.im_knuckle_select)
                    cv2.waitKey()
                    cv2.destroyWindow("Select Knuckles")
                    print('select knuckle finish')
                    self.knuckle_select_flag = 0
            
            if self.generate_flag:
                cv2.imshow("Generated_depth", self.im_display)


            cv2.waitKey(3)    

    def depth_img_norm(self, depth_image):
        depth_image = depth_image.astype(np.float)
        print(np.amax(depth_image))
        depth_image = np.maximum(255*(depth_image - 240) / (np.amax(depth_image) - 240), 0)
        depth_image = depth_image.astype(np.uint8)
        return depth_image


    def setup_ui(self):
        self.ROI_button = QtWidgets.QPushButton("Select ROI!")
        self.ROI_button.clicked.connect(self.btn_select_ROI)
        self.start_button = QtWidgets.QPushButton("Start!")
        self.start_button.clicked.connect(self.btn_start) 
        self.stop_button = QtWidgets.QPushButton("Stop!")
        self.stop_button.clicked.connect(self.btn_stop)
        self.select_knuckle_button = QtWidgets.QPushButton("Select Knuckle Position!")
        self.select_knuckle_button.clicked.connect(self.btn_select_knuckle)
        self.depth_model_button = QtWidgets.QPushButton("Generate depth model!")
        self.depth_model_button.clicked.connect(self.btn_generate_depth_model)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.ROI_button)
        self.main_layout.addWidget(self.stop_button)
        self.main_layout.addWidget(self.select_knuckle_button)
        self.main_layout.addWidget(self.depth_model_button)

        self.setLayout(self.main_layout)
    
    def btn_select_ROI(self):
        print('please select ROI:')
        self.refPt = []
        self.im_ROI_select = self.RGB_image.copy()
        cv2.namedWindow("Select ROI", 1)
        cv2.setMouseCallback("Select ROI", self.click_and_crop)
        self.ROI_flag = 1
    
    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt.append((x, y))
    
    def btn_select_knuckle(self):
        print('please select knuckles:')
        self.knuckle_position = []
        self.im_knuckle_select = self.RGB_image.copy()
        cv2.namedWindow("Select Knuckles", 1)
        cv2.setMouseCallback("Select Knuckles", self.click_knckle)
        self.knuckle_select_flag = 1
    
    def click_knckle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.knuckle_position.append((x, y))
        
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

    def btn_generate_depth_model(self):
        self.generate_flag = 0
        if self.knuckle_position.__len__() != 2:
            print('please select knuckle positions correctly')
        else:
            print self.knuckle_position
            test_keypoints = {"knckle_position": [], "center_position": []}
            test_keypoints["knckle_position"].append(self.knuckle_position[0])
            test_keypoints["knckle_position"].append(self.knuckle_position[1])
            test_keypoints["center_position"].append((0,0))
            scale_ratio, rotate_angle, translation_shift = self.Crab_model.model_transform_para(test_keypoints)
            print scale_ratio, rotate_angle, translation_shift

            Height_map = self.Crab_model.model3D2depth(self.Crab_model.Crab_3D)
            transformed_Height_map = self.Crab_model.model2_zoom(scale_ratio, Height_map)
            transformed_Height_map = self.Crab_model.model2_rotate(rotate_angle, transformed_Height_map)
            transformed_Height_map = self.Crab_model.model2_translate(translation_shift, transformed_Height_map)
            
            transformed_knuckle_left = self.Crab_model.point_scaling(self.Crab_model.keypoints["knckle_position"][0], scale_ratio)
            transformed_knuckle_right = self.Crab_model.point_scaling(self.Crab_model.keypoints["knckle_position"][1], scale_ratio)
            transformed_center = self.Crab_model.point_scaling(self.Crab_model.keypoints["center_position"][0], scale_ratio)
            transformed_knuckle_left = self.Crab_model.point_rotation(transformed_knuckle_left, rotate_angle, scale_ratio)
            transformed_knuckle_right = self.Crab_model.point_rotation(transformed_knuckle_right, rotate_angle, scale_ratio)
            transformed_center = self.Crab_model.point_rotation(transformed_center, rotate_angle, scale_ratio)

            print transformed_knuckle_left, transformed_knuckle_right


            transformed_knuckle_left = self.Crab_model.point_translation(transformed_knuckle_left, translation_shift)
            transformed_knuckle_right = self.Crab_model.point_translation(transformed_knuckle_right, translation_shift)
            transformed_center = self.Crab_model.point_translation(transformed_center, translation_shift)

            transformed_keypoints = {"knckle_position": [], "center_position": []}
            transformed_keypoints["knckle_position"].append(transformed_knuckle_left)
            transformed_keypoints["knckle_position"].append(transformed_knuckle_right)
            transformed_keypoints["center_position"].append(transformed_center)


            self.im_display = np.zeros((transformed_Height_map.shape[0], transformed_Height_map.shape[1], 3), dtype=np.uint8)
            self.im_display[:,:,0] = transformed_Height_map.copy()
            self.im_display[:,:,1] = transformed_Height_map.copy()
            self.im_display[:,:,2] = transformed_Height_map.copy()


            cv2.circle(self.im_display, (transformed_keypoints["center_position"][0][0], transformed_keypoints["center_position"][0][1]), 3, color=(0,0,255), thickness=-1)
            cv2.circle(self.im_display, (transformed_keypoints["knckle_position"][0][0], transformed_keypoints["knckle_position"][0][1]), 3, color=(255,0,0), thickness=-1)
            cv2.circle(self.im_display, (transformed_keypoints["knckle_position"][1][0], transformed_keypoints["knckle_position"][1][1]), 3, color=(255,0,0), thickness=-1)

            self.generate_flag = 1



    def closeEvent(self, event):
        self.UI_close_flag = 1
        print('Qapp close')


if __name__ == "__main__":
    rospy.init_node('display')
    app = QtWidgets.QApplication()
    widget = RGBD_display()
    widget.show()
    sys.exit(app.exec_())
 
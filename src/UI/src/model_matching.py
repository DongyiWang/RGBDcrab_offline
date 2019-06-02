#!/usr/bin/python
from scipy.io import loadmat
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import numpy as np
import cv2, math
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
        test_keypoints = {"knckle_position": [], "center_position": []}
        test_keypoints["knckle_position"].append((200, 100))
        test_keypoints["knckle_position"].append((600, 120))
        test_keypoints["center_position"].append((0,0))
        self.model_transform_para(test_keypoints)
        

    def _display_3D(self, crab_3D):
        mlab.figure('crab_shell')
        mlab.contour3d(crab_3D)
        mlab.show()
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
        
        
        rotate_angle = (test_knuckle_angle - template_knuckle_angle) * 180 / math.pi
        transformed_knuckle_left = self.point_rotation(transformed_knuckle_left, rotate_angle, scale_ratio)
        transformed_knuckle_right = self.point_rotation(transformed_knuckle_right, rotate_angle, scale_ratio)


        transformed_knuckle_center = [(self.keypoints["knckle_position"][1][0] + self.keypoints["knckle_position"][0][0]) / 2.0, (self.keypoints["knckle_position"][1][1] + self.keypoints["knckle_position"][0][1]) / 2.0]
        test_knuckle_center = [(test_keypoints["knckle_position"][1][0] + test_keypoints["knckle_position"][0][0]) / 2.0, (test_keypoints["knckle_position"][1][1] + test_keypoints["knckle_position"][0][1]) / 2.0]
        translation_shift = [int(test_knuckle_center[0] - transformed_knuckle_center[0]), int(test_knuckle_center[1] - transformed_knuckle_center[1])]
        
        transformed_3D = ndimage.zoom(self.Crab_3D, scale_ratio, mode='nearest')
        transformed_3D = ndimage.rotate(transformed_3D, rotate_angle, (1, 2)) 
        print transformed_3D.shape



        # 
        # 
        # 
        # transformed_3D = ndimage.shift(transformed_3D, [0, translation_shift[0], translation_shift[1]], mode='nearest')
        


           

        # model translation

        
        

        transformed_keypoints = {"knckle_position": [], "center_position": []}
        transformed_keypoints["knckle_position"].append(transformed_knuckle_left)
        transformed_keypoints["knckle_position"].append(transformed_knuckle_right)
        transformed_keypoints["center_position"].append(test_keypoints["center_position"][0])

        
        transformed_2D = self.Projection(transformed_3D)
        self.vis_keypoint(transformed_2D, transformed_keypoints)


        cv2.imshow('transformed_2D', transformed_2D)
        cv2.waitKey(10000)
        # print translation_shift
    
    def point_scaling(self, point_in, scaling_ratio):
        point_out = copy.deepcopy(point_in) 
        point_out[0] = int(point_in[0] * scaling_ratio)
        point_out[1] = int(point_in[1] * scaling_ratio)
        return point_out
    
    def point_rotation(self, point_in, rotation_angle_D, scaling_ratio=1):
    ## rotation center is always the center of self.Crab_2D_proj
        rotation_angle = rotation_angle_D * math.pi / 180

        transform_center = [self.Crab_2D_proj.shape[1]/2 * scaling_ratio, self.Crab_2D_proj.shape[0]/2 * scaling_ratio]
        new_transform_center = copy.deepcopy(transform_center)
        new_transform_center[0] = transform_center[0] * math.cos(rotation_angle) + transform_center[1] * math.sin(rotation_angle)
        new_transform_center[1] = transform_center[1] * math.cos(rotation_angle) + transform_center[0] * math.sin(rotation_angle)

        point_out = copy.deepcopy(point_in) 

        point_out[0] = point_out[0] - transform_center[0] 
        point_out[1] = point_out[1] - transform_center[1] 
        ## notice that the y value needs to '-' to satisfy the right hand rule
        point_out[0] = point_out[0] * math.cos(rotation_angle) - (-point_out[1])*math.sin(rotation_angle)
        point_out[1] = point_out[0] * math.sin(rotation_angle) + (-point_out[1])*math.cos(rotation_angle)

        point_out[0] = int(point_out[0] + new_transform_center[0])
        point_out[1] = int(-point_out[1] + new_transform_center[1])

        return point_out


if __name__ == "__main__":

    Crab_model = Crab_3D()
    # Crab_model._display_3D(Crab_model.Crab_3D)    
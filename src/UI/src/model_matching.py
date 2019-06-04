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
        
        
        rotate_angle = (test_knuckle_angle - template_knuckle_angle) * 180 / math.pi
        transformed_knuckle_left = self.point_rotation(transformed_knuckle_left, rotate_angle, scale_ratio)
        transformed_knuckle_right = self.point_rotation(transformed_knuckle_right, rotate_angle, scale_ratio)


        transformed_knuckle_center = [(self.keypoints["knckle_position"][1][0] + self.keypoints["knckle_position"][0][0]) / 2.0, (self.keypoints["knckle_position"][1][1] + self.keypoints["knckle_position"][0][1]) / 2.0]
        test_knuckle_center = [(test_keypoints["knckle_position"][1][0] + test_keypoints["knckle_position"][0][0]) / 2.0, (test_keypoints["knckle_position"][1][1] + test_keypoints["knckle_position"][0][1]) / 2.0]
        translation_shift = [int(test_knuckle_center[0] - transformed_knuckle_center[0]), int(test_knuckle_center[1] - transformed_knuckle_center[1])]

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


        
        
        
                


# if __name__ == "__main__":

#     Crab_model = Crab_3D()


#     test_keypoints = {"knckle_position": [], "center_position": []}
#     test_keypoints["knckle_position"].append((500, 500))
#     test_keypoints["knckle_position"].append((600, 520))
#     test_keypoints["center_position"].append((0,0))
    
    
#     scale_ratio, rotate_angle, translation_shift = Crab_model.model_transform_para(test_keypoints)

#     Height_map = Crab_model.model3D2depth(Crab_model.Crab_3D)
#     transformed_Height_map = Crab_model.model2_zoom(scale_ratio, Height_map)
#     transformed_Height_map = Crab_model.model2_rotate(rotate_angle, transformed_Height_map)
#     transformed_Height_map = Crab_model.model2_translate(translation_shift, transformed_Height_map)


#     # transformed_2D = Crab_model.model2_zoom(scale_ratio, Crab_model.Crab_2D_proj)
#     # transformed_2D = Crab_model.model2_rotate(rotate_angle, transformed_2D)
#     # transformed_2D = Crab_model.model2_translate(translation_shift, transformed_2D)
    
#     # # transformed_3D = Crab_model.model3_zoom(scale_ratio, Crab_model.Crab_3D)
#     # # transformed_3D = Crab_model.model3_rotate(rotate_angle, transformed_3D)
#     # # transformed_3D = Crab_model.model3_translate(translation_shift, transformed_3D)
    


#     transformed_knuckle_left = Crab_model.point_scaling(Crab_model.keypoints["knckle_position"][0], scale_ratio)
#     transformed_knuckle_right = Crab_model.point_scaling(Crab_model.keypoints["knckle_position"][1], scale_ratio)
#     transformed_center = Crab_model.point_scaling(Crab_model.keypoints["center_position"][0], scale_ratio)
#     transformed_knuckle_left = Crab_model.point_rotation(transformed_knuckle_left, rotate_angle, scale_ratio)
#     transformed_knuckle_right = Crab_model.point_rotation(transformed_knuckle_right, rotate_angle, scale_ratio)
#     transformed_center = Crab_model.point_rotation(transformed_center, rotate_angle, scale_ratio)
#     transformed_knuckle_left = Crab_model.point_translation(transformed_knuckle_left, translation_shift)
#     transformed_knuckle_right = Crab_model.point_translation(transformed_knuckle_right, translation_shift)
#     transformed_center = Crab_model.point_translation(transformed_center, translation_shift)

#     transformed_keypoints = {"knckle_position": [], "center_position": []}
#     transformed_keypoints["knckle_position"].append(transformed_knuckle_left)
#     transformed_keypoints["knckle_position"].append(transformed_knuckle_right)
#     transformed_keypoints["center_position"].append(transformed_center)

#     # Crab_model.vis_keypoint(transformed_2D, transformed_keypoints)
#     Crab_model.vis_keypoint(transformed_Height_map, transformed_keypoints)
#     # Crab_model.vis_keypoint(transformed_Height_map, Crab_model.keypoints)

    
#     # Crab_model._display_3D(Crab_model.Crab_3D)    
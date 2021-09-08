import cv2
import torch
import numpy

def load_image(filename, data_type=torch.float32):
    color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
    h, w, c = color_img.shape
    color_data = color_img.astype(numpy.float32).transpose(2, 0, 1)
    return torch.from_numpy(
        color_data.reshape(1, c, h, w)        
    ).type(data_type) / 255.0

def crop_image(filename, data_type=torch.float32):  #to use the 640*360 pretrained model
    color_img = filename
    center_cropped_color_img = color_img[60:420, 0:640, :]
    h, w, c = center_cropped_color_img.shape
    color_data = center_cropped_color_img.astype(numpy.float32).transpose(2, 0, 1)
#    print("In crop image (w, h): ", w, ", ", h)
    return torch.from_numpy(
        color_data.reshape(1, c, h, w)
    ).type(data_type) / 255.0
    
# def crop_image(filename, data_type=torch.float32):  #to use the 640*360 pretrained model
#     color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
#     center_cropped_color_img = color_img[60:420, 0:640, :]
#     h, w, c = center_cropped_color_img.shape
#     color_data = center_cropped_color_img.astype(numpy.float32).transpose(2, 0, 1)
# #    print("In crop image (w, h): ", w, ", ", h)
#     return torch.from_numpy(
#         color_data.reshape(1, c, h, w)
#     ).type(data_type) / 255.0

def load_depth(filename, data_type=torch.float32, scale=0.0002):
    depth_img = filename
    depth_img = cv2.resize(depth_img, (640, 360)) # YIHENG ADD
    h, w = depth_img.shape
#    depth_data = (255-depth_img.astype(numpy.float32)) * scale / 100.
    depth_data = depth_img.astype(numpy.float32) * scale
#    print("$$$$$$$$$$$$$$LOADING DEPTH MAP$$$$$$$$$$$$$$$$$")
    # print(filename, depth_data[74,74]);
    # print(filename, depth_data[60,60]);
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)

# def load_depth(filename, data_type=torch.float32, scale=0.0002):
#     depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
#     depth_img = cv2.resize(depth_img, (640, 360)) # YIHENG ADD
#     h, w = depth_img.shape
# #    depth_data = (255-depth_img.astype(numpy.float32)) * scale / 100.
#     depth_data = depth_img.astype(numpy.float32) * scale
# #    print("$$$$$$$$$$$$$$LOADING DEPTH MAP$$$$$$$$$$$$$$$$$")
#     print(filename, depth_data[74,74]);
#     print(filename, depth_data[60,60]);
#     return torch.from_numpy(
#         depth_data.reshape(1, 1, h, w)        
#     ).type(data_type)

def crop_depth(filename, data_type=torch.float32, scale=0.0002):
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    center_cropped_depth_img = depth_img[60:420, 0:640]
 #  zzz change the code to the following, or it gives you 640 * 360
 #   center_cropped_depth_img = depth_img[:,:]
    h, w = center_cropped_depth_img.shape
    depth_data = center_cropped_depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)
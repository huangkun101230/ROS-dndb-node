import torch
import cv2
import numpy
import math

def save_image(filename, tensor, scale=1.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        cv2.imwrite(filename.replace("#", str(n)), array)

def save_image2(filename, tensor, image, scale=1.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        uvs = tensor[n, :, :, :].detach().cpu().numpy()
        d, w, h = uvs.shape
        _, channels, w2, h2 = image.shape
        assert(w == w2)
        assert(h == h2)
        assert(d == 2)
        destination_pixels = numpy.zeros((channels, w, h))
        source_pixels = image[0, :, :, :].detach().cpu().numpy()
        for target_x in range(w):
            for target_y in range(h):
                source_x = uvs[1][target_x][target_y]
                source_x = math.floor(source_x)
                source_x = source_x if source_x < w else w-1
                source_x = source_x if source_x >= 0 else 0
                source_y = uvs[0][target_x][target_y]
                source_y = math.floor(source_y)
                source_y = source_y if source_y < h else h-1
                source_y = source_y if source_y >= 0 else 0
                destination_pixels[:, target_x, target_y] = source_pixels[:, source_x, source_y]

        destination_pixels = destination_pixels.transpose(1, 2, 0) / scale
        cv2.imwrite(filename.replace("#", str(n)), destination_pixels)
def save_imageF(filename, tensor, image, scale=1.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        uvs = tensor[n, :, :, :].detach().cpu().numpy()
        # print("uvs:")
        # print(uvs)
        d, h, w = uvs.shape
        # print("d, h, w", d, h, w)
        _, channels, h2, w2 = image.shape
        assert(w == w2)
        assert(h == h2)
        assert(d == 2)
        destination_pixels = numpy.zeros((channels, h, w))
        source_pixels = image[0, :, :, :].detach().cpu().numpy()
        for s_x in range(w):
            for s_y in range(h):
                t_x = uvs[0][s_y][s_x]
                t_x = math.floor(t_x)
                t_x = t_x if t_x < w else w-1
                t_x = t_x if t_x >= 0 else 0
                t_y = uvs[1][s_y][s_x]
                t_y = math.floor(t_y)
                t_y = t_y if t_y < h else h-1
                t_y = t_y if t_y >= 0 else 0
                destination_pixels[:, t_y, t_x] = source_pixels[:, s_y, s_x]
        destination_pixels = destination_pixels.transpose(1, 2, 0) * 255
        cv2.imwrite(filename.replace("#", str(n)), destination_pixels)       
        
def save_depth(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        array = numpy.uint16(array)
        cv2.imwrite(filename.replace("#", str(n)), array)

# customised function
def return_depth(tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        array = numpy.uint16(array)
        return array

def save_depth2(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        b2, _, _ = array.shape
        for n2 in range(b2):
            array2 = array[n2] / scale
            array2 = numpy.uint16(array2)
            cv2.imwrite(filename.replace("#", str(n) + '-' + str(n2)), array2)

def save_data(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        array = numpy.float32(array)
        cv2.imwrite(filename.replace("#", str(n)), array)

def save_depth_from_3d(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        depth_channel = numpy.zeros((1, array.shape[1], array.shape[2]))
        depth_channel[0,:,:] = array[2, :, :]
        depth_channel = depth_channel.transpose(1, 2, 0) * scale
        depth_channel = numpy.uint16(depth_channel)
        cv2.imwrite(filename.replace("#", str(n)), depth_channel)

def save_normals(filename, tensor, scale=255.0):
    b, _, __, ___ = tensor.size()    
    for n in range(b):   
        normals = tensor[n, :, :, :].detach().cpu().numpy()
        # transpose for consistent rendering, multiplied by scale
        normals = (normals.transpose(1, 2, 0) + 1) * scale // 2 + 1
        # image write
        cv2.imwrite(filename.replace("#", str(n)), normals)

def save_phong_normals(filename, tensor):
    b, _, __, ___ = tensor.size()
    for n in range(b):            
        # the z-component data of each normal vector is retrieved 
        z_comp = tensor[n, 2, :, :].detach().cpu().numpy()
        # phong image (1, z_comp.shape[0], z_comp.shape[1]) is initialized
        phong = numpy.zeros((1, z_comp.shape[0], z_comp.shape[1]))
        # z value is inversed to paint properly based on misalignment from camera's FOV direction 
        phong[0,:,:] = 1 - z_comp
        # get max uint16 value
        iui16 = numpy.iinfo(numpy.uint16)
        scale = iui16.max
        # transpose for consistent rendering
        phong = phong.transpose(1, 2, 0) * scale
        # to unsigned int16 
        phong = numpy.uint16(phong)
        # image write
        cv2.imwrite(filename.replace("#", str(n)), phong)

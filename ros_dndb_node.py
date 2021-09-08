#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ros_numpy
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

####DNDB####
import torch
import models
import utils
import dataset
import importers
import argparse
import os
import sys
import argparse

from supervision import *
from exporters import *
from importers import *

import datetime

import torch.nn.functional as F

from models import AE
from supervision import *
from exporters import *
from utils import *
from models.shallow_partial import *


class VideoStreamer:
    """
    Video streamer that continuously is reading frames through subscribing to d435/d435i images.
    Frames are then ready to read when program requires.
    """
    def __init__(self, pub, video_file=None):
        self._pub = pub
        self.colour_retrieved = False
        self.depth_retrieved = False
        self.intrin_retrieved = False
        self.color_image = np.zeros([480,640,3], dtype=np.uint8)
        self.depth_image = np.zeros([480,640], dtype=np.uint8)

    def read(self):
        return (self.color_image, self.depth_image)

    def colour_callback(self, msg):
        if not self.colour_retrieved:
            im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            # inverse rgb to bgr
            im = im[:,:,::-1]
            self.color_image = im
            self.colour_retrieved = True
    
    def depth_callback(self, msg):
        if not self.depth_retrieved:
            im = ros_numpy.numpify(msg)
            self.depth_image = im
            self.depth_retrieved = True

    def intrin_callback(self, cameraInfo):
        """
        D435/D435i camera intrinsic values can be derived from CameraInfo ROS message.
        """
        if not self.intrin_retrieved:
            self.intrin = rs2.intrinsics()
            self.intrin.width = cameraInfo.width
            self.intrin.height = cameraInfo.height
            self.intrin.ppx = cameraInfo.K[2]
            self.intrin.ppy = cameraInfo.K[5]
            self.intrin.fx = cameraInfo.K[0]
            self.intrin.fy = cameraInfo.K[4]
            #self.intrin.model = cameraInfo.distortion_model
            self.intrin.model  = rs2.distortion.none     
            self.intrin.coeffs = [i for i in cameraInfo.D]
            self.intrin_retrieved = True

    def set_not_retrieved(self):
        self.colour_retrieved = False
        self.depth_retrieved = False
        self.intrin_retrieved = False

    def publish(self, image):
        # Convert image array to smassage from sensor_msgs
        img_msg = ros_numpy.msgify(Image, image, encoding='mono16')
        self._pub.publish(img_msg)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def convtranspose3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResEncoder(nn.Module):
    def __init__(self, in_features, out_features, stride=1, downsample=None):
        super(ResEncoder, self).__init__()
        self.conv1 = conv3x3(in_features, out_features, stride)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResDecoder(nn.Module):
    def __init__(self, in_features, out_features, stride=1, upsample=None):
        super(ResDecoder, self).__init__()
        self.conv1 = convtranspose3x3(in_features, out_features, stride)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.upsample = upsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.upsample is not None:
            residual = self.upsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class DB(torch.nn.Module):
    def __init__(self, width, height, ndf):
        super(DB, self).__init__()

        feature = 64

        downsample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            )

        upsample = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False),
                nn.BatchNorm2d(64),
            )

        upsample2 = nn.Sequential(
                nn.ConvTranspose2d(64*2, 64, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False),
                nn.BatchNorm2d(64),
            )

        upsample3 = nn.Sequential(
                nn.ConvTranspose2d(64*2, 1, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False),
                nn.BatchNorm2d(1),
            )

        self.encoder_1 = nn.Sequential(nn.Conv2d(2, feature, 3, stride=1, padding=1), Mish())

        self.encoder_2 = nn.Sequential(ResEncoder(feature, feature, 2, downsample), Mish())
        self.encoder_3 = nn.Sequential(ResEncoder(feature, feature, 2, downsample), Mish())
        self.encoder_4 = nn.Sequential(ResEncoder(feature, feature, 2, downsample), Mish())

        self.conn_1 = nn.Sequential(nn.Conv2d(feature, feature, 1, stride=1, padding=0), Mish())
        self.conn_2 = nn.Sequential(nn.Conv2d(feature, feature, 1, stride=1, padding=0), Mish())
        self.conn_3 = nn.Sequential(nn.Conv2d(feature, feature, 1, stride=1, padding=0), Mish())

        self.decoder_4 = nn.Sequential(ResDecoder(feature, feature, 2, upsample), Mish())
        self.decoder_3 = nn.Sequential(ResDecoder(feature, feature, 2, upsample), Mish())
        self.decoder_2 = nn.Sequential(ResDecoder(2*feature, feature, 2, upsample2), Mish())
        self.decoder_1 = nn.Sequential(nn.ConvTranspose2d(2*feature, 1, 3, stride=1, padding=1, output_padding=0))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)

    def forward(self, x, source):

        out = torch.cat((x, source), 1)

        encoder_1_out = self.encoder_1(out)
        # print("encoder_1_out", encoder_1_out.shape)

        encoder_2_out = self.encoder_2(encoder_1_out)
        # print("encoder_2_out", encoder_2_out.shape)

        encoder_3_out = self.encoder_3(encoder_2_out)
        # print("encoder_3_out", encoder_3_out.shape)

        encoder_4_out = self.encoder_4(encoder_3_out)
        # print("encoder_4_out", encoder_4_out.shape)

        out = self.conn_1(encoder_4_out)
        out = self.conn_2(out)
        out = self.conn_3(out)

        decoder_4_out = self.decoder_4(out)
        # print("decoder_4_out", decoder_4_out.shape)

        decoder_3_out = self.decoder_3(encoder_3_out)
        # print("decoder_3_out", decoder_3_out.shape)

        out = torch.cat((decoder_3_out, encoder_2_out), 1)

        decoder_2_out = self.decoder_2(out)
        # print("decoder_2_out", decoder_2_out.shape)

        out = torch.cat((decoder_2_out, encoder_1_out), 1)

        decoder_1_out = self.decoder_1(out)

        # print("decoder_1_out", decoder_1_out.shape)

        # out = torch.cat((decoder_1_out, source), 1)
        # out = self.decoder_conn_1(out)
        # out = self.decoder_conn_2(out)
        # out = self.decoder_conn_3(out)
        return decoder_1_out

__WIDTH__ = 640
__HEIGHT__ = 360

def run_model(model_path: str, input: np.ndarray, device: str, scale: float):
    uv_grid_t = create_image_domain_grid(model_params['width'], model_params['height'])

    if args.pointclouds:
        device_repo_path = os.path.join(args.input_path,"device_repository.json")
        device_repository = importers.intrinsics.load_intrinsics_repository(device_repo_path)
    
    # for file in files:
    #     filename, extension = os.path.basename(file).split('.')
    #     if extension == "json":
    #         continue
    depthmap = load_depth(
        filename = input,
        scale = scale
    )
        
    if depthmap.shape[3] != model_params['width'] or depthmap.shape[2] != model_params['height']:
        depthmap = crop_depth(# for inference /w InteriorNet (__WIDTH__x480), center cropped to __WIDTH__x__HEIGHT__
            filename = input,
            scale = scale
        )

    mask, _ = get_mask(depthmap)
        
    mask, depthmap = mask.to(device), depthmap.to(device)

    predicted_depth = model(depthmap, mask)

    masked_predicted_depth = predicted_depth * mask

    DB_predicted = deblur_model(masked_predicted_depth, depthmap)

    DB_predicted = DB_predicted * mask

    DB_predicted = DB_predicted.detach().cpu().numpy()
    depthmap = depthmap.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    for i in range(360):
        for j in range(640):
            if DB_predicted[0,0,i,j] <= 0.1 or DB_predicted[0,0,i,j] >= 6.5:
                DB_predicted[0,0,i,j] = depthmap[0,0,i,j]

    DB_predicted = torch.from_numpy(DB_predicted.reshape(1, 1, 360, 640)).type(torch.float32).to(device)
    depthmap = torch.from_numpy(depthmap.reshape(1, 1, 360, 640)).type(torch.float32).to(device)
        
    # save denoising depthmap
    # output_file = os.path.join(output_path, filename + "_denoised." + extension)
    # output_file_ori = os.path.join(output_path, filename + "_noisy." + extension)
    # save_depth(output_file, masked_predicted_depth, 1/scale)
    # save_depth(output_file_ori, depthmap, 1/scale)
    # save_depth(os.path.join(output_path, filename + "_" + "_deblur.png"), DB_predicted, 1/scale)
    dndb_depthmap = return_depth(DB_predicted, 1/scale)

    return dndb_depthmap

def parse_arguments(args):
    usage_text = (
        "python test.py <plus options as shown below>"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("--model_path", type=str, default='src/ROS-dndb-node/deployData/pre_models/Denoising_Aug', help="Path to saved model to load.")
    parser.add_argument("--input", type=np.ndarray, help="Path to files for inference.")
    # parser.add_argument("--output_path", type=str, default='output', help="Path to directory to save the infered files.")
    parser.add_argument("--pointclouds", type=bool, default=False, help = "Save original and denoised pointclouds for RealSense input.")
    parser.add_argument("--autoencoder", type=bool, default=False, help = "Set model to autoencoder mode (i.e. trained without multi-view supervision, but as a depth map autoencoder).")
    parser.add_argument("-g","--gpu", type=str, default="0", help="The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.")
    parser.add_argument("--scale", type=float, default="0.001", help="How much meters does one bit represent in the input data.")
    return parser.parse_known_args(args)


if __name__ == '__main__':
    ###Initial Pytorch model###
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device("cuda:{}" .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else "cpu")

    ndf = 16 if args.autoencoder else 8
    model_params = {
        'width': __WIDTH__,
        'height': __HEIGHT__,
        'ndf': ndf,
        'dilation': 1,
        'norm_type': "elu",
        'upsample_type': "nearest"
    }

    model = models.get_model(model_params).to(device)

    deblur_model = DB(__WIDTH__, __HEIGHT__, ndf)
    utils.init.initialize_weights(deblur_model, "src/ROS-dndb-node/default_name_deblur_epoch_50")
    deblur_model = deblur_model.to(device)

    utils.init.initialize_weights(model, args.model_path)

    ###ROS Part###
    # publish a ROS topic of the denoised & deblurred depth results
    pub = rospy.Publisher('DNDB_depth', Image, queue_size=1)

    # initial node
    rospy.init_node('ros_dndb')

    rospy.loginfo('ROS_dndb node started...')

    # Initialise video streams from D435
    video_streamer = VideoStreamer(pub)

    """
    This is the ROS topic to get images from. If no image is being found, type
    'rostopic list' in the console to find available topics. It may be called 
    '/d400/color/image_raw' instead.
    """
    rospy.Subscriber("/d400/color/image_raw", Image, video_streamer.colour_callback)
    rospy.Subscriber("/d400/aligned_depth_to_color/image_raw", Image, video_streamer.depth_callback)
    # rospy.Subscriber("/d400/depth/camera_info", CameraInfo, video_streamer.intrin_callback)

    while True:
        color_img, depth_img = video_streamer.read()
        cv2.imwrite('src/ROS-dndb-node/examples/input/depth/%s.png' % rospy.get_time(), depth_img)
        cv2.imwrite('src/ROS-dndb-node/examples/input/rgb/%s.png' % rospy.get_time(), color_img)

        dndb_depthmap = run_model(
        args.model_path,
        input = depth_img,
        device = device,
        scale = args.scale
        )

        cv2.imwrite('src/ROS-dndb-node/examples/output/%s.png' % rospy.get_time(), dndb_depthmap)
        video_streamer.publish(dndb_depthmap)
        rospy.loginfo('A DNDB depthmap just published on DNDB_depth topic...')
        video_streamer.set_not_retrieved()

        # garbage collection
        # del output
        torch.cuda.empty_cache()
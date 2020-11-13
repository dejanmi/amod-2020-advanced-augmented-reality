#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import rospkg 

from dt_apriltags import Detector
"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        rospack = rospkg.RosPack()
        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')

        #
        #   Write your code here
        #
        self.bridge = CvBridge()
        self.image = None
        self.image_height = None
        self.image_width = None
        self.cam_info = None
        self.camera_info_received = False
        self.at_detector = Detector(families='tag36h11',
                               nthreads=1,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=1,
                               decode_sharpening=0.25,
                               debug=0)

        self.sub_image_comp = rospy.Subscriber(
            f'/{self.veh}/camera_node/image/compressed',
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )

        self.sub_cam_info = rospy.Subscriber(f'/{self.veh}/camera_node/camera_info', CameraInfo,
                                             self.cb_camera_info, queue_size=1)

        self.pub_aug_image = rospy.Publisher(f'/{self.veh}/{node_name}/image/compressed',
                                             CompressedImage, queue_size=10)

    def image_cb(self, image_msg):
        try:
            image = self.readImage(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return

        self.image = image
        self.image_height = np.shape(image)[0]
        self.image_width = np.shape(image)[1]

    def cb_camera_info(self, msg):
        if not self.camera_info_received:
            self.cam_info = msg

        self.camera_info_received = True

    def detect_april_tag(self):
        K = np.array(self.cam_info.K).reshape((3, 3))
        camera_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        img_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(img_grey, True, camera_params, 0.065)
        list_hom_tag = []
        for tag in tags:
            list_hom_tag.append(tag.homography)

        return list_hom_tag


    def projection_matrix(self, intrinsic, homography):
        """
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """
        K = intrinsic
        P_ext = np.linalg.inv(K) @ homography
        r_1 = P_ext[:, 0]
        r_1_norm = np.linalg.norm(r_1)
        r_1 = r_1 / r_1_norm
        r_2 = P_ext[:, 1]
        r_2_norm = np.linalg.norm(r_2)
        r_2 = r_2 / r_2_norm
        r_3 = np.cross(r_1, r_2)
        t_norm = (r_1_norm + r_2_norm)/2
        t = P_ext[:, 2] / t_norm
        P_ext_new = np.concatenate((r_1, r_2, r_3, t)).reshape((-1, 4), order='F')
        P = K @ P_ext_new

        return P

    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def onShutdown(self):
        super(ARNode, self).onShutdown()

    def run(self):
        while not rospy.is_shutdown():
            if self.image is None:
                continue
            if not self.camera_info_received:
                continue

            list_hom_tag = self.detect_april_tag()
            if not list_hom_tag:
                continue
            img_rend = self.image
            for h in list_hom_tag:
                proj_mat = self.projection_matrix(np.array(self.cam_info.K).reshape((3, 3)), h)
                img_rend = self.renderer.render(img_rend, proj_mat)

            comp_image = self.bridge.cv2_to_compressed_imgmsg(img_rend)
            self.pub_aug_image.publish(comp_image)


if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    camera_node.run()
    # Keep it spinning to keep the node alive
    rospy.spin()

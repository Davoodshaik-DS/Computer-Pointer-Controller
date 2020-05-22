#!/usr/bin/env python3

from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import facedetection
from facial_landmarks_detection import facelandmarksdetection
from gaze_estimation import Gaze
from head_pose_estimation import head_pose


import cv2
import os
import time
import sys
import logging
import numpy as np


def build_argparser():
    
    parser = ArgumentParser()
    
    parser.add_argument("-fd", "--facedetectionmodel", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimationmodel", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")

    parser.add_argument("-flags", "--Flags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fl, hp, ge like --flags fd hp fl (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fl for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )

    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")

    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                        "for a suitable plugin for device specified"
                        "(CPU by default)")

    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    
    
    return parser


def main():

    args = build_argparser().parse_args()
    Flags = args.Flags

    logger = logging.getLogger()
    inputFilePath = args.input
    inputFeeder = None
    
    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)

    Dir = {'facedetection':args.facedetectionmodel, 'facelandmarksdetection':args.faciallandmarkmodel, 
    'Gaze':args.gazeestimationmodel, 'head_pose':args.headposemodel}

    for fileKey in Dir.keys():
        if not os.path.isfile(Dir[fileKey]):
            logger.error("Unable to find  "+fileKey+" xml file")
            exit(1)
            
    Fd = facedetection(Dir['facedetection'], args.device, args.cpu_extension)
    Fl = facelandmarksdetection(Dir['facelandmarksdetection'], args.device, args.cpu_extension)
    Ge = Gaze(Dir['Gaze'], args.device, args.cpu_extension)
    Hp = head_pose(Dir['head_pose'], args.device, args.cpu_extension)
    Mc = MouseController('medium','fast')


    #loading
    start_model_load_time=time.time()

    inputFeeder.load_data()
    Fd.load_model()
    Fl.load_model()
    Hp.load_model()
    Ge.load_model()

    total_model_load_time = time.time() - start_model_load_time


    count=0
    start_inference_time=time.time()


    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        count+=1

        if count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = Fd.predict(frame.copy(),args.prob_threshold)

        if type(croppedFace)==int:
            logger.error("unsupported layers, could not detect face")
            if key==27:
                break
            continue


        hp_out=Hp.predict(croppedFace.copy())
         
        l_coords,r_coords,coords=Fl.predict(croppedFace.copy())    

        new_coord,Gaze_vec=Ge.predict(l_coords,r_coords,hp_out)

        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)

        fps=count/total_inference_time

         

        if (not len(Flags)==0):
            new_frame = frame.copy()
            if 'fd' in Flags:
                new_frame = croppedFace

            if 'fl' in Flags:
                cv2.rectangle(croppedFace, (coords[0][0]-10, coords[0][1]-10), (coords[0][2]+10, coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(croppedFace, (coords[1][0]-10, coords[1][1]-10), (coords[1][2]+10, coords[1][3]+10), (0,255,0), 3)
                
            if 'hp' in Flags:
                cv2.putText(new_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            
            if 'ge' in Flags:
                x, y, w = int(Gaze_vec[0]*12), int(Gaze_vec[1]*12), 160
                le =cv2.line(l_coords.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(r_coords.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                croppedFace[coords[0][1]:coords[0][3],coords[0][0]:coords[0][2]] = le
                croppedFace[coords[1][1]:coords[1][3],coords[1][0]:coords[1][2]] = re
                
            cv2.imshow("visualization",cv2.resize(new_frame,(500,500)))

        if count%5==0:   
            Mc.move(new_coord[0],new_coord[1])    
        if key==27:
                break

    logger.error("Video Done...")
    print(total_inference_time)
    print(fps)
    print(total_model_load_time)

    cv2.destroyAllWindows()
    inputFeeder.close()


if __name__ == '__main__':
    main() 

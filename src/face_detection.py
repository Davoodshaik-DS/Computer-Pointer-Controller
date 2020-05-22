import cv2
import numpy as np
import os
from openvino.inference_engine import IECore,IENetwork,IEPlugin


class facedetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape


             
    def load_model(self):
        
        
        self.plugin=IECore()       
        
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        
        if len(unsupported_layers)!=0:
            print("unsupported layers found")
            exit(1)
           
       
        self.exec_net=self.plugin.load_network(network=self.model,device_name=self.device,num_requests=1)
       
    def predict(self, image,prob_threshold):
       
        processed_image=self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:processed_image})
        coords = self.preprocess_output(outputs, prob_threshold)

        if (len(coords)==0):
            return 0, 0
        coords = coords[0] 
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
   
        self.image=cv2.resize(image,(self.input_shape[3],self.input_shape[2]))   ## cv2.resize(frame, (w, h))
        self.image=self.image.transpose((2, 0, 1))  
        self.image=self.image.reshape(1, *self.image.shape)
       
        return self.image
        

    def preprocess_output(self, outputs,prob_threshold):
   
        coords =[]
        outs = outputs[self.output_name][0][0]
        for out in outs:
            conf = out[2]
            if conf>prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min,y_min,x_max,y_max])
        return coords

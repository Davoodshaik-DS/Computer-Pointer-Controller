import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork


class head_pose:
    
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


        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_names = [a for a in self.model.outputs.keys()]


        

    def load_model(self):
        
        self.plugin=IECore()       
        
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        
        if len(unsupported_layers)!=0:
            print("unsupported layers found")
            exit(1)
           
       
        self.exec_net=self.plugin.load_network(network=self.model,device_name=self.device,num_requests=1)


               

    def predict(self, image):
        
        self.processed_image=self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:self.processed_image})
        Result = self.preprocess_output(outputs)
        return Result


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    
        self.image=cv2.resize(image,(self.input_shape[3],self.input_shape[2])) ## cv2.resize(frame, (w, h))
        self.image=self.image.transpose((2, 0, 1))  
        self.image=self.image.reshape(1, *self.image.shape)
        
        return self.image

    def preprocess_output(self, outputs):
    
        res = []
        res.append(outputs['angle_y_fc'].tolist()[0][0])
        res.append(outputs['angle_p_fc'].tolist()[0][0])
        res.append(outputs['angle_r_fc'].tolist()[0][0])
        return res


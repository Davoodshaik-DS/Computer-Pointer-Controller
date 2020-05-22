import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork


class facelandmarksdetection:
    
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
        

    def predict(self, image):
        
        self.processed_image=self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:self.processed_image})
        coords = self.preprocess_output(outputs)
        
        h=image.shape[0]
        w=image.shape[1]
        
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32) 

        l_xmin=coords[0]-10
        l_xmax=coords[0]+10


        l_ymin=coords[1]-10            
        l_ymax=coords[1]+10
        
        r_xmin=coords[2]-10
        r_xmax=coords[2]+10

        r_ymin=coords[3]-10              
        r_ymax=coords[3]+10
        

        left_eye =  image[l_ymin:l_ymax, l_xmin:l_xmax]
        right_eye = image[r_ymin:r_ymax, r_xmin:r_xmax]

        coords = [[l_xmin,l_ymin,l_xmax,l_ymax], [r_xmin,r_ymin,r_xmax,r_ymax]]
        
        return left_eye, right_eye, coords


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    
        image_ct = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image=cv2.resize(image_ct,(self.input_shape[3],self.input_shape[2]))   ## cv2.resize(frame, (w, h))
        self.image=self.image.transpose((2, 0, 1))  
        self.image=self.image.reshape(1, *self.image.shape)
       
        return self.image

    def preprocess_output(self, outputs):
    
        res=outputs[self.output_name][0]
        lx = res[0].tolist()[0][0]
        ly = res[1].tolist()[0][0]
        rx = res[2].tolist()[0][0]
        ry = res[3].tolist()[0][0]

        return(lx,ly,rx,ry)

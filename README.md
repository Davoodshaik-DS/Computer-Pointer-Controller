### Author: SHAIK DAVOOD


# Computer Pointer Controller

Computer Pointer Controller application that makes use Intel OpenVino toolkit to run inference on an image or video or camera stream of a person, with four models in pipeline to finally extract eye gazing direction of the person and automatically move mouse pointer there.Computer Pointer Controller app focuses on changing the position of mouse Pointer by the direction of eyes and Head Pose.This can take Video file or Webcam as Input and then can perform accordingly.It uses combination of different models to give us a desired output.
## Project Set Up and Installation
### Here is the installation guide [link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html) for it.
<br>You can run it on Linux,Windows as well as macOS. 
* Now you have to initialze the OpenVino Environment.You can do this by the following command.
* cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
* setupvars.bat
- The project directory is structured as follows:
```
                    starter
                    |  
                    |_ bin
                    |  |_demo.mp4
                    |  |_pipeline.png
                    |      
                    |_ README.md    
                    |   
                    |_ requirements.txt   
                    |    
                    |_src
                       |_ main.py
                       |_ input_feeder.py
                       |_ mouse_controller.py
                       |_ face_detection.py
                       |_ head_pose_estimation.py
                       |_ facial_landmarks_detection.py
                       |_ gaze_estimation.py
    
```
To setup the project follow steps in project root directory: 
* Install [virtualenv](https://pypi.org/project/virtualenv/)
```
      pip install virtualenv
```
* Create a virutal environment
```
      virtualenv -p python3 env
```
* Activate the created virtual environment
```
      source env/bin/activate
```
* Install packages listed in `requirements.txt`
```
      pip3 install -r requirements.txt
```


### Models:
* [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
### Or You can run these command:-
* python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

* [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
### Or You can run these command:-
* python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

* [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
### Or You can run these command:-
* python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
### Or You can run these command:-
* python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"

## Demo
* Open a new terminal and run the following commands:-
* cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
* setupvars.bat
#### With these commands your path is intialized,Now change directory to source directory of cloned project.
* cd  /src
* Run the main.py file
 
* python main.py -fd  models\Face_detection\face-detection-adas-binary-0001.xml  -fl models\Landmarks_detection\FP32\landmarks-regression-retail-0009.xml  -hp models\Head_Pose\FP32\head-pose-estimation-adas-0001.xml  -ge models\Gaze_Estimation\FP32\gaze-estimation-adas-0002.xml  -i bin\demo.mp4 -l opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libinference_engined.dylib -d CPU -pt 0.6

#### For GPU

* python main.py -fd 'Path of xml file of face detection model' -fl 'Path of xml file of facial landmarks detection model' -hp 'Path of   xml file of head pose estimation model' -ge 'Path of xml file of gaze estimation model' -i 'Path of input video file or enter cam for   taking input video from webcam' -d 'GPU'

#### For FPGA

* python main.py -fd 'Path of xml file of face detection model' -fl 'Path of xml file of facial landmarks detection model' -hp 'Path of   xml file of head pose estimation model' -ge 'Path of xml file of gaze estimation model' -i 'Path of input video file or enter cam for   taking input video from webcam' -d 'HETERO:FPGA,CPU'

### Command Line Arguments Used:
Following are the command line arguments that can be used for running main.py file.
* -fd (required) : Specify the path of Face Detection model's xml file
* -fl (required) : Path to .xml file of Facial Landmark Detection model

* -hp (required) : Path to .xml file of Head Pose Estimation model
* -ge (required) : Path to .xml file of Gaze Estimation model.
* -i (required) : Specify the path of input video file or enter cam for taking input video from webcam

* -d (optional) : Specify the target device to infer on,"CPU, GPU, FPGA or MYRIAD is acceptable. Looks
                        for a suitable plugin for device specified "(CPU by default)".
* -l (optional) : Specify the absolute path of cpu extension if some layers of models are not supported on the device.
* -pt (optional): Probability threshold for model to detect the face accurately from the video frame.


## Documentation
Code base is moduler with each module having seperate concerns:<br>
- `face_detection.py`: Class for utilizing Face Detection model to extract box coordinates of face of the person in frame. These coordinates are used to crop face from frame.
- `facial_landmarks_detection.py`: Class for utilizing Facial Landmarks Detection model to get the facial landmarks coordinates from face. However, for the app only required eye landmarks are returned which are later used to extract left and right eye.
- `head_pose_estimaion.py`: Class for utilizing Head Pose Estimation model to extract, from face, the head pose angles- yaw, pitch and roll as list with indices in order respectively. These angles are later required in pipeline.
- `gaze_estimation.py`: Class for utilizing Gaze Estimation model which given left and right eye images as well as head pose angles, yields the gaze vectors. Gaze vectors define direction of person's gaze.
- `input_feeder.py`: Convenient class for reading and feeding frames from input media.
- `mouse_controller.py`: Convenient class for controlling mouse pointer.
- `main.py`: Script, which is the starting point for the app.
- `download_models.sh`: Bash script to download all required models from model zoo automatically.
## Benchmarks
Benchmark was done on CPU with inputs from `demo.mp4` file present in `media` directory. Following tables show the benchmark stats for models used in the app:<br>
1. Face Detection Model<br>

|             | Load Time(ms) | Total Infernce Time(ms)  | FPS(frames/s) |
|-------------|---------------|--------------------------|---------------|
|**FP16**     |     225       |         9276             |     0.64      |
|**FP16-INT8**|     454       |         8042             |     0.74      |
|**FP32**     |     229       |         11060            |     0.54      |

2. Facial Landmarks Detection Model<br>

|             | Load Time(ms) | Total Infernce Time(ms)  | FPS(frames/s) |
|-------------|---------------|--------------------------|---------------|
|**FP16**     |      55       |          309             |  1.9          |
|**FP16-INT8**|      100      |          295             |  2.1          |
|**FP32**     |      56       |          358             |  1.60         |

3. Head Pose Estimation Model<br>

|             | Load Time(ms) | Total Infernce Time(ms)  | FPS(frames/s) |
|-------------|---------------|--------------------------|---------------|
|**FP16**     |      74       |           740            |     8.04      |
|**FP16-INT8**|      217      |           545            |     10.91     |
|**FP32**     |      64       |           833            |     7.15      |

4. Gaze Estimation Model<br>

|             | Load Time(ms) | Total Infernce Time (ms) | FPS(frames/s) |
|-------------|---------------|--------------------------|---------------|
|**FP16**     |     90        |          851             |    6.98       |
|**FP16-INT8**|     283       |          592             |    10.05      |
|**FP32**     |     83        |          988             |    6.02       |


## Results
* I have ran mine models on different hardwares with different precisions:
For running on CPU,I tried different combinations with different precisions like INT8,FP16,FP32.
I have tried to reduce precision Value but precision also reduces accuracy. 
* So when you use lower precision model then you can get lower accuracy than higher precision model.

* General trend for the benchmarks of all models hints that models of pricision FP16-INT8 has fastest inference timings. And FP32 is slowest (though, fastest model loading times) in inference. FP16 stands intermediate between these two precisions in stats.<br>

* Each individual model (of multiple precisions) is derived/converted from same saved model of a DL framework. However, optimization during conversion is done differently to get a specific precision of model. For precision (or data type) altering, **Quantization** technique is used. Quantization trades better inference speed and lower model size & memory consumption with the accuracy. This is because parameters stored are down sampled from 32 bit floating point values to 16 bit floats or 8 bit ints, for example. And 32 bit floats take up more storage than 16 bit floats, which in turn takes more storage than 8 bit ints. Operations on these data types follow the same trend. Models can be quantized as far as accuracy is acceptable for use case.<br>

* FP16-INT8 precision model's parameters are quantized more than FP16 ones, hence former has faster inference speed. While FP32 is intact with it's 32 bit float type of parameter values and are slower.
## Stand Out Suggestions
* I have tried to build an inference pipeline for both video file and webcam feed as input.
* Allowing the user to select their input option in the command line arguments:
-i argument takes the input video file or a webcam, for accessing video file the command is -i " path of video file " whereas for accessing webcam -i "cam".
 
* Depending on chosen option it will work.

### Edge Cases
Sometimes if Face Detection model is unable to detect box coordinates for face, nothing is done or fed to subsequent models. Instead app continues to try to detect face in next frame and so on.

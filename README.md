
# Virtual Keyboard for Smart Glasses - Demo

This python project implements a virtual keyboard which hypothetically could be used in smart glasses. It has not been tested in smart glasses yet but only on pc.

Watch virtual keyboard on <a href="https://youtu.be/81pK924fW3g" target="_blank">youtube</a>.

<img style="margin: 15px;" src="https://i.imgur.com/7iw7M0o.jpg" alt="virtual keyboard hellow world typing" title="Hello world virtual keyboard">

## Requirements

 - We use **Python version 3.7.3** to build the project.
 - All the dependencies of the project are listed in the requirements.txt file.


## How to setup the project on your pc

**1\.** Download this project.

**2\.** Download and unzip the folder contained [here](https://drive.google.com/drive/folders/1Qor3ywnjTYXUx-6Ppc3t1OX0DuZOQdHg?usp=sharing) and move the extracted folders in the project's directory. The project structure you should have is shown below:   
    
    ├─ virtual-keyboard-for-smart-glasses-demo
    │   ├── click_detection_models
    │   ├── hand_segmentation_model
    │   ├── keys
    │   ├── test_videos
    │   ├── .gitignore.txt
    │   ├── README.txt
    │   ├── virtual_keyboard.py
    └   ├── requirements.txt
    
The folders "click_detection_models" and "hand_segmentation_model" contain ML-models necessary to run the code, the folder "keys" contains the keys' images (the buttons of the keyboard) and the folder "test_videos" contains some sample videos you can use to test the keyboard. If you want to type in the keyboard with your own hands, see [below](#how-to-test-the-keyboard-with-your-own-hands).

**3\.** Install the requirements using pip:
```shell
pip install -r requirements.txt 
```


## How to test the keyboard with your own hands

You need to connect an external camera to your pc and get access to the cameras' stream. Position the camera so that it is facing downwards and put your hands under the camera. In the virtual_keyboard.py file in the "Basic parameters.." section set the video source to be the external camera. Then run the virtual_keyboard.py and you will see your hands inside the keyboard. You can type only with your index fingers and in order to achieve better hand segmentation it's better to have all the rest fingers closed as shown [here](#virtual-keyboard-for-smart-glasses---demo). In the image below you can see our setup. We use a phone as an external camera and we connect it to the pc through the DroidCam app.

<img style="display:block; margin:0 auto; margin-top: 25px;" src="https://i.imgur.com/Hyrs9Ni.jpg" alt="virtual keyboard type with your hands" width="400" title="virtual keyboard type with your hands">


## A little bit about this virtual keyboard

 - This implementation is part of my thesis which can be found <a href="http://ikee.lib.auth.gr/record/342948/files/Poulios%20Ilias.pdf" target="_blank">here</a> (available only in greek).

 - An originality of this virtual keyboard is that it offers the feeling of natural typing. In other virtual keyboards, users see their hands under the virtual letters as shown in the first image below. In our implementation, using hand segmentation, we detect where hands are and we hide the letters from those areas so that it appears that the hands are above the letters. This makes typing more convenient. For the hand segmentation we use a convolutional neural network.
 
    <img style="float: left; margin: 15px;" src="https://i.imgur.com/SzoTz03.jpg" alt="hands-below-keyboard" width="350" title="Hands below keyboard's letters">
    <img style=" margin: 15px;" src="https://i.imgur.com/s5ayMkD.jpg" alt="hands-below-keyboard" width="350" title="Hands above keyboard's letters">

 - To detect clicks we track the movement of three points of each finger for seven consecutive frames. We use <a href="https://developers.google.com/mediapipe/solutions/vision/hand_landmarker" target="_blank">MediaPipe Hands</a> to find the points. At each frame we check whether was a click in the last seven frames or not. In the image below you can see the points we track for the right hand. For the click detection we train a fully connected neural network which takes as input the relative motion of the three points between consecutive frames.  
  
    <img style="display:block; margin: 0 auto; margin-top:20px;" src="https://i.imgur.com/5tPopdd.jpg" alt="hand points for click detection"  title="right hand points for click detection">


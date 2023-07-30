
import math
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import onnxruntime as rt


####################################
##  Basic parameters you can set  ##
####################################

## Choose hand segmentation model (raw or optimized)
# HAND_SEGMENTATION_MODEL = 'raw'
HAND_SEGMENTATION_MODEL = 'optimized'

## Set hand segmentation prediction threshold
HAND_SEGMENTATION_THRESHOLD = 0.7

## Set click detection prediction threshold
CLICK_DETECTION_THRESHOLD = 0.85 

## Choose videoCapture source (there is a testing_videos folder you can peek videos from or you can use your own)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('testing_videos/test2.mp4')


####################################
####################################



tf.config.set_visible_devices([], 'GPU')

mp_hands = mp.solutions.hands
hands_skel = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2)


def get_hand_box_coordinates(handLadmark, image_shape):
    all_x, all_y = [], [] # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1])) # multiply x by image width
        all_y.append(int(handLadmark.landmark[hnd].y * image_shape[0])) # multiply y by image height
    return max(min(all_x)-20,0), max(min(all_y)-20,0), min(max(all_x)+20,640), min(max(all_y)+20,480) # return as (xmin, ymin, xmax, ymax)
    # x horizontal axis
    # y vertical axis
    # (0,0) is in left up corner


###################################
### Load Hand Segmentaion Model ###
###################################
SEGM_MOD_INPUT_WIDTH = 112
SEGM_MOD_INPUT_HEIGHT = 112
SEGM_MOD_INPUT_CHANNELS = 3
if HAND_SEGMENTATION_MODEL == 'raw':
    RAW_MODEL_PATH = './hand_segmentation_model/raw_model/'
    segmentation_model = load_model(RAW_MODEL_PATH, compile=True)
    input_img = np.zeros((1, SEGM_MOD_INPUT_WIDTH, SEGM_MOD_INPUT_HEIGHT, SEGM_MOD_INPUT_CHANNELS), 'uint8')
    preds = segmentation_model(input_img)
elif HAND_SEGMENTATION_MODEL == 'optimized':
    OPTIMIZED_MODEL_PATH = './hand_segmentation_model/onnx_optimized_model/'
    sess = rt.InferenceSession(OPTIMIZED_MODEL_PATH + "model.onnx", providers=['CPUExecutionProvider']) # python 3.8.7 crashes here!
    input_name = sess.get_inputs()[0].name


###################################
### Load Click Detection Models ###
###################################
LEFT_CLICK_MODEL_PATH = './click_detection_models/left_hand_click_detection_model'
RIGHT_CLICK_MODEL_PATH = './click_detection_models/right_hand_click_detection_model'
left_click_model = load_model(LEFT_CLICK_MODEL_PATH, compile=True)
right_click_model = load_model(RIGHT_CLICK_MODEL_PATH, compile=True)


##################################
### Click Detection Parameters ###
##################################
TIME_WINDOW = 7
left_p1_x = [100] * TIME_WINDOW # Creates a list of size TIME_WINDOW. All elements have value 100
left_p1_y = [100] * TIME_WINDOW
left_p2_x = [100] * TIME_WINDOW
left_p2_y = [100] * TIME_WINDOW
left_p3_x = [100] * TIME_WINDOW
left_p3_y = [100] * TIME_WINDOW
right_p1_x = [100] * TIME_WINDOW # Creates a list of size TIME_WINDOW. All elements have value 100
right_p1_y = [100] * TIME_WINDOW
right_p2_x = [100] * TIME_WINDOW
right_p2_y = [100] * TIME_WINDOW
right_p3_x = [100] * TIME_WINDOW
right_p3_y = [100] * TIME_WINDOW
left_p1_previous = (-1,-1) # Left finger-p1 previus coordinates
right_p1_previous = (-1,-1) # Right finger-p1 previus coordinates
left_p2_previous = (-1,-1) # Left finger-p2 previus coordinates
right_p2_previous = (-1,-1) # Right finger-p2 previus coordinates
left_p3_previous = (-1,-1) # Left finger-p3 previus coordinates
right_p3_previous = (-1,-1) # Right finger-p3 previus coordinates


########################
### Frame parameters ###
########################
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


#######################
### Keyboard layout ###
#######################
w  =  0.7   # keyboard width compared with frame width
t  =  0.1   # keyboard offset from top compared with frame height
x  =  0.2   # distance between keys on x axis compared with key width
y  =  0.2   # distance between keys on y axis compared with key width (line-height)
keyboard_width = math.floor(w*FRAME_WIDTH)
keyboard_offset_x = math.floor((FRAME_WIDTH - keyboard_width)/2)  # keyboard distance from right side of frame
keyboard_offset_y = math.floor(t*FRAME_HEIGHT)                    # keyboard distance from top side of frame
key_width = math.floor(keyboard_width/(7+6*x)) # key width = key height 
original_key_width = 70 # pixels
key_ratio = key_width/original_key_width
key_distance_x = math.floor(x*key_width)
key_distance_y = math.floor(y*key_width)
key_height = key_width
keysNames = ["q", "w", "e", "r", "i", "o", "backspace",   #1
             "a", "s", "d", "f", "g",  "k", "p",          #2
             "z", "x", "c", "v", "b", "n", "l",           #3 
             "t", "y", "u", "space2", "h", "j", "m"]      #4            
keys_backgrounds = []
keys_texts = []

## Load keys
for i in range(0, len(keysNames)):
    key = cv2.imread("./keys/" + keysNames[i] + ".png")
    original_key_width = np.size(key, 0)
    original_key_height = np.size(key, 1)
    new_key_width = math.floor(original_key_width*key_ratio)
    new_key_height = math.floor(original_key_height*key_ratio)
    key = cv2.resize(key, (new_key_width,new_key_height), fx=0, fy=0, interpolation=cv2.INTER_NEAREST) 
    key_gray = cv2.cvtColor(key, cv2.COLOR_BGR2GRAY)
    keys_backgrounds.append(np.where(key_gray <= 200))
    keys_texts.append(np.where(key_gray <= 140))

## Calculate keys' offsets
keys_offsets = np.zeros((len(keysNames), 2), 'uint')
for i in range(0, 7):
    keys_offsets[i] = [keyboard_offset_x + (key_distance_x + key_width)*i, keyboard_offset_y]
for i in range(7, 14):
    keys_offsets[i] = [keyboard_offset_x + (key_distance_x + key_width)*(i-7), keyboard_offset_y + key_height + key_distance_y]
for i in range(14, 21):
    keys_offsets[i] = [keyboard_offset_x + (key_distance_x + key_width)*(i-14), keyboard_offset_y + 2*(key_distance_y + key_height)]
for i in range(21, 28):
    keys_offsets[i] = [keyboard_offset_x + (key_distance_x + key_width)*(i-21), keyboard_offset_y + 3*(key_distance_y + key_height)]

## Calculate keys' centers
keys_centers = np.zeros((len(keysNames), 2))
half_key_height = round(key_height/2)
half_key_width = round(key_width/2)
for i in range(0, len(keysNames)):
    keys_centers[i] = keys_offsets[i] + [half_key_width, half_key_height]

## Text area
text_area = cv2.imread("./keys/space.png")
text_area = cv2.resize(text_area, (436,90), fx=0, fy=0, interpolation=cv2.INTER_NEAREST) 
text_area_gray = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
text_area_background = np.where(text_area_gray <= 200)

## Horizontal lines (y coordinate) - "hypothetical" lines between keys
horizontal_lines = np.zeros(5,'int')
for i in range(-1,4):
    horizontal_lines[i] = keyboard_offset_y + (i+1)*key_height + (2*i+1)*math.floor(key_distance_y/2)
## Vertical lines (x coordinate) - "hypothetical" lines between keys
vertical_lines = np.zeros(8,'int')
for i in range(-1,7):
    vertical_lines[i] = keyboard_offset_x + (i+1)*key_width + (2*i+1)*math.floor(key_distance_x/2)


####################
### Main Program ###
####################

stop_right_click_pred = False
right_counter = 0
stop_left_click_pred = False
left_counter = 0
clicked_letter_idxs = []

while True:
    ret, frame_bgr = cap.read()

    if not ret:
      print("Ignoring empty camera frame.")
      break

    # frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    ## To improve skeletonization performance, optionally mark the image as not writeable to pass by reference.
    frame_rgb.flags.writeable = False
    skel_results = hands_skel.process(frame_rgb)
    frame_rgb.flags.writeable = True    

    right_hand_exists = False
    left_hand_exists = False
    right_index = -1
    left_index = -1
    hand_indexs = []
    if skel_results.multi_hand_landmarks:
        num_of_hands_in_frame = len(skel_results.multi_handedness)
        num_of_hands_in_frame = min(2,num_of_hands_in_frame)
        for i in range(0,num_of_hands_in_frame):
            if skel_results.multi_handedness[i].classification[0].index == 0:
                right_index = i
                hand_indexs.append(right_index)
                right_hand_exists = True
            else:
                left_index = i
                hand_indexs.append(left_index)
                left_hand_exists = True

        ## Initialize empty 3d array to store the images that will be segmented
        imgs_to_be_segmented = np.zeros((num_of_hands_in_frame, SEGM_MOD_INPUT_WIDTH, SEGM_MOD_INPUT_HEIGHT, 3), 'float32')
        hand_box_coordinates = [(),()]

        for hand_index in hand_indexs:
            ## Get hand's box coordinates
            (xmin, ymin, xmax, ymax) = get_hand_box_coordinates(skel_results.multi_hand_landmarks[hand_index],frame_rgb.shape)

            if hand_index == right_index:
                distances_x = np.abs(vertical_lines[:] - xmax)
                idx = np.argmin(distances_x)
                if (distances_x[idx] + 3 > math.floor(key_distance_x/2)) and (distances_x[idx] < (math.ceil(key_distance_x/2) + math.ceil(key_width/2))):
                    xmax = vertical_lines[idx]
            
                distances_y = np.abs(horizontal_lines[:] - ymax)
                idx = np.argmin(distances_y)
                if (distances_y[idx] + 3 > math.floor(key_distance_y/2)) and (distances_y[idx] < (math.ceil(key_distance_y/2) + math.ceil(key_height/2))):
                    ymax = horizontal_lines[idx]
            if hand_index == left_index:
                distances_x = np.abs(vertical_lines[:] - xmin)
                idx = np.argmin(distances_x)
                if (distances_x[idx] + 3 > math.floor(key_distance_x/2)) and (distances_x[idx] < (math.ceil(key_distance_x/2) + math.ceil(key_width/2))):
                    xmin = vertical_lines[idx]
            
                distances_y = np.abs(horizontal_lines[:] - ymax)
                idx = np.argmin(distances_y)
                if (distances_y[idx] + 3 > math.floor(key_distance_y/2)) and (distances_y[idx] < (math.ceil(key_distance_y/2) + math.ceil(key_height/2))):
                    ymax = horizontal_lines[idx]

            ## Resize hand's box and store it in imgs_to_be_segmented array
            imgs_to_be_segmented[hand_index] = cv2.resize(frame_rgb[ymin:ymax,xmin:xmax,:],(SEGM_MOD_INPUT_WIDTH,SEGM_MOD_INPUT_HEIGHT),interpolation=cv2.INTER_NEAREST)

            ## Store hand's box coordinates
            hand_box_coordinates[hand_index] = (xmin, ymin, xmax, ymax)     
        
        ## Segment hand images
        if HAND_SEGMENTATION_MODEL == 'raw':
            preds = segmentation_model(imgs_to_be_segmented)
            mask = (preds.numpy() > HAND_SEGMENTATION_THRESHOLD).astype(np.uint8)
        elif HAND_SEGMENTATION_MODEL == 'optimized':
            preds = sess.run(None, {input_name: imgs_to_be_segmented})[0]
            mask = (preds > HAND_SEGMENTATION_THRESHOLD).astype(np.uint8)
  
        ## Find where the hands are on current frame and store it in hand_idxs array. We also store the values of hands' pixels on hand_values array 
        hand_idxs = [[],[]]
        hand_values = [[],[]] 
        for hand_index in hand_indexs:
            (xmin, ymin, xmax, ymax) = hand_box_coordinates[hand_index]
            hand_idxs[hand_index] = np.where(cv2.resize(mask[hand_index], (xmax-xmin,ymax-ymin), interpolation=cv2.INTER_NEAREST) == 1)
            hand_values[hand_index] = frame_bgr[ymin + hand_idxs[hand_index][0], xmin + hand_idxs[hand_index][1],:]

    ## Add keys inside frame
    for i in range(0, len(keysNames)):
        frame_bgr[keys_backgrounds[i][0] + keys_offsets[i][1] , keys_backgrounds[i][1] + keys_offsets[i][0]] = [0,255,0]
        frame_bgr[keys_texts[i][0] + keys_offsets[i][1] , keys_texts[i][1] + keys_offsets[i][0]] = [3,26,3]


    if right_hand_exists:
        ## Get the appropriate right hand landmarks from mediapipe library
        right_tip_landmark = skel_results.multi_hand_landmarks[right_index].landmark[8]
        right_p1_landmark = skel_results.multi_hand_landmarks[right_index].landmark[7]
        right_p2_landmark = skel_results.multi_hand_landmarks[right_index].landmark[6]
        right_p3_landmark = skel_results.multi_hand_landmarks[right_index].landmark[5]

        ## Tranform the above landmarks from relative position to exact position
        right_tip = (int(right_tip_landmark.x*FRAME_WIDTH),
                     int(right_tip_landmark.y*FRAME_HEIGHT - 7))
        right_p1 = (int(right_p1_landmark.x*FRAME_WIDTH),
                    int(right_p1_landmark.y*FRAME_HEIGHT))
        right_p2 = (int(right_p2_landmark.x*FRAME_WIDTH),
                    int(right_p2_landmark.y*FRAME_HEIGHT))
        right_p3 = (int(right_p3_landmark.x*FRAME_WIDTH),
                    int(right_p3_landmark.y*FRAME_HEIGHT))   
        
        ## Calculate the nearest key of the right tip landmark and change its colour to red
        distances = [np.sqrt(pow(keys_centers[:,0] - right_tip[0],2) + pow(keys_centers[:,1] - right_tip[1],2))]
        idx = np.argmin(distances)
        frame_bgr[keys_backgrounds[idx][0] + keys_offsets[idx][1] , keys_backgrounds[idx][1] + keys_offsets[idx][0]] = [0,0,255]
        frame_bgr[keys_texts[idx][0] + keys_offsets[idx][1] , keys_texts[idx][1] + keys_offsets[idx][0]] = [3,26,3]
                      
        ## Calculate and store the relative movement of right finger-p1
        if right_p1_previous == (-1,-1):
            right_p1_previous = right_p1
        else:
            right_p1_x.insert(0, right_p1[0] - right_p1_previous[0])
            right_p1_x.pop(TIME_WINDOW)
            right_p1_y.insert(0, right_p1[1] - right_p1_previous[1])
            right_p1_y.pop(TIME_WINDOW)
            right_p1_previous = right_p1

        ## Calculate and store the relative movement of right finger-p2
        if right_p2_previous == (-1,-1):
            right_p2_previous = right_p2
        else:
            right_p2_x.insert(0, right_p2[0] - right_p2_previous[0])
            right_p2_x.pop(TIME_WINDOW)
            right_p2_y.insert(0, right_p2[1] - right_p2_previous[1])
            right_p2_y.pop(TIME_WINDOW)
            right_p2_previous = right_p2

        ## Calculate and store the relative movement of right finger-p3
        if right_p3_previous == (-1,-1):
            right_p3_previous = right_p3
        else:
            right_p3_x.insert(0, right_p3[0] - right_p3_previous[0])
            right_p3_x.pop(TIME_WINDOW)
            right_p3_y.insert(0, right_p3[1] - right_p3_previous[1])
            right_p3_y.pop(TIME_WINDOW)
            right_p3_previous = right_p3
        click_right = right_p1_x + right_p1_y + right_p2_x + right_p2_y + right_p3_x + right_p3_y

        ## If the stop_right_click_pred flag is true then we skip right click detection for the current frame. This flag gets true after we detect a right click and remains true for the next 'TIME_WINDOW - 1' frames. The idea behind this flag is that if a click happens in a frame then there is no chance it will happen again in the next 'TIME_WINDOW - 1' frames. 
        if stop_right_click_pred == True:
            right_counter += 1
            if right_counter == TIME_WINDOW:
                stop_right_click_pred = False
        else:
            ## Execute right click predicition
            input_data = np.zeros((1,6*TIME_WINDOW), 'int8')
            input_data[0] = click_right
            input_data = tf.keras.utils.normalize(input_data, axis=1)
            right_click_pred = right_click_model(input_data)

            ## If the prediction is over CLICK_DETECTION_THRESHOLD then we have a right click
            if right_click_pred > CLICK_DETECTION_THRESHOLD:

                ## If the nearest key to the fingertip is the 'backspace' (idx = 6) then we remove a letter from the clicked_letter_idxs list.
                if idx == 6 and len(clicked_letter_idxs) > 0:
                    clicked_letter_idxs.pop()
                elif idx != 6:
                    clicked_letter_idxs.append(idx)

                ## We change the colour of the clicked letter to blue.   
                frame_bgr[keys_backgrounds[idx][0] + keys_offsets[idx][1] , keys_backgrounds[idx][1] + keys_offsets[idx][0]] = [255,30,30]
                frame_bgr[keys_texts[idx][0] + keys_offsets[idx][1] , keys_texts[idx][1] + keys_offsets[idx][0]] = [3,26,3]
                right_click_pred = 0.0
                stop_right_click_pred = True
                right_counter = 0

    if left_hand_exists:
        ## Get the appropriate left hand landmarks from mediapipe library
        left_tip_landmark = skel_results.multi_hand_landmarks[left_index].landmark[8]
        left_p1_landmark = skel_results.multi_hand_landmarks[left_index].landmark[7]
        left_p2_landmark = skel_results.multi_hand_landmarks[left_index].landmark[6]
        left_p3_landmark = skel_results.multi_hand_landmarks[left_index].landmark[5]

        ## Tranform the above landmarks from relative position to exact position
        left_tip = (int(left_tip_landmark.x*FRAME_WIDTH),
                    int(left_tip_landmark.y*FRAME_HEIGHT - 7))
        left_p1 = (int(left_p1_landmark.x*FRAME_WIDTH),
                    int(left_p1_landmark.y*FRAME_HEIGHT))
        left_p2 = (int(left_p2_landmark.x*FRAME_WIDTH),
                    int(left_p2_landmark.y*FRAME_HEIGHT))
        left_p3 = (int(left_p3_landmark.x*FRAME_WIDTH),
                    int(left_p3_landmark.y*FRAME_HEIGHT))
        
        ## Calculate the nearest key of the left tip landmark and change its colour to red
        distances = [np.sqrt(pow(keys_centers[:,0] - left_tip[0],2) + pow(keys_centers[:,1] - left_tip[1],2))]
        idx = np.argmin(distances)
        frame_bgr[keys_backgrounds[idx][0] + keys_offsets[idx][1] , keys_backgrounds[idx][1] + keys_offsets[idx][0]] = [0,0,255]
        frame_bgr[keys_texts[idx][0] + keys_offsets[idx][1] , keys_texts[idx][1] + keys_offsets[idx][0]] = [3,26,3]
        
        ## Calculate and store the relative movement of left finger-p1
        if left_p1_previous == (-1,-1):
            left_p1_previous = left_p1
        else:
            left_p1_x.insert(0, left_p1[0] - left_p1_previous[0])
            left_p1_x.pop(TIME_WINDOW)
            left_p1_y.insert(0, left_p1[1] - left_p1_previous[1])
            left_p1_y.pop(TIME_WINDOW)
            left_p1_previous = left_p1

        ## Calculate and store the relative movement of right finger-p2
        if left_p2_previous == (-1,-1):
            left_p2_previous = left_p2
        else:
            left_p2_x.insert(0, left_p2[0] - left_p2_previous[0])
            left_p2_x.pop(TIME_WINDOW)
            left_p2_y.insert(0, left_p2[1] - left_p2_previous[1])
            left_p2_y.pop(TIME_WINDOW)
            left_p2_previous = left_p2

        ## Calculate and store the relative movement of left finger-p3
        if left_p3_previous == (-1,-1):
            left_p3_previous = left_p3
        else:
            left_p3_x.insert(0, left_p3[0] - left_p3_previous[0])
            left_p3_x.pop(TIME_WINDOW)
            left_p3_y.insert(0, left_p3[1] - left_p3_previous[1])
            left_p3_y.pop(TIME_WINDOW)
            left_p3_previous = left_p3
        click_left = left_p1_x + left_p1_y + left_p2_x + left_p2_y + left_p3_x + left_p3_y

        ## If the stop_left_click_pred flag is true then we skip left click detection for the current frame. This flag gets true after we detect a left click and remains true for the next 'TIME_WINDOW - 1' frames. The idea behind this flag is that if a click happens in a frame then there is no chance it will happen again in the next 'TIME_WINDOW - 1' frames. 
        if stop_left_click_pred == True:
            left_counter += 1
            if left_counter == TIME_WINDOW:
                stop_left_click_pred = False
        else:
            ## Execute left click predicition
            input_data = np.zeros((1,6*TIME_WINDOW), 'int8')
            input_data[0] = click_left
            input_data = tf.keras.utils.normalize(input_data, axis=1)
            left_click_pred = left_click_model(input_data)

            ## If the prediction is over CLICK_DETECTION_THRESHOLD then we have a left click
            if left_click_pred > CLICK_DETECTION_THRESHOLD:

                ## If the nearest key to the fingertip is the 'backspace' (idx = 6) then we remove a letter from the clicked_letter_idxs list.
                if idx == 6 and len(clicked_letter_idxs) > 0:
                    clicked_letter_idxs.pop()
                elif idx != 6:
                    clicked_letter_idxs.append(idx)

                ## We change the colour of the clicked letter to blue.
                frame_bgr[keys_backgrounds[idx][0] + keys_offsets[idx][1] , keys_backgrounds[idx][1] + keys_offsets[idx][0]] = [255,30,30]
                frame_bgr[keys_texts[idx][0] + keys_offsets[idx][1] , keys_texts[idx][1] + keys_offsets[idx][0]] = [3,26,3]
                left_click_pred = 0.0
                stop_left_click_pred = True
                left_counter = 0

    
    ## Rewrite hand pixels to overwrite the keys
    if right_hand_exists or left_hand_exists:
        for hand_index in hand_indexs:
            (xmin, ymin, xmax, ymax) = hand_box_coordinates[hand_index]
            frame_bgr[ymin + hand_idxs[hand_index][0], xmin + hand_idxs[hand_index][1],:] = hand_values[hand_index]

    ## Uncomment the following lines to display the mediapipe landmarks (tip, p1, p2, p3) on fingers
    # if right_hand_exists:
    #     cv2.circle(frame_bgr,(right_tip[0],right_tip[1]),2, (0,0,255), 2)
    #     cv2.circle(frame_bgr, (right_p1[0],right_p1[1]), 2, (0,255,0), 2)
    #     cv2.circle(frame_bgr, (right_p2[0],right_p2[1]), 2, (0,255,0), 2)
    #     cv2.circle(frame_bgr, (right_p3[0],right_p3[1]), 2, (0,255,0), 2)
    #     cv2.line(frame_bgr,(right_p1[0],right_p1[1]),(right_p2[0],right_p2[1]),(0,255,0),2)
    #     cv2.line(frame_bgr,(right_p2[0],right_p2[1]),(right_p3[0],right_p3[1]),(0,255,0),2)
    # if left_hand_exists:
    #     cv2.circle(frame_bgr,(left_tip[0],left_tip[1]),2, (0,255,0), 2)
    #     cv2.circle(frame_bgr, (left_p1[0],left_p1[1]), 2, (0,0,255), 2)
    #     cv2.circle(frame_bgr, (left_p2[0],left_p2[1]), 2, (0,0,255), 2)
    #     cv2.circle(frame_bgr, (left_p3[0],left_p3[1]), 2, (0,0,255), 2)
    #     cv2.line(frame_bgr,(left_p1[0],left_p1[1]),(left_p2[0],left_p2[1]),(0,0,255),2)
    #     cv2.line(frame_bgr,(left_p2[0],left_p2[1]),(left_p3[0],left_p3[1]),(0,0,255),2)
    
    ## Add text area in frame
    frame_bgr[text_area_background[0] + 330, text_area_background[1] + keyboard_offset_x] = [0,255,0]

    ## Add clicked letters on text area
    for j in range(0,len(clicked_letter_idxs)):
        frame_bgr[keys_texts[clicked_letter_idxs[j]][0] + 317 + divmod(j,45)[0]*15, keys_texts[clicked_letter_idxs[j]][1] + keyboard_offset_x - 7 + divmod(j,45)[1]*9] = [3,26,3]



    cv2.imshow('frame',frame_bgr)
    k = cv2.waitKey(1)
    
    if k == ord('q'): # To exit press 'q'
        break
    elif k == ord('c'): # To clear text area press 'c'  
        clicked_letter_idxs = []


cap.release()
cv2.destroyAllWindows()

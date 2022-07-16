import cv2
# import time
import threading as th
# import pygame
# import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands

hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, draw=True):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, and the output 
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''

    output_image = image.copy()

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks and draw:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image = output_image, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                     thickness=2, circle_radius=2))

    return output_image, results
    
def countFingers(image, results, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image of the hands.
        draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                 output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:     A copy of the input image with the fingers count written, if it was specified.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
    '''
    
    height, width, _ = image.shape
    output_image = image.copy()
    
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    
    # Initialize a dictionary to store the status (True for open and False for close) for each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    

    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        
        # Retrieve the landmarks of the found hand.
        hand_landmarks =  results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]
            
            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                count[hand_label.upper()] += 1
        
        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            count[hand_label.upper()] += 1
     
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:

        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_image, " Total Fingers: ", (10, 25),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (20,255,155), 10, 10)

    return output_image, fingers_statuses, count


def recognizeGestures(image, fingers_statuses, count, draw=True, display=True, command = False):
    '''
    This function will determine the gesture of the left and right hand in the image.
    Args:
        image:            The image of the hands on which the hand gesture recognition is required to be performed.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands. 
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        draw:             A boolean value that is if set to true the function writes the gestures of the hands on the
                          output image, after recognition.
        display:          A boolean value that is if set to true the function displays the resultant image and 
                          returns nothing.
    Returns:
        output_image:   A copy of the input image with the left and right hand recognized gestures written if it was 
                        specified.
        hands_gestures: A dictionary containing the recognized gestures of the right and left hand.
    '''

    output_image = image.copy()
    hands_labels = ['RIGHT', 'LEFT']
    
    # Initialize a dictionary to store the gestures of both hands in the image.
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}

    for hand_index, hand_label in enumerate(hands_labels):
        color = (0, 0, 255)

        # Check the number of fingers that are up, then check which gesture is being made
        
        if count[hand_label] == 1:
            # INDEX = Finger up is: index
            if fingers_statuses[hand_label+'_INDEX']:
                hands_gestures[hand_label] = "INDEX"
                color=(0,255,0)
        
        elif count[hand_label] == 2:
            # PEACE = Fingers up are: middle and index
            if fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_INDEX']:
                hands_gestures[hand_label] = "PEACE"
                color=(0,255,0)
            # WHATUP = Fingers up are: thumb and pinky
            elif fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_PINKY']:
                hands_gestures[hand_label] = "WHATUP"
                color=(0,255,0)

        elif count[hand_label] == 3:
            # THREE PINKY = Fingers up are: middle, ring and pinky
            if fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_RING'] and fingers_statuses[hand_label+'_PINKY']:
                hands_gestures[hand_label] = "THREE PINKY"
                color = (0,255,0)
            # THREE THUMB = Fingers up are: thumb, index and middle
            elif fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_MIDDLE']:
                hands_gestures[hand_label] = "THREE THUMB"
                color = (0,255,0)
            # SPIDERMAN = Fungers up are: thumb, index and pinky
            elif fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_PINKY']:
                hands_gestures[hand_label] = "SPIDERMAN"
                color=(0,255,0)

        elif count[hand_label] == 4:
            # FOUR = All fingers save thumb are up
            if not fingers_statuses[hand_label+'_THUMB']:
                hands_gestures[hand_label] = "FOUR"
                color = (0,255,0)

        elif count[hand_label] == 5:
            # HIGH-FIVE = All fingers are up
            hands_gestures[hand_label] = "HIGH-FIVE"
            color=(0,255,0)
        
        elif count[hand_label] == 0:
            # STOP = All fingers are down
            hands_gestures[hand_label] = "STOP"
            color=(0,255,0)

        if draw:
            cv2.putText(output_image, hand_label +': '+ hands_gestures[hand_label] , (10, (hand_index+1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 5)

    if display:
        cv2.imshow("Gestures", output_image)
    else:
        return output_image, hands_gestures


def repeatedGestures(capturing, capturing_frames, repeated, leeway, previous_gestures, hands_gestures):
    # capturing = startingGesture()
    
    if (repeated["RIGHT"] or repeated["LEFT"] >= 15) and (hands_gestures["RIGHT"] == "HIGH-FIVE" and hands_gestures["LEFT"] == "HIGH-FIVE"):
        # T.start()
        capturing_frames = 30
        capturing = True

    for side in repeated:
        if repeated[side] >= 15:
            if not hands_gestures[side] == "UNKNOWN":
                # print("[INFO] {} hand with gesture {}".format(side, hands_gestures[side]))
                repeated[side] = 0
                if capturing:
                    if hands_gestures[side] == "PEACE":
                        # t1 = th.Thread(target=pdf_stuff.next_Page, args=(pdf_stuff.cur_page,))
                        # t1.start()
                        print("[INFO] Here goes the play slideshow command")
                    elif hands_gestures[side] == "STOP":
                        print("[INFO] Here goes the stop slideshow command")
                    elif hands_gestures[side] == "TRHEE THUMB":
                        print("[INFO] Here goes the next slide command")
                    elif hands_gestures[side] == "TRHEE PINKY":
                        print("[INFO] Here goes the previous slide command")
                    elif hands_gestures[side] == "FOUR":
                        print("[INFO] Here goes the exit presentation command")
                    # print("[INFO] Ahora si!!", hands_gestures[side])
        else:
            if previous_gestures[side] == hands_gestures[side]:
                repeated[side] += 1
            else:
                leeway -= 1
                if leeway == 0:
                    leeway = 4
                    repeated[side] = 0
    capturing_frames -= 1
    if capturing_frames < 0:
        capturing = False
        capturing_frames = 0
    # else:
        # print("[INFO] Remaining frames", capturing_frames)
    return capturing, capturing_frames, repeated, leeway, hands_gestures


def asd(repeated, leeway):
    print("asd")
    print(repeated, leeway)
    repeated = {'RIGHT': 0, 'LEFT': 0}
    leeway = 5
    print(repeated, leeway)
    return repeated, leeway

capturing_frames = 0
number_frames = 15
leeway = 4
repeated = {'RIGHT': 0, 'LEFT': 0}
previous_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}
hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}
capturing = False
# T = th.Thread(target=asd, args=[repeated, leeway])
# repeated, leeway = T.start()
# T.cancel()

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
# camera_video.set(3,1280)
# camera_video.set(4,960)


# t2 = th.Thread(pdf_stuff)
# t2.start()

# Create named window for resizing purposes.
cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    
    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_videos)
    
    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:
            
        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count = countFingers(frame, results, display=False, draw=False)
        
        gesture_frame, hands_gestures = recognizeGestures(frame, fingers_statuses, count, display=False)
        capturing, capturing_frames, repeated, leeway, previous_gestures = repeatedGestures(capturing, capturing_frames, repeated, leeway, previous_gestures, hands_gestures)
        cv2.imshow('Gesture frame', gesture_frame)
        import pdf_stuff

    # Display the frame.
    cv2.imshow('Fingers Counter', frame)
    
    
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()


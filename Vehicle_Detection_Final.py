import cv2
import numpy as np
import math
import time
import datetime

import color_classifier
import model_classifier
from Alpr import ObjectDetection

from flask import Flask,render_template,Response
from pymongo import MongoClient

myclient = MongoClient("mongodb://localhost:27017/")
db= myclient['testingDB']
mycol= db['codiis']

app=Flask(__name__)


color_classifier = color_classifier.Classifier()
model_classifier = model_classifier.Classifier()

def gen_frame():
    od = ObjectDetection()
    start = time.time()
    alpr_classes = []
    with open("./YOLOV4-ALPR_MODELS/obj.names", "r") as f:
        for alpr_class_name in f.readlines():
            alpr_class_name = alpr_class_name.strip()
            alpr_classes.append(alpr_class_name)

    classes = []
    with open("dnn_model/classes.txt", "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)

    # yolo = cv2.dnn.readNetFromDarknet("dnn_model/yolov4.cfg", "dnn_model/yolov4.weights")
    yolo = cv2.dnn.readNetFromDarknet("dnn_model/yolov3.cfg", "dnn_model/yolov3.weights")

    # cap = cv2.VideoCapture("./Input_Videos/three_bikes.mp4")

    cap = cv2.VideoCapture("./Input_Videos/cars_on_highway.mp4")
    # cap = cv2.VideoCapture("./Input_Videos/los_angeles.avi")
    # cap = cv2.VideoCapture("./Input_Videos/Pexels Videos.avi")
    # cap = cv2.VideoCapture("./Input_Videos/input_video.mp4")
    # cap = cv2.VideoCapture("./Input_Videos/white_car1.avi")
    # cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize count
    count = 0
    center_points_prev_frame = []
    tracking_objects = {}
    track_id = 0

    allowed_objects = ['bicycle', 'car', 'motorbike', 'bus', 'truck']

    while True:
        ret, frame = cap.read()
        height, width, layers = frame.shape
        count += 1
        if not ret:
            break

        center_points_cur_frame =[]
        datat = 'Time: ' + str(datetime.datetime.now())
        cv2.rectangle(frame, (0, 0), (320, 60), (0, 0, 250), -1)
        cv2.putText(frame, datat, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # ..............................................................................................
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        yolo.setInput(blob)

        # determine only the *output* layer names that we need from YOLO
        output_layers = yolo.getUnconnectedOutLayersNames()
        outputs = yolo.forward(output_layers)
        # ...............................................................................................

        # --------------------------------------------------------------------------
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        class_ids = []
        confidences = []
        boxes = []

        # loop over each of the layer outputs
        for output in outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # ..............................ALPR.........................................................................................
        (alpr_class_ids, alpr_scores, alpr_boxes) = od.detect(frame)
        for alpr_class_id, alpr_score, alpr_box in zip(alpr_class_ids, alpr_scores, alpr_boxes):
            (xp, yp, wp, hp) = alpr_box

            alpr_confidence = str(round(alpr_scores[alpr_class_id], 2))
            # print("alpr_confidence", alpr_confidence)
            # print("alpr_box: ", alpr_box)

            alpr_class_name = str(alpr_classes[alpr_class_id])

            # print("plate: ",alpr_class_name)

            alpr_label = alpr_class_name + ": " + alpr_confidence
            cv2.rectangle(frame, (xp - 45, yp + hp), (xp + 150, yp + hp + 30), (255, 0, 0), -1)
            cv2.rectangle(frame, (xp, yp), (xp + wp, yp + hp), (0, 0, 250), 2)
            cv2.putText(frame, alpr_label, (xp - 40, yp + hp + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 250, 250), 2)
    # ......................................................................................................................................

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ensure at least one detection exists
        if len(indexes) > 0:
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                # extract the bounding box coordinates
                x, y, w, h = boxes[i]

                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))

                if label in allowed_objects:
                    cx = int((x + x + w) / 2)
                    cy = int(((y) + (y) + ((7 * h) / 6)) / 2)
                    center_points_cur_frame.append((cx, cy))

                    cv2.rectangle(frame, (x, y - 2), (x + 180, y - 90), (255, 0, 0), 1)
                    cv2.rectangle(frame, (x + 1, y - 3), (x + 179, y - 89), (0, 255, 255), -1)
                    cv2.putText(frame, label + ": " + confidence, (x + 4, y - 70), font, 0.6, (255, 0, 0), 2)


                    # draw a bounding box rectangle and label on the image
                    if class_ids[i] == 2:
                        color_result = color_classifier.predict(frame[max(y, 0):y + h, max(x, 0):x + w])
                        model_result = model_classifier.predict(frame[max(y, 0):y + h, max(x, 0):x + w])

                        color_text = "{}: {:.4f}".format(color_result[0]['color'], float(color_result[0]['prob']))
                        cv2.putText(frame, color_text, (x + 4, y - 10), font, 0.6, (255, 0, 0), 2)

                        make_text = "{}: {:.4f}".format(model_result[0]['make'], float(model_result[0]['prob']))
                        cv2.putText(frame, make_text, (x + 4 , y - 50), font, 0.6, (255, 0, 0), 2)
                        model_text = model_result[0]['model']
                        cv2.putText(frame, model_text, (x + 4, y- 30 ), font, 0.6, (255, 0, 0), 2)
                        # print("[OUTPUT] {}, {}, {}".format(make_text, model_text, color_text))

                        vehicle_details = {"Time_and_Date":datat, "Vehicle_type": label, 'CAR_Make': make_text, 'CAR_Model': model_text,
                                       'CAR_Color': color_text, "Vehicle_plate": alpr_class_name}
                        mycol.insert_one(vehicle_details)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 250, 50), 2)
                    else:
                        color_result = color_classifier.predict(frame[max(y, 0):y + h, max(x, 0):x + w])
                        color_text = "{}: {:.4f}".format(color_result[0]['color'], float(color_result[0]['prob']))

                        cv2.putText(frame, color_text, (x + 4, y - 10), font, 0.6, (255, 0, 0), 2)
                        vehicle_details = {"Time_and_Date":datat, "Vehicle_type": label, 'Vehicle_color':color_text, "Vehicle_plate": alpr_class_name}

                        mycol.insert_one(vehicle_details)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (130, 250, 50), 2)

        # .......................................................................
        # Only at the beginning we compare previous and current frame
        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    if distance < 20:
                        tracking_objects[track_id] = pt
                        track_id += 1
        else:
            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # Update IDs position
                    if distance < 20:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for pt in center_points_cur_frame:
                tracking_objects[track_id] = pt
                track_id += 1

        # TRACKING
        for object_id, pt in tracking_objects.items():

            # cv2.circle(frame, pt , 7, (255, 255, 255), -1)
            cv2.circle(frame, pt, 12, (255, 255, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0]-6, pt[1]+ 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2)
        # ............................................................................................................

        # cv2.imshow("Result", frame)

        # Make a copy of the points
        center_points_prev_frame = center_points_cur_frame.copy()
        end = time.time()

        time_diff = end - start
        fps = 1 / time_diff
        fps_text = "FPS:{:.2f}".format(fps)
        start = end

        cv2.putText(frame, fps_text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # show timing information
        # print("[INFO] classifier took {:.6f} seconds".format(end - start))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # cv2.imshow("Frame", frame)
        # cv2.imshow("Result", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

    # cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()
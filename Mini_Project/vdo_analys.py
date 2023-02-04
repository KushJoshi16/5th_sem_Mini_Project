import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *


list_of_vehicles = ["bicycle","car","motorbike","bus","truck", "train"]

FRAMES_BEFORE_CURRENT = 10  
inputWidth, inputHeight = 416, 416

LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()

np.random.seed(16)
COLORS = np.random.randint(0,255,size=(len(LABELS),3), dtype=np.uint8)


def displayVehicleCount(frame, vehicle_count):
    cv2.putText(
        frame,
        'Detect Vehicles: ' + str(vehicle_count),
        (20,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0xFF,0),
        2,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )


def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
    x1_line, y1_line, x2_line, y2_line = line_coordinates

    if (x_mid_point>=x1_line and x_mid_point <= x2_line + 5) and (x_mid_point>=y1_line and x_mid_point <= y2_line + 5):
        return True
    return False

def displayFPS(start_time, num_frames,videoStream):
    fps = None
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = videoStream.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = videoStream.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        return start_time,num_frames


def drawDetectionBoxes(indcs, boxes, classIDs, confidences, frame):
    if len(indcs)>0:
        for i in indcs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame,(x,y),(x+w,y+h), color, 2)
            text = f"{LABELS[classIDs[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.circle(frame, (x + (w//2), y + (h//2)), 2, (0,0xFF,0), thickness=2)

def initializeVideoWriter(video_width, video_height, videoStream):
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps, (video_width,video_height), True)

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf

    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:
            continue
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX,centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]
    if (dist>(max(width,height)/2)):
        return False
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y+ (h//2)

            if(LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = vehicle_count
                if (not boxInPreviousFrames(previous_frame_detections, (centerX,centerY,w,h), current_detections)):
                    vehicle_count += 1
                ID = current_detections.get((centerX,centerY))

                if(list(current_detections.values()).count(ID)>1):
                    current_detections[(centerX,centerY)] = vehicle_count
                    vehicle_count += 1
                cv2.putText(frame, str(ID), (centerX,centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

    return vehicle_count, current_detections
init_kj()


print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if USE_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]

videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

x1_line = 0
y1_line = (video_height+200)//2
x2_line = video_width
y2_line = (video_height+200)//2

#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]

num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())

while True:
	print("================NEW FRAME================")
	num_frames+= 1
	print("FRAME:\t", num_frames)
	# Initialization for each iteration
	boxes, confidences, classIDs = [], [], [] 
	vehicle_crossed_line_flag = False 

	#Calculating fps each second
	start_time, num_frames = displayFPS(start_time, num_frames, videoStream)
	# read the next frame from the file
	(grabbed, frame) = videoStream.read()
	
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for i, detection in enumerate(output):
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > preDefinedConfidence:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
                            
				#Printing the info of the detection
				#print('\nName:\t', LABELS[classID],
					#'\t|\tBOX:\t', x,y)

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
		preDefinedThreshold)

	# Draw detection box 
	drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

	vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)

	# 	# # Changing line color to green if a vehicle in the frame has crossed the line 
	# if vehicle_crossed_line_flag:
	# 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0xFF, 0), 2)
	# # Changing line color to red if a vehicle in the frame has not crossed the line 
	# else:
	# 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 0xFF), 2)

	# Display Vehicle Count if a vehicle has passed the line

	displayVehicleCount(frame, vehicle_count)

    # write the output frame to disk
	writer.write(frame)

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	
	
	# Updating with the current frame detections
	previous_frame_detections.pop(0)
	previous_frame_detections.append(current_detections)

print("[INFO] cleaning up...")
writer.release()
videoStream.release()

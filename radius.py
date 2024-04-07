

# import cv2
# import numpy as np

# # Read the image
# image = cv2.imread('frame_3752.jpg')

# # Convert image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding
# _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# # Show the result
# cv2.imshow('Result', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# for checking value of pixel at each point
# import cv2

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         pixel_value = gray[y, x]  # Access pixel value at (x, y) in the grayscale image
#         print("Pixel value at (x={}, y={}): {}".format(x, y, pixel_value))

# # Read the image
# image = cv2.imread('frame_3752.jpg')

# # Convert image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Display the grayscale image
# cv2.imshow('Grayscale Image', gray)

# # Set mouse callback function
# cv2.setMouseCallback('Grayscale Image', mouse_callback)

# # Wait for a mouse click and print pixel value
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# line on image but reload version
# import cv2
# import numpy as np

# # Read the image
# image = cv2.imread('frame_3752.jpg', cv2.IMREAD_GRAYSCALE)

# # Define lower and upper bounds for the pixel value range you want to keep
# lower_bound = 174
# upper_bound = 190

# # Threshold the image to keep only the pixel values within the specified range
# binary_image = cv2.inRange(image, lower_bound, upper_bound)

# # Show the original grayscale image
# cv2.imshow('Original Grayscale Image', image)

# # Show the thresholded image
# cv2.imshow('Thresholded Image', binary_image)

# # Global variables to store the clicked points
# point1 = None
# point2 = None
# clicked = False

# def mouse_callback(event, x, y, flags, param):
#     global point1, point2, clicked

#     if event == cv2.EVENT_LBUTTONDOWN:
#         if not clicked:
#             point1 = (x, y)
#             clicked = True
#         else:
#             point2 = (x, y)
#             clicked = False
#             distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
#             print("Pixel-to-pixel distance: {:.2f}".format(distance))

#             # Draw the line on the image
#             cv2.line(image, point1, point2, (0, 255, 0), 2)
#             cv2.imshow('Image', image)

# # Read the image
# # image = cv2.imread('your_image.jpg')
# clone = binary_image.copy()

# # Create a window and set mouse callback
# cv2.imshow('Image', binary_image)
# cv2.setMouseCallback('Image', mouse_callback)

# # Wait for a key press and close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()





#modified version of distance line
import cv2
import numpy as np

# Read the image
image = cv2.imread('frame_3752.jpg', cv2.IMREAD_GRAYSCALE)

# Define lower and upper bounds for the pixel value range you want to keep
lower_bound = 174
upper_bound = 190

# Threshold the image to keep only the pixel values within the specified range
binary_image = cv2.inRange(image, lower_bound, upper_bound)

# Show the original grayscale image
cv2.imshow('Original Grayscale Image', image)

# Show the thresholded image
cv2.imshow('Thresholded Image', binary_image)

# Global variables to store the starting and ending points of the line
start_point = None
end_point = None
drawing = False

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False

# Create a black image
image = binary_image

# Create a window and set the mouse callback function
cv2.namedWindow('Drawing Board')
cv2.setMouseCallback('Drawing Board', mouse_callback)

while True:
    # Copy the original image to draw on
    drawing_board = image.copy()

    # Draw the line if the start and end points are defined
    if start_point is not None and end_point is not None:
        cv2.line(drawing_board, start_point, end_point, (0, 0, 255), 2)
        distance = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        cv2.putText(drawing_board, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the drawing board
    cv2.imshow('Drawing Board', drawing_board)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Exit loop if 'q' is pressed
    if key == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()





# import cv2
# import numpy as np

# # Read the image
# image = cv2.imread('frame_3752.jpg', cv2.IMREAD_GRAYSCALE)

# # Define lower and upper bounds for the pixel value range you want to keep
# lower_bound = 100
# upper_bound = 200

# # Threshold the image to keep only the pixel values within the specified range
# binary_image = cv2.inRange(image, lower_bound, upper_bound)

# # Perform morphological operations to build boundary between regions
# kernel = np.ones((3,3), np.uint8)
# boundary_image = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)

# # Show the original grayscale image
# cv2.imshow('Original Grayscale Image', image)

# # Show the thresholded image
# cv2.imshow('Thresholded Image', binary_image)

# # Show the boundary image
# cv2.imshow('Boundary Image', boundary_image)

# # Wait for a key press and close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# creating circular things
# import cv2
# import numpy as np

# # Load YOLO model and configuration
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# # Load the image
# image = cv2.imread('frame_3752.jpg')

# # Resize and preprocess the image for YOLO
# blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
# net.setInput(blob)
# outs = net.forward(output_layers)

# # Process the detections
# conf_threshold = 0.5
# nms_threshold = 0.4
# height, width, channels = image.shape
# mask = np.zeros((height, width), dtype=np.uint8)

# for out in outs:
#     for detection in out:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > conf_threshold and classes[class_id] == "car":  # Assuming cars are circular/spherical objects
#             center_x = int(detection[0] * width)
#             center_y = int(detection[1] * height)
#             w = int(detection[2] * width)
#             h = int(detection[3] * height)

#             # Extract the region of interest based on the bounding box
#             roi = image[max(0, center_y - h // 2):min(center_y + h // 2, height),
#                         max(0, center_x - w // 2):min(center_x + w // 2, width)]

#             # Create a mask based on the pixel intensity range
#             lower_intensity = 174
#             upper_intensity = 190
#             mask_roi = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), lower_intensity, upper_intensity)

#             # Apply the mask to the original image
#             mask[max(0, center_y - h // 2):min(center_y + h // 2, height),
#                  max(0, center_x - w // 2):min(center_x + w // 2, width)] = mask_roi

# # Apply morphological operations to clean up the mask
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# # Display the result
# cv2.imshow('Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







#mask making

# import os

# import cv2


# input_dir = './data/images'
# output_dir = './data/labels'

# for j in os.listdir(input_dir):
#     image_path = os.path.join(input_dir, j)
#     # load the binary mask and get its contours
#     mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

#     H, W = mask.shape
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # convert the contours to polygons
#     polygons = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 200:
#             polygon = []
#             for point in cnt:
#                 x, y = point[0]
#                 polygon.append(x / W)
#                 polygon.append(y / H)
#             polygons.append(polygon)

#     # print the polygons
#     with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
#         for polygon in polygons:
#             for p_, p in enumerate(polygon):
#                 if p_ == len(polygon) - 1:
#                     f.write('{}\n'.format(p))
#                 elif p_ == 0:
#                     f.write('0 {} '.format(p))
#                 else:
#                     f.write('{} '.format(p))

#         f.close()



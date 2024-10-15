import cv2
import numpy as np
import matplotlib.pyplot as plt 

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)         
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

# def average_slope_intercept(image, lines):
#     left_fit    = []
#     right_fit   = []
#     if lines is None:
#         return None
#     for line in lines:
#             x1, y1, x2, y2  = line.reshape(4)
#             parameters = np.polyfit((x1,x2), (y1,y2), 1)
#             slope = parameters[0]
#             intercept = parameters[1]
#             if slope < 0: 
#                 left_fit.append((slope, intercept))
#             else:
#                 right_fit.append((slope, intercept))
                
#     left_fit_average  = np.average(left_fit, axis=0)
#     right_fit_average = np.average(right_fit, axis=0)
    
#     left_line  = make_coordinates(image, left_fit_average)
#     right_line = make_coordinates(image, right_fit_average)
    
#     return np.array([left_line,right_line])
    
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < 0:  # Negative slope (left lane)
            left_fit.append((slope, intercept))
        else:          # Positive slope (right lane)
            right_fit.append((slope, intercept))
    
    # Ensure that there is at least one line on each side before averaging
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    else:
        left_line = None  # Handle the case where no left lines were detected
    
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    else:
        right_line = None  # Handle the case where no right lines were detected
    
    # Return only the lines that exist
    lines = []
    if left_line is not None:
        lines.append(left_line)
    if right_line is not None:
        lines.append(right_line)
    
    return np.array(lines)


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255) 
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
  
def canny(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blurred_image,50,150)
    return canny
        

# image = cv2.imread('test_image.jpg')
# copy_image = np.copy(image)
# canny = canny(copy_image)
# cropped_image =  region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# averaged_lines =  average_slope_intercept(copy_image,lines)
# line_image = display_lines(copy_image,averaged_lines)
# final_image = cv2.addWeighted(copy_image,0.8,line_image,1,1)
# cv2.imshow("lanes",final_image)
# cv2.waitKey(0)
  
cap = cv2.VideoCapture("test_video.mp4")
while(cap.isOpened()):
    _,frame = cap.read()
    canny_img = canny(frame)
    cropped_image =  region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines =  average_slope_intercept(frame,lines)
    line_image = display_lines(frame,averaged_lines)
    final_image = cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow("lanes",final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 

# coding: utf-8

# In[ ]:

# Importing the necessary packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import datetime
from math import sqrt
get_ipython().magic('matplotlib inline')

# Packages below needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[ ]:

# Constants
kTestImagesRelativeInputPathDir = "test_images/"
kTestImagesRelativeOutputPathDir = "test_images_output/"
kTestVideosRelativeInputPathDir = "test_videos/"
kTestVideosRelativeOutputPathDir = "test_videos_output/"

# Global variable(s)
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)

# This boolean is used to determine if we need to produce intermediate image artifacts, after each processing operation like Gray Scaling etc.
# These artifacts are useful for debugging
# The artifacts once generated are placed within the 'test_images_output' folder
generateIntermediateArtifacts = False


# In[ ]:

# Helper method(s)
def get_region_of_interest_vertices(image):
    xsize = image.shape[1]
    ysize = image.shape[0]
    y_offset = 42
    left_bottom = [120, ysize]
    right_bottom = [850, ysize]
    left_top = [480, ysize/2 + y_offset]
    right_top = [490, ysize/2 + y_offset]
    region_of_interest_vertices = np.array([[left_top,right_top,right_bottom,left_bottom]], dtype=np.int32)
    return region_of_interest_vertices

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
        
 try:
    global global_previous_left_lane_bottom_roi_intersection_point
    global global_previous_right_lane_bottom_roi_intersection_point
    leftLaneLineMaxLength = 0
    rightLaneLineMaxLength = 0
    longestLeftLaneLine = (0,0,0,0)
    longestRightLaneLine = (0,0,0,0)

    for line in lines:
        for x1,y1,x2,y2 in line:
            dy = y2 - y1
            dx = x2 - x1
            slope = dy / dx
            lineLength = sqrt(dy**2 + dx**2)
            
            if slope < 0:
               if(lineLength > leftLaneLineMaxLength):
                    leftLaneLineMaxLength = lineLength
                    longestLeftLaneLine = line
                    
            else:
               if(lineLength > rightLaneLineMaxLength):
                    rightLaneLineMaxLength = lineLength
                    longestRightLaneLine = line                
    
    region_of_interest_vertices = get_region_of_interest_vertices(img)
    region_of_interest_left_top = region_of_interest_vertices[0][0]
    region_of_interest_right_top = region_of_interest_vertices[0][1]
    region_of_interest_right_bottom = region_of_interest_vertices[0][2]
    region_of_interest_left_bottom = region_of_interest_vertices[0][3]
    
    region_of_interest_top_line = get_line(region_of_interest_left_top, region_of_interest_right_top)
    region_of_interest_bottom_line = get_line(region_of_interest_left_bottom, region_of_interest_right_bottom)

    longest_left_lane_line = get_line([longestLeftLaneLine[0][0],longestLeftLaneLine[0][1]],[longestLeftLaneLine[0][2],longestLeftLaneLine[0][3]])
    
    longest_right_lane_line = get_line([longestRightLaneLine[0][0],longestRightLaneLine[0][1]],[longestRightLaneLine[0][2],longestRightLaneLine[0][3]])
    
    top_left_intersection_point = intersection(region_of_interest_top_line, longest_left_lane_line)
    bottom_left_intersection_point = intersection(region_of_interest_bottom_line, longest_left_lane_line)    
    if global_previous_left_lane_bottom_roi_intersection_point == (0,0):
        global_previous_left_lane_bottom_roi_intersection_point = bottom_left_intersection_point
        
    top_right_intersection_point = intersection(region_of_interest_top_line, longest_right_lane_line)
    bottom_right_intersection_point = intersection(region_of_interest_bottom_line, longest_right_lane_line)
    if global_previous_right_lane_bottom_roi_intersection_point == (0,0):
        global_previous_right_lane_bottom_roi_intersection_point = bottom_right_intersection_point
    
    if (top_left_intersection_point and bottom_left_intersection_point and top_right_intersection_point and bottom_right_intersection_point):
        cv2.line(img, tuple(top_left_intersection_point), tuple(global_previous_left_lane_bottom_roi_intersection_point), color, thickness)
        cv2.line(img, tuple(top_right_intersection_point), tuple(global_previous_right_lane_bottom_roi_intersection_point), color, thickness)

      
 except TypeError: 
        print("Ignoring sporadic type error as noise.")
    
def get_line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return (int(x),int(y))
    else:
        return False

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1, λ=0):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    currentTime = datetime.datetime.now()
    currentTimeString = str(currentTime)
    # Important to not modify the original image, but instead work on it's copy
    original_image_copy = np.copy(image)
    greyscale_image = grayscale(original_image_copy)    
    gaussian_blurred_image = gaussian_blur(greyscale_image,5)
    canny_image = canny(gaussian_blurred_image,50,150)
    
    xsize = canny_image.shape[1]
    ysize = canny_image.shape[0]
    y_offset = 42
    left_bottom = [120, ysize]
    right_bottom = [850, ysize]
    left_top = [480, ysize/2 + y_offset]
    right_top = [490, ysize/2 + y_offset]
    region_of_interest_vertices = np.array([[left_top,right_top,right_bottom,left_bottom]], dtype=np.int32)
    region_of_interest_image = region_of_interest(canny_image,region_of_interest_vertices)
    hough_lines_image = hough_lines(region_of_interest_image, 2, np.pi/180, 15, 4, 10)
    original_image_overlaid_with_lanes = weighted_img(hough_lines_image,original_image_copy)
    
    if generateIntermediateArtifacts == True:
        original_image_copy_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_1_original_image_copy.jpg" 
        plt.imshow(original_image_copy,cmap='gray')
        plt.savefig(original_image_copy_filename)
        
        greyscale_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_2_grayscale_image.jpg" 
        plt.imshow(greyscale_image,cmap='gray')
        plt.savefig(greyscale_image_filename)
        
        gaussian_blurred_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_3_gaussian_blurred_image.jpg" 
        plt.imshow(gaussian_blurred_image,cmap='gray')
        plt.savefig(gaussian_blurred_image_filename)
        
        canny_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_4_canny_image.jpg" 
        plt.imshow(canny_image,cmap='gray')
        plt.savefig(canny_image_filename)
        
        region_of_interest_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_5_region_of_interest_image.jpg" 
        plt.imshow(region_of_interest_image,cmap='gray')
        plt.savefig(region_of_interest_image_filename)
        
        hough_lines_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_6_hough_lines_image.jpg" 
        plt.imshow(hough_lines_image,cmap='gray')
        plt.savefig(hough_lines_image_filename)       

    return original_image_overlaid_with_lanes


# In[16]:

# # Using the Pipeline above to process image(s) - solidWhiteCurve.jpg
generateIntermediateArtifacts = False
imageFile =  "solidWhiteCurve.jpg"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_file_relative_path = kTestImagesRelativeInputPathDir + imageFile
output_file_relative_path = kTestImagesRelativeOutputPathDir + imageFile
image_with_detected_lanes = process_image(mpimg.imread(input_file_relative_path))
plt.imshow(image_with_detected_lanes)
plt.savefig(output_file_relative_path)


# In[17]:

# # Using the Pipeline above to process image(s) - solidWhiteRight.jpg
generateIntermediateArtifacts = False
imageFile =  "solidWhiteRight.jpg"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_file_relative_path = kTestImagesRelativeInputPathDir + imageFile
output_file_relative_path = kTestImagesRelativeOutputPathDir + imageFile
image_with_detected_lanes = process_image(mpimg.imread(input_file_relative_path))
plt.imshow(image_with_detected_lanes)
plt.savefig(output_file_relative_path)


# In[18]:

# # Using the Pipeline above to process image(s) - solidYellowCurve.jpg
generateIntermediateArtifacts = False
imageFile =  "solidYellowCurve.jpg"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_file_relative_path = kTestImagesRelativeInputPathDir + imageFile
output_file_relative_path = kTestImagesRelativeOutputPathDir + imageFile
image_with_detected_lanes = process_image(mpimg.imread(input_file_relative_path))
plt.imshow(image_with_detected_lanes)
plt.savefig(output_file_relative_path)


# In[19]:

# # Using the Pipeline above to process image(s) - solidYellowCurve2.jpg
generateIntermediateArtifacts = False
imageFile =  "solidYellowCurve2.jpg"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_file_relative_path = kTestImagesRelativeInputPathDir + imageFile
output_file_relative_path = kTestImagesRelativeOutputPathDir + imageFile
image_with_detected_lanes = process_image(mpimg.imread(input_file_relative_path))
plt.imshow(image_with_detected_lanes)
plt.savefig(output_file_relative_path)


# In[20]:

# # Using the Pipeline above to process image(s) - solidYellowLeft.jpg
generateIntermediateArtifacts = False
imageFile =  "solidYellowLeft.jpg"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_file_relative_path = kTestImagesRelativeInputPathDir + imageFile
output_file_relative_path = kTestImagesRelativeOutputPathDir + imageFile
image_with_detected_lanes = process_image(mpimg.imread(input_file_relative_path))
plt.imshow(image_with_detected_lanes)
plt.savefig(output_file_relative_path)


# In[21]:

# # Using the Pipeline above to process image(s) - whiteCarLaneSwitch.jpg
generateIntermediateArtifacts = False
imageFile =  "whiteCarLaneSwitch.jpg"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_file_relative_path = kTestImagesRelativeInputPathDir + imageFile
output_file_relative_path = kTestImagesRelativeOutputPathDir + imageFile
image_with_detected_lanes = process_image(mpimg.imread(input_file_relative_path))
plt.imshow(image_with_detected_lanes)
plt.savefig(output_file_relative_path)


# In[22]:

# # Using the Pipeline above to process video(s) - solidWhiteRight.mp4
generateIntermediateArtifacts = False
videoFile =  "solidWhiteRight.mp4"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_videofile_relative_path = kTestVideosRelativeInputPathDir + videoFile
output_file_relative_path = kTestVideosRelativeOutputPathDir + videoFile
input_clip = VideoFileClip(input_videofile_relative_path)
output_clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time output_clip.write_videofile(output_file_relative_path, audio=False)')


# In[24]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output_file_relative_path))


# In[29]:

# # Using the Pipeline above to process video(s) - solidYellowLeft.mp4
generateIntermediateArtifacts = False
videoFile =  "solidYellowLeft.mp4"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_videofile_relative_path = kTestVideosRelativeInputPathDir + videoFile
output_file_relative_path = kTestVideosRelativeOutputPathDir + videoFile
input_clip = VideoFileClip(input_videofile_relative_path)
output_clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time output_clip.write_videofile(output_file_relative_path, audio=False)')


# In[30]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output_file_relative_path))


# In[32]:


def hough_lines_challenge(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_challenge(line_img, lines)
    return line_img

def process_image_challenge(image):
    currentTime = datetime.datetime.now()
    currentTimeString = str(currentTime)
    # Important to not modify the original image, but instead work on it's copy
    original_image_copy = np.copy(image)
    greyscale_image = grayscale(original_image_copy)    
    gaussian_blurred_image = gaussian_blur(greyscale_image,5)
    canny_image = canny(gaussian_blurred_image,50,150)
        
    region_of_interest_image = region_of_interest(canny_image,get_region_of_interest_vertices_challenge(canny_image))
    
    hough_lines_image = hough_lines_challenge(region_of_interest_image, 2, np.pi/180, 15, 4, 10)

    original_image_overlaid_with_lanes = weighted_img(hough_lines_image,original_image_copy)
    
    if generateIntermediateArtifacts == True:
        original_image_copy_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_1_original_image_copy.jpg" 
        plt.imshow(original_image_copy,cmap='gray')
        plt.savefig(original_image_copy_filename)
        
        greyscale_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_2_grayscale_image.jpg" 
        plt.imshow(greyscale_image,cmap='gray')
        plt.savefig(greyscale_image_filename)
        
        gaussian_blurred_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_3_gaussian_blurred_image.jpg" 
        plt.imshow(gaussian_blurred_image,cmap='gray')
        plt.savefig(gaussian_blurred_image_filename)
        
        canny_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_4_canny_image.jpg" 
        plt.imshow(canny_image,cmap='gray')
        plt.savefig(canny_image_filename)
        
        region_of_interest_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_5_region_of_interest_image.jpg" 
        plt.imshow(region_of_interest_image,cmap='gray')
        plt.savefig(region_of_interest_image_filename)
        
        hough_lines_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_6_hough_lines_image.jpg" 
        plt.imshow(hough_lines_image,cmap='gray')
        plt.savefig(hough_lines_image_filename)
        
    return original_image_overlaid_with_lanes

def get_region_of_interest_vertices_challenge(image):
    xsize = image.shape[1]
    ysize = image.shape[0]
    y_offset = 52
    bottom_offset = 30
    left_bottom = [240, ysize - bottom_offset]
    right_bottom = [1100, ysize - bottom_offset]
    left_top = [630, ysize/2 + y_offset]
    right_top = [800, ysize/2 + y_offset]
    region_of_interest_vertices = np.array([[left_top,right_top,right_bottom,left_bottom]], dtype=np.int32)
    return region_of_interest_vertices

def draw_lines_challenge(img, lines, color=[255, 0, 0], thickness=10):
        
 try:
    global global_previous_left_lane_bottom_roi_intersection_point
    global global_previous_right_lane_bottom_roi_intersection_point
    leftLaneLineMaxLength = 0
    rightLaneLineMaxLength = 0
    longestLeftLaneLine = (0,0,0,0)
    longestRightLaneLine = (0,0,0,0)

    for line in lines:
        for x1,y1,x2,y2 in line:
            dy = y2 - y1
            dx = x2 - x1
            slope = dy / dx
            lineLength = sqrt(dy**2 + dx**2)
            
            if slope < 0:
               if(lineLength > leftLaneLineMaxLength):
                    leftLaneLineMaxLength = lineLength
                    longestLeftLaneLine = line
                    
            elif slope >= 0:
               if(lineLength > rightLaneLineMaxLength):
                    rightLaneLineMaxLength = lineLength
                    longestRightLaneLine = line                
    
    region_of_interest_vertices = get_region_of_interest_vertices_challenge(img)
    region_of_interest_left_top = region_of_interest_vertices[0][0]
    region_of_interest_right_top = region_of_interest_vertices[0][1]
    region_of_interest_right_bottom = region_of_interest_vertices[0][2]
    region_of_interest_left_bottom = region_of_interest_vertices[0][3]
    
    region_of_interest_top_line = get_line(region_of_interest_left_top, region_of_interest_right_top)
    region_of_interest_bottom_line = get_line(region_of_interest_left_bottom, region_of_interest_right_bottom)

    longest_left_lane_line = get_line([longestLeftLaneLine[0][0],longestLeftLaneLine[0][1]],[longestLeftLaneLine[0][2],longestLeftLaneLine[0][3]])
    
    longest_right_lane_line = get_line([longestRightLaneLine[0][0],longestRightLaneLine[0][1]],[longestRightLaneLine[0][2],longestRightLaneLine[0][3]])
    
    top_left_intersection_point = intersection(region_of_interest_top_line, longest_left_lane_line)
    bottom_left_intersection_point = intersection(region_of_interest_bottom_line, longest_left_lane_line)    
    if global_previous_left_lane_bottom_roi_intersection_point == (0,0):
        global_previous_left_lane_bottom_roi_intersection_point = bottom_left_intersection_point
        
    top_right_intersection_point = intersection(region_of_interest_top_line, longest_right_lane_line)
    bottom_right_intersection_point = intersection(region_of_interest_bottom_line, longest_right_lane_line)
    if global_previous_right_lane_bottom_roi_intersection_point == (0,0):
        global_previous_right_lane_bottom_roi_intersection_point = bottom_right_intersection_point
    
    if (top_left_intersection_point and bottom_left_intersection_point and top_right_intersection_point and bottom_right_intersection_point):
        cv2.line(img, tuple(top_left_intersection_point), tuple(global_previous_left_lane_bottom_roi_intersection_point), color, thickness)
        cv2.line(img, tuple(top_right_intersection_point), tuple(global_previous_right_lane_bottom_roi_intersection_point), color, thickness)

 except TypeError: 
        print("Ignoring sporadic type error as noise.")
        
def process_image_challenge(image):
    currentTime = datetime.datetime.now()
    currentTimeString = str(currentTime)
    # Important to not modify the original image, but instead work on it's copy
    original_image_copy = np.copy(image)
    greyscale_image = grayscale(original_image_copy)    
    gaussian_blurred_image = gaussian_blur(greyscale_image,5)
    canny_image = canny(gaussian_blurred_image,50,150)

    region_of_interest_image = region_of_interest(canny_image,get_region_of_interest_vertices_challenge(canny_image))
    hough_lines_image = hough_lines_challenge(region_of_interest_image, 2, np.pi/180, 15, 4, 10)
    original_image_overlaid_with_lanes = weighted_img(hough_lines_image,original_image_copy)
    
    if generateIntermediateArtifacts == True:
        original_image_copy_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_1_original_image_copy.jpg" 
        plt.imshow(original_image_copy,cmap='gray')
        plt.savefig(original_image_copy_filename)
        
        greyscale_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_2_grayscale_image.jpg" 
        plt.imshow(greyscale_image,cmap='gray')
        plt.savefig(greyscale_image_filename)
        
        gaussian_blurred_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_3_gaussian_blurred_image.jpg" 
        plt.imshow(gaussian_blurred_image,cmap='gray')
        plt.savefig(gaussian_blurred_image_filename)
        
        canny_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_4_canny_image.jpg" 
        plt.imshow(canny_image,cmap='gray')
        plt.savefig(canny_image_filename)
        
        region_of_interest_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_5_region_of_interest_image.jpg" 
        plt.imshow(region_of_interest_image,cmap='gray')
        plt.savefig(region_of_interest_image_filename)
        
        hough_lines_image_filename = kTestImagesRelativeOutputPathDir + currentTimeString + "_6_hough_lines_image.jpg" 
        plt.imshow(hough_lines_image,cmap='gray')
        plt.savefig(hough_lines_image_filename)       

    return original_image_overlaid_with_lanes

# # Using the Pipeline above to process video(s) - challenge.mp4
generateIntermediateArtifacts = False
videoFile =  "challenge.mp4"
global_previous_left_lane_bottom_roi_intersection_point = (0,0)
global_previous_right_lane_bottom_roi_intersection_point = (0,0)
input_videofile_relative_path = kTestVideosRelativeInputPathDir + videoFile
output_file_relative_path = kTestVideosRelativeOutputPathDir + videoFile
input_clip = VideoFileClip(input_videofile_relative_path)
output_clip = input_clip.fl_image(process_image_challenge) #NOTE: this function expects color images!!
get_ipython().magic('time output_clip.write_videofile(output_file_relative_path, audio=False)')


# In[33]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output_file_relative_path))


# In[ ]:




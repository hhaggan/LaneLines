#importing the required libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import datetime
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

#Pipeline functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

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

def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

#function to store the images in the output folder
def store_test_images(img, imgname):
    path = 'test_images_output/'
    cv2.imwrite(os.path.join(path , imgname+'.jpg'), img)
    return True

#building Pipeline for Images 
os.listdir("test_images/")
images = []
for img_name in os.listdir("test_images/"):
    images = mpimg.imread("test_images/"+img_name)
    print('This image is:', type(images), 'with dimensions:', images.shape)

#setting the parameters
#gaussian Blur Parameters 
kernel_size = 3

#canny Paramters
low_threshold = 80 
high_threshold = 180

#Hough Transformation Parameters
#distance in pixels
rho = 2 
#angular resolution in radians 
theta = np.pi/180
#minimum number of interesections in hough grid
threshold = 15
#minimum number of pixels for a line
min_line_length = 40
#maximum gap between lines
max_line_gap = 100

for img_name in os.listdir("test_images/"):
    img = mpimg.imread("test_images/"+img_name)
    print('This image is:', type(img), 'with dimensions:', img.shape)

    #grayscaling the image
    grimg = grayscale(img)
    
    #identifying the gaussian blur paratmers and applying them
    blur_gray = gaussian_blur(grimg, kernel_size)
    
    edges = canny(img, low_threshold, high_threshold)

    
    #identifying the vertices to project the lanes 
    #applying the vertices to identify the region of interest
    imshape = img.shape
    #the vertices parameters to identify the region of interest
    vertices = np.array([[(100,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)
    
    #identifying the hough lines parameters and applying them
    line_image = np.copy(img)*0
    #applying hough lines parameters
    hough_result_img = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)
    
    #applying weights to the images
    #Create a "color" binary image to combine with line image
    #color_edges = np.dstack((edges, edges, edges)) 
    fimage = weighted_img(hough_result_img, img)
    
    #storing the images in the output folder
    store_test_images(fimage, img_name)

#Pipeline for Vidoe Clips
#defining the process for the Videos

def process_image(image):
    #setting the parameters
    #gaussian Blur Parameters 
    kernel_size = 5 #kernel size doesnt matter much, Canny automatically applies kernel size equal to 5

    #canny Paramters
    low_threshold = 70 
    high_threshold = 150

    #Hough Transformation Parameters
    #distance in pixels
    rho = 4 
    #angular resolution in radians 
    theta = np.pi/180
    #minimum number of interesections in hough grid
    threshold = 100
    #minimum number of pixels for a line
    min_line_length = 140
    #maximum gap between lines
    max_line_gap = 100
    #Transforming the image to gray
    grimg = grayscale(image)
    
    #applying gaussian blur
    blur_gray = gaussian_blur(grimg, kernel_size)
    plt.imshow(blur_gray)
    
    #applying canny
    edges = canny(blur_gray, low_threshold, high_threshold)
    plt.imshow(edges)
    
    line_image = np.copy(image)*0
    
    imshape = image.shape
    #the vertices parameters to identify the region of interest
    vertices = np.array([[(100,imshape[0]),(450, 320), (550, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    #applying the region of interest
    masked_image = region_of_interest(edges, vertices)
    plt.imshow(masked_image)
    
    #applying hough lines
    hough_result_img = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)
    plt.imshow(hough_result_img)
    #applying weights to the images
    fimage = weighted_img(hough_result_img, image)
    plt.imshow(fimage)
    
    return fimage

#first Video:

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
%time white_clip.write_videofile(white_output, audio=False)

#second Video:
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)

#challenge Video:
challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
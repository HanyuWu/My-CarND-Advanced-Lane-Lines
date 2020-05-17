import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os


os.chdir('CarND-Advanced-Lane-Lines')

class Line():
    count = 0;
    def __init__(self): 
        #polynomial coefficients of the last n iterations
        self.left_recent_fit = []
        #polynomial coefficients averaged over the last n iterations
        self.left_best_fit = None  
        #polynomial coefficients for the most recent fit
        self.left_current_fit = [np.array([False])]  
        #polynomial coefficients of the last n iterations
        self.right_recent_fit = []
        #polynomial coefficients averaged over the last n iterations
        self.right_best_fit = None  
        #polynomial coefficients for the most recent fit
        self.right_current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.left_diffs = np.array([0,0,0], dtype='float') 
        self.right_diffs = np.array([0,0,0], dtype='float') 
        Line.count = Line.count+1
    def recentmid(self,a):   # reserve the last 5 x-coordinate of the middle point of the road
        self.recent_mid.append(a)
        if len(self.recent_mid)>5:
            self.recent_mid = self.recent_mid[1:]
    def bestmid(self):     # calculate the mean average of the x-coordinate of the middle point of the road
        self.best_mid = np.mean(self.recent_mid)
    def leftrecentfit(self,a):      # reserve the last 5 polyfit parameters of the left lane
        self.left_recent_fit.append(a)
        if len(self.left_recent_fit)>5:
            self.left_recent_fit = self.left_recent_fit[1:]
    def leftbestfit(self):    # calculate the average of last 5 polyfit parameters of the left lane
        self.left_best_fit = np.mean(self.left_recent_fit,axis = 0)
    def leftcurrentfit(self,a):   # reserve the latest polyfit parameters of the left lane
        self.left_current_fit = a
    def rightrecentfit(self,a):   # reserve the last 5 polyfit parameters of the right lane
        self.right_recent_fit.append(a)
        if len(self.right_recent_fit)>5:
            self.right_recent_fit = self.right_recent_fit[1:]
    def rightbestfit(self):   # calculate the average of last 5 polyfit parameters of the right lane
        self.right_best_fit = np.mean(self.right_recent_fit,axis = 0)
    def rightcurrentfit(self,a):   # reserve the latest polyfit parameters of the right lane
        self.right_current_fit = a
    def radiusofcurvature(self,a):   # reserve the latest radius of curvature
        self.radius_of_curvature = a
    def linebasepos(self,a):    # reserve the offset from the center of the road
        self.line_base_pos = a
    def getdiffs(self):  # get the difference between the last polyfit parameter and the current polyfit parameter
        if len(self.left_recent_fit)>1:
            self.left_diffs = self.left_recent_fit[-1] - self.left_recent_fit[-2]
        if len(self.right_recent_fit)>1:
            self.right_diffs = self.right_recent_fit[-1] - self.right_recent_fit[-2]

# generate a Line class, and we use it to reserve the data we want to use
line = Line()
        
def getundist():
    images = glob.glob('camera_cal\calibration*.jpg')
    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def undist(img, mtx, dist):
    dis = cv2.undistort(img, mtx, dist, None, mtx)
    return dis

def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0,255)):
    if orient == 'x':
        x = 1
        y = 0
    elif orient == 'y':
        x = 0
        y = 1
    gradient = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=sobel_kernel)
    abs_gradient = np.absolute(gradient)
    scaled_gradient = np.uint8(255*abs_gradient/np.max(abs_gradient))
    binary_output = np.zeros_like(scaled_gradient)
    binary_output[(scaled_gradient >= thresh[0]) & (scaled_gradient <= thresh[0])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel/np.max(sobel)*255).astype(np.uint8)
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel >= mag_thresh[0]) & (sobel <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    return left_fitx, right_fitx, ploty , left_fit , right_fit

def search_around_poly(binary_warped, left_fit, right_fit):
    margin = 100
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty , left_fit , right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    return left_fitx, right_fitx, ploty , left_fit , right_fit, leftx, lefty, rightx, righty



images = glob.glob('video_snip\*.jpg')

# get undist parameters
ret, mtx, dist, rvecs, tvecs = getundist()

# set up windows
nwindows = 9
margin = 125
minpix = 30  # minimum number of pixels found in a window to be a estimated point
window_height = np.int(720//nwindows)

# set transform parameters
src = np.float32([[264,686],[1055,686],[699,457],[585,457]])
dst = np.float32([[250,720],[1000,720],[1000,0],[250,0]])
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# For draw function to get rid of the 'for loop'
xlin = np.array([np.linspace(0,1279,1280)])
xmap = np.repeat(xlin,720,axis = 0)
ylin = np.array([np.linspace(0,719,720)])
ymap = np.repeat(ylin,1280,axis = 0)
ymap = np.transpose(ymap)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 3/104 # meters per pixel in y dimension
xm_per_pix = 3.7/750 # meters per pixel in x dimension





def getlines(img):
    global line
    undis_img = undist(img,mtx,dist)
    warped = cv2.warpPerspective(undis_img, M, (1280,720), flags=cv2.INTER_LINEAR)
    hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    # get s channel
    s = hls[:,:,2]
    kernel_size = 3
    s = cv2.GaussianBlur(s,(kernel_size, kernel_size), 0)
    # color threshold mask
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    col_mask = np.zeros((720,1280))
    col_mask[gray>50] = 1
    # gradient threshold
    ksize = 7
    gradx = abs_sobel_thresh(s, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady = abs_sobel_thresh(s, orient='y', sobel_kernel=ksize, thresh=(0, 100))
    mag_binary = mag_thresh(s, sobel_kernel=ksize, mag_thresh=(20, 255))
    dir_binary = dir_threshold(s, sobel_kernel=ksize, thresh=(0, np.pi/3))
    # gradient and color combined
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[col_mask==0]=0



    # set up midpoint and left and right lane base
    raw = combined.copy()
    his = np.sum(raw[raw.shape[0]//2:,:], axis=0)
    midpoint = np.int(his.shape[0]//2)
    leftx_base = np.argmax(his[:midpoint])
    rightx_base = np.argmax(his[midpoint:])+midpoint

    nonzero = raw.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = []
    right_lane_inds = []

    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # if we have polyfit data from the last frame, we use it to detect the current potential lane area
    # for the first detection we speculate the potential area by histogram method
    if len(line.left_recent_fit) == 0:      
        for window in range(nwindows):
            win_y_low = raw.shape[0] - (window+1)*window_height
            win_y_high = raw.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # update the line class
        line.leftrecentfit(left_fit)
        line.leftbestfit()
        line.leftcurrentfit(left_fit)
        line.rightrecentfit(right_fit)
        line.rightbestfit()
        line.rightcurrentfit(right_fit)
        line.getdiffs()
    # utilize Look-Ahead Filter
    else:
        left_fitx, right_fitx, ploty , left_fit , right_fit, leftx, lefty, rightx, righty = search_around_poly(raw, line.left_current_fit , line.right_current_fit)      
        # update the line class
        line.leftrecentfit(left_fit)
        line.leftbestfit()
        line.leftcurrentfit(left_fit)
        line.rightrecentfit(right_fit)
        line.rightbestfit()
        line.rightcurrentfit(right_fit)
        line.getdiffs()
    return undis_img, left_fit, right_fit, leftx, lefty, rightx, righty


# find the actual center coordinate of the image in the warpped image

def findwindowmidpoint():
    blank = np.zeros((720,1280))
    cv2.line(blank, (640, 0), (640, 720), 255, 5)   # draw a line in the center of image
    blank_warped = cv2.warpPerspective(blank, M, (1280,720), flags=cv2.INTER_LINEAR)   # to see the transformed line in the warpped image
    midnonzero = np.nonzero(blank_warped[719,:])  # the actual bottom center point in the warpped image
    midpointx = np.mean(midnonzero)
    return midpointx


windowmid = findwindowmidpoint()

# Now we can find the offset from the center of the image 


# define a funtion to draw the necessary text and Graphics
def draw(undis_img, left_fit, right_fit, leftx, lefty, rightx, righty):
    global line
    # Generate x and y values for plotting
    ploty = np.linspace(0, undis_img.shape[0]-1, undis_img.shape[0] )
    
    '''
    the two lane lines we identified, but this time we don't draw them.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    '''
    
    # add_img is the blank background we draw the area between the lanes on, and it's already wraped,
    # after we draw the area on, we need to unwarp it.
    add_img = np.zeros((720,1280,3))    
    add_img[(xmap > (left_fit[0]*ymap**2 + left_fit[1]*ymap + left_fit[2]))&
        (xmap < (right_fit[0]*ymap**2 + right_fit[1]*ymap + right_fit[2]))]  = [180,200,0]
    
    add_img = add_img.astype('uint8')
    output = (cv2.warpPerspective(add_img, Minv, (1280,720), flags=cv2.INTER_LINEAR)).astype(np.uint8)
    
    # calculate midpoint between two lanes
    mid_index = output[680,:,1] == 200     
    # because we fill the area between two lane using a specified color
    # we can identify the x-coordinate of the pixel with this color (G-channel's value is 200)
    road_mid = np.mean(xlin[0,:][mid_index])
    offset = round((windowmid-road_mid) * xm_per_pix,2)
    line.line_base_pos = offset

    
    output = cv2.addWeighted(undis_img, 0.8, output, 1, 0) 
    
    # find the curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    average_curverad = int((left_curverad + right_curverad)/2)   # use the mean of the laft and right curvatures
    cv2.putText(output,"Radius of Curvature = " + str(average_curverad) + "(m)", (100,70), cv2.FONT_HERSHEY_SIMPLEX, 
                2,(255,255,255), 2, cv2.LINE_AA)
    
    # determine whether the vehicle is on the left side or the right side of the road center
    if offset>0:
        cv2.putText(output,"Vehicle is " + str(abs(offset)) + "(m) right of the center", (100,170), cv2.FONT_HERSHEY_SIMPLEX, 
                2,(255,255,255), 2, cv2.LINE_AA)
    elif offset<0:
        cv2.putText(output,"Vehicle is " + str(abs(offset)) + "(m) left of the center", (100,170), cv2.FONT_HERSHEY_SIMPLEX, 
                2,(255,255,255), 2, cv2.LINE_AA)
    else:
        cv2.putText(output,"Vehicle is at the center", (100,170), cv2.FONT_HERSHEY_SIMPLEX, 
                2,(255,255,255), 2, cv2.LINE_AA)
    
    return(output)

# define a function to utilize the functions above, and 
def process_image(image):
    undis_img, left_fit, right_fit, leftx, lefty, rightx, righty = getlines(image)
    
    # we use the mean value of of the last 5 polyfit parameters
    output = draw(undis_img, line.left_best_fit, line.right_best_fit, leftx, lefty, rightx, righty)
    return output

# generate project video
from moviepy.editor import VideoFileClip
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) 
white_clip.write_videofile("project_output.mp4", audio=False)
    
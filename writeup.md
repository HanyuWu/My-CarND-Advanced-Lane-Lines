**Advanced Lane Finding Project**
---

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./md/undistorted_2.JPG "Undistorted 2"
[image3]: ./md/4points.JPG "4 points"
[image4]: ./md/ptransform.JPG "perspective transform"
[image5]: ./md/binary.JPG "binary image"
[image6]: ./md/polyfit.JPG "binary image"
[image7]: ./md/formula.JPG "binary image"
[image8]: ./output_images/straight_lines1.jpg "binary image"


### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients.

First, we use the 'glob' function to save the path to all the camera calibration images, and we prepare the 'object points' which are (x,y,z) coordinates in the real world. We can assume that the object points are in (x,y) plane as z = 0. 

Secondly, we utilize 'cv2.findChessboardCorners' function to detect the corners in every calibration images, every time we successfully detect the corners, we reserve their coordinates as 'imgpoints' and append the prepared 'object points' to 'objpoints'.

With these data, we use 'cv2.calibrateCamera' function to get the camera matrix and distortion coefficients. We then use 'cv2.undistort' to unditort the camera images as Fig.1.

![alt text][image1]


### Pipeline (single images)

#### 1. Unditort the image

Use the camera matrix and distortion coefficients we get earlier to undistort the image as below.

![alt text][image2]

#### 2. Apply perspective transform

To approach proper perspective transform, we need to find 4 points in an image, and prepare 4 corresponding 'desired points' as these points' coordinates will be the exact coordintes of the 4 points' after perspective transform.

To find those 4 points, we'd better use the straight lines image. In my project, I select 2 points at the bottom of the image respectively located on the left and right lanes' centers and select other 2 points at the midlle of the imege respectively located on the left and right lanes' centers as Fig.3. We could easily assume that the 4 corresponding points after perspective transform would make up a rectangle. 

![alt text][image3]

The source and destination points I chose are as follows:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 264, 686      | 250, 720      | 
| 1055, 686     | 1000, 720     |
| 699, 457      | 1000, 0       |
| 585, 457      | 250,  0       |

With 'cv2.getPerspectiveTransform' and 'cv2.warpPerspective' function we can get the warped image we need as Fig.4.

![alt text][image4]

#### 3. Apply color transform, color and gradient threshold to generate binary image focus on the lanes

In this part, we will first convert the warpped image to HLS color space, and we extract the S channel. We will apply sobel filter on this channel, to get the gradient information we need, furthermore, we will use these gradient information to do thresholding (gradient along x,y direction, magnitude of the gradient and diretion of the gradient). On the other hand, we convert the warpped image to gray scale image to make our color threshold. 

I set my gradient threshold as follows:

```python
    gradx = abs_sobel_thresh(s, orient='x', sobel_kernel=7, thresh=(20, 255))
    grady = abs_sobel_thresh(s, orient='y', sobel_kernel=7, thresh=(0, 100))
    mag_binary = mag_thresh(s, sobel_kernel=7, mag_thresh=(20, 255))
    dir_binary = dir_threshold(s, sobel_kernel=7, thresh=(0, np.pi/3))
```

After gradient thresholding, we refuse all the pixel whose gray scale values are below 50. Meaning we will make a mask to further process the binary imgae. Finally, the binary image is as followed, and I do this part in 'project_code.py' and in its 'getlines' function.

![alt text][image5]

#### 5. Identify lane-line pixels and fit their positions with a polynomial

Before identify lane-line pixels, we need to know which area the left or right lane will be in. In common sence, we can equally divide the binary image into two left and right parts. Mostlikely, the left lane is in the left part and the right lane is in the right part. Then we use the bottom half of the binary image to generate a histogram along x-axis. The reason we only use the bottom half image is because we want to refuse as much noise as possible. The two x-positions which have the most pixels on is the 2 bases of the lanes. Then we set up n windows to move along the lane-lines from the 2 bases.

We set margin as the 1/2 * width of the window, and the height of window is the height of the binary image divied by n. The window start form base, then we calulate the mean of the non-zero pixels' x-coordinate in this window. We also save those non-zero pixels as our lane-line pixels. Then we move the window along y-axis by its height, and move it to the x-mean we calculated before. From now on, we set up a 'min' as the minimal non-zero pixels. If the non-zero pixels' amount achieve this 'min', we save the pixels as the lane-line pixels and calculate a new x-mean as the base of next window, othersise, we discard this window, and move along y-axis for next pixel dectection.

Finally, we can get the lane-line pixels of the left and right lane-line, then we utilze 'np.polyfit' function to do second-order polyfit. The belowing figure is a image which was already identified and polyfiitted.

I do this part in the 'project_code.py', and in its 'getlines' function.

 ![alt text][image6]

#### 6. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

In this part, we use find the conversions in x and y from pixels space to meters because we are using pixels as x and y coordinate but not meters, we know that U.S. regulations require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each. Comparing our image, we can define:


| meters per pixel in y dimension | meters per pixel in x dimension   | 
|:-------------------------------:|:---------------------------------:| 
|                3/104            |              3.7/750              | 

And with:

![alt text][image7]

In this case, we set y as the maximum y-coordinate value, and multiply it with 3/104 as measured in meters. So that we can calculate the radius of curvature.

For the position of the vehicle with respect to center, we should first know where the middle of the road is. In my project, I use such a method that filling up the area between two lanes with a special color, then I unwarp the warped image, find this projected area's bottom line, and calculate this line's mean x-coordinate value which is the road's center. To compute the offset from this center, we can directly subtract the middle x-coordinate of the image from this center's x-coordinate, and convert its unit form pixels to meters, that is multiplying it by 3.7/750.

The processed image is shown below:

![alt text][image8]

### Pipeline (video)

#### 1. using smooth method

Comparing to processing single image, we have to smooth the polyfit process, to make sure the curvature will not change dramatically, to do so, I set up a 'Line' class, and use it to perserve the last 5 polyfit parameters. For each new frame, I use the mean polyfit parameters to calculate the radius of curvature, and detect the area between two lane-lines.

#### 2. using Look-Ahead filter

Every time we process a new frame, we don't have to calculate a new base of the lane-lines, and sliding the window, we can use the polyfit parameter from the last n frames. Using a proper margin to detect the nonzero pixels in around the lines generated by those polyfit parameter.

Here's a [link to my video result](./project_output_video.mp4)


### Discussion

In my project, I think the calculated radius of curvature is not precise, I think I need to tweak the perspective transform parameters to get a better transfrom if I need to further improve this project.

I just simply append every polyfit paramters and in fact sometimes, we should discard them if we don't have enough confidence on this time's detection. We may use a better method to calculate the new polyfit paramters in the current frame but not just using the mean method, We can try using decay coefficient, because we should put more confidence in the new detection. Using a decay coefficient, we can decrease its influence as we detecting from new frame.

What's more,  I think my programme is pretty slow, I should imporve its speed if I want to use it in the real-time environment.
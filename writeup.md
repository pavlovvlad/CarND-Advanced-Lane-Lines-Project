## Writeup

Based on the writeup template provided by Udacity-team.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/Calib_matrix.png "Calibration matrix"
[image1]: ./output_images/Undist_image_example.JPG "Undistorted"
[image2]: ./output_images/Undist_road.JPG "Undistorted Road"
[image3]: ./output_images/Thres_s_channel.JPG "Binary after S-channel filtering"
[image4]: ./output_images/Thres_combined_s_channel_grad_magn.JPG "Binary image after combined filtering on S-channel filtering and gradient magnitude"
[image5]: ./output_images/Thres_combined_s_channel_grad_magn_and_direct.JPG "Binary image after combined filtering on S-channel filtering, gradient magnitude and gradient direction"
[image6]: ./output_images/Perspective_Transform.JPG "Result of the perspective transformation on image with straight road"
[image7]: ./output_images/Warped_image.JPG "Result of the perspective transformation on binary image with curve road"
[image8]: ./output_images/Window_search_convolution.JPG "Result of the window search with convolution"
[image9]: ./output_images/Window_search_histogram.JPG "Result of the window search with histogram"

[image10]: ./output_images/Overlay_lane.JPG "Example image of the result plotted back down onto the road"

[video1]: ./result_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The code for this step is contained in the code cell 1-3 of the IPython notebook located in "./Advanced_Lane_Lines.ipynb" .  

To calculate the camera calibration matrix and distortion coefficients with the `cv2.calibrateCamera()`-function the "object points" `objpoints` and the image points `imgpoints` are required.

The "object points" are the chessboard corners [9x6] and defined for each calibration image in (x, y, z) coordinates in 3D cartesian system as discrete numbers like an array `objp` = [[0,0,0],[1,0,0],...,[8,5,0]]. Under assumption that the chessboard is fixed on the (x, y) plane at z=0, the "object points" has been appended with a copy of standard array `objp` with 9x6 corner points every time all chessboard corners successfully detected in a calibration image.

The "image points" has been appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Number of the calibration images: 17 images.

Image shape:  (720, 1280, 3) (raw, col, depth) in pixel.

Example of the distortion corrected calibration image: 
![alt text][image1]

The resulting distortion coefficients are:
 `k1 =  -0.241017967805 
 k2 =  -0.0530720497347 
 p1 =  -0.00115810317674 
 p2 =  -0.000128318543555 
 k3 =  0.0267124302878`
 
Negative value of k1 points on the negative radial distortion (pincushion distortion).

The value of the third radial distortion term k3 is not high and verifys that the choosen camera is typical web-cam and has no high order distortions. Therefore the "fast check" (CV_CALIB_CB_FAST_CHECK switch) by the corner points extraction with `cv2.findChessboardCorners()`-function is appropriate.

The small values of the tangential distortion coefficients p1 and p2 verify that the camera’s lens is aligned parallel to the imaging plane. 

The camera calibration matrix ![alt text][image0] [c](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html):

[[  1.15396093e+03   0.00000000e+00   6.69705359e+02]
 [  0.00000000e+00   1.14802495e+03   3.85656232e+02]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]

The principal point __(cx, cy) = (669.705359, 385.656232) [pixel]__ is close to the image center (1280/2, 720/2)=(640, 360) [pixel] with the shift of (30,25) [pixel]. 

The ratio between focal lengths __fx = 1.15396093e+03 [pixel] and fy = 1.14802495e+03 [pixel]__ is close to 1:1, that means that the focus along x- and y- axis are in good proportion.

In order to estimate how exact the found parameters are the re-projection error has been calculated, as described [here](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html).

The mean re-projection error calculated using `cv2.projectPoints()`- and `cv2.norm()`- functions as the average of the absolute norms between transformation and the corner finding algorithm over all 17 calibration images:  __0.153521693302 +/- 0.0501852358604 [pixel]__ 

So, the calibration is precisely enough.

The camera calibration matrix and distortion coefficients are intrinsic parameters of the camera and are independent on the scene. Once estimated the values can be used for all frames.

### Pipeline (single images)

#### 1. Distortion correction

To see the distortion of the camera and the performance of the distortion correction based on the camera calibration matrix and distortion coefficients calculated before the test images from the real scenarios has been used:
![alt text][image2]

Image shape:  (720, 1280, 3) (raw, col, depth) in pixel

#### 2. Image pre-processing

The aim of this step is to generate the thresholded binary image (0/1 in pixels) with highlighted lines.

The HLS color scape has been used in order to filter only on the saturation-channel (S-channel) independently from the hue channel (H-channel). So the RGB to HLS convertion has been applied to the input image (the code cell 7 of the IPython notebook).

After parameter tuning:
- thresholds for S-channel: [ 170 : 255 ] (to reduce the influence of the shadow on the road as much as possible)

![alt text][image3]

By the color thresholding the lane-lines near the camera are highlighted very distinctly, but the far ends of the lines and the parts under shadow are not visible due to low level of saturation-values. So, the additional thresholding technique is required which is able to show the far parts of the lane-lines. The using of the gradient by the Canny edge detection in certain ranges can be applied in order to solve this problem. 

The optimal color channel for applying of the gradient is the lightness-channel (L-channel) since it represents the relative lightness or darkness of a color.  

After parameter tuning:
- kernel of sobel-operator for x and y directions: 9 (to smooth over noisy intensity fluctuations)
- thresholds for the gradient magnitude : [ 50 : 100 ]

The result of the binary combination (OR-operation) of these two techniques `(s_binary OR m_binary)` is:
![alt text][image4]

By thresholding with the gradient magnitude the edge direction plays no role and besides the lane lines also other edges are highlighted. As it can be seen in the figure above in the middle of the road the horizontal edges has been detected. This noise will __not__ be filtered out by applying a mask to the image. 

Since the lane lines appears in frames of the front camera mostly with the particular orientations, the thresholding along the direction of the gradient can be performed (the code cell 7 of the IPython notebook) to avoid the passing of the horizontal edges on the road. 

After parameter tuning:
- direction thresholds: [ np.pi/18 : np.pi/3 ] in [rad] or [ 10 : 72 ] in [deg]

The result of the binary combination (AND-operation) of the described techniques  `((s_binary OR m_binary) AND d_binary)` is:
![alt text][image5]

The whole lane-lines without noise from shadow and daylight within the driven lane can be seen on the thresholded binary image.

#### 3. Perspective transform

For perspective transform four characteristic landmarks "source points" on the ground plane has to be choosen. The difficulty is that the exact positions of the source points on the ground plane in the given frames are unknown. So, the frame with the straight lanes has been used in order to reduce the transformation error. The source points are placed on the lane lines that the distination points were the vertexes of the rectangle on the bird's eye view.

The code for the perspective transform to the top view ("bird's eye view") includes a function called `corners_unwarp()`, which appears in the 8th code cell of the IPython notebook.  This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.

The source and destination points has been hardcoded in the following manner:

```python
#img_size = (720, 1280)

#define 4 source points
src_points = np.float32(
[[(img_size[1] / 2) - 45,(img_size[0] / 2) + 90],
[(img_size[1] / 2) + 45,(img_size[0] / 2) + 90],
[img_size[1] - 170, img_size[0]],
[200, img_size[0]]])

#define 4 destination points
dst_points = np.float32(
[[img_size[1]/3, 0], img_size[0]/4],
[img_size[1]*2/3, 0], img_size[0]/4],
[img_size[1]*2/3, img_size[0]],
[img_size[1]/3, img_size[0]]])

```


The scale for the bird's eye view has been chosen in accordance with the appearance area of the lane lines to be detected. To cover the area of interest also by the curved lanes the destination points has been placed relatively to the image size `img_size[1]/4 = approx. 320 pixels ()` from the left and `img_size[1]*3/4 = approx. 960 pixels ()` from the right side. As also a far parts of the lines are significant by the extraction of the lane-line features, so, the quarter of the source image `img_size[0]/4 = 180 pixels ( 7 meter)` has been added to the top of the destination image.

This resulted in the following source and destination points in pixels [X, Y]:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 320, 180      | 
| 685, 450      | 960, 180      |
| 1110, 720     | 960, 720      |
| 200, 720      | 320, 720      |

To verify the perspective transform the `src` and `dst` points has been plotted on the test image with the straight road and its warped counterpart:

![alt text][image6]

Result of the perspective transformation on binary image with curve road:

![alt text][image7]

#### 4. Extraction of the lane-line features and polynomial fitting

For searching of the pixels for fitting both algorithms mentioned in the lesson has been compared: 
1. the sliding window with convolution (s. the code cell 9 of the IPython notebook); and 
2. the sliding window with histogram  (s. the code cell 10 of the IPython notebook).

After comparision on the difficult image with the shadow the 2. method has been decided as more robust:
1. ![alt text][image8]
2. ![alt text][image9]

The fitting of the lane lines has been made with a 2nd order polynomial (curvature, curvature change).

The margin for both set to +/- 100 pixels.

Also the line fitting without sliding window has been added in order to reduce the procesing time (code cell 11). 


#### 5. Calculation of the curvature of the lane and the position of the vehicle with respect to center

As the "source points" has been placed on the lane lines and the lane width and length of the segments of the dashed lines are standard, therefore the factors to convert pixels in meter on the "bird's eye view" has been chosen as follows:

- _x-factor_ = 3.7/(960 - 320) pixels:  minimum lane width on the highway is 3.7 m, and the x-coordinates of the destination points are 960 and 360 in pixels.
- _y-factor_ = 35/(720 - 180) pixels:  lengths of the dashed line segments are 5 m and 8 m, the distance between destination points in y-axis is approximately = 3 x 5m + 2.5 x 8m = 35 m and the y-coordinates of the destination points are 180 and 720 in pixels.


#### 6. Project detected lane back onto the original image

This step is implemented in the code cell 14 of the IPython notebook.  Here is an example of result on a test image:

![alt text][image10]

---

### Pipeline (video)

The pipeline is implemented in the code cells 16 and 17 of the IPython notebook.

After processing of the video without smooting and additional checks following problems have been figured out:
- jitter at the far end of the lines in some noisy frames
- in case of shadow the left road border has been wrongly recognized as the left line of lane

In order to avoid such behaviour:
- the smoothing averaging of fitted coefficients and x-values over __ 3 frames__ has been added (to avoid the jitter in geometry of the detected lane between different video frames. The class Lane() has been involved to accumulate the data over several video frames. 

- additional sanity checks of the parallelism of the right and left lines has been included into the pipeline. In case of the wring result the values from the previous succesfull frame have been used for projection of the detected line onto the original image and for the radius calculation.

Here's a [link to the video result](./result_video.mp4) after some parameter tuning.

---

### Discussion

#### 1. Problems / issues by the implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The "brute-force" solution by the perspective transformation is to hardcode the points for source and for destination. But by the roads with slope the plane of the top-view will be reconstructed with the high error, so the approximated polynom will also divergate.  To make the perspective transofrm more robust the adaptive ground planeestimation e.g. based on  [texture segmentation](https://lear.inrialpes.fr/people/cherian/papers/3dpaperICRA09.pdf) can be applied.

Sliding window as the algo for points filtering can be optimized with the adaptive window size. 

The next restriction is the approximation of the clothoid road segments with the polymonials. The S-curves, and circle road segments the approximation for the far line parts will fail. So, the b-splines or combination of the polynomial segments with the steady transition will be more robust. For the case of the highway with the high radius in curves the current implementation is the optimal choise. 

For city-scenario with the complicated combinations of the segments the polylines can be the better way to represent the lane lines. 

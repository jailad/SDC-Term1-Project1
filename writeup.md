#**Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[TestImageSolidWhiteRight_Original]: ./test_images_output/solidWhiteRight_1_original_image_copy.jpg "Solid White Right - Original "
[TestImageSolidWhiteRight_Gray]: ./test_images_output/solidWhiteRight_2_grayscale_image.jpg "Solid White Right - Gray Scale"
[TestImageSolidWhiteRight_Gaussian]: ./test_images_output/solidWhiteRight_3_gaussian_blurred_image.jpg "Solid White Right - Gaussian"
[TestImageSolidWhiteRight_Canny]: ./test_images_output/solidWhiteRight_4_canny_image.jpg "Solid White Right - Canny"
[TestImageSolidWhiteRight_ROI]: ./test_images_output/solidWhiteRight_5_region_of_interest_image.jpg "Solid White Right - Region Of Interest Selected"
[TestImageSolidWhiteRight_Hough]: ./test_images_output/solidWhiteRight_6_hough_lines_image.jpg "Solid White Right - Hough Lines with detected edges"
[TestImageSolidWhiteRight_Hough_SolidLines]: ./test_images_output/solidWhiteRight_7_hough_lines_image_solid_lines.jpg "Solid White Right - with solid line"

[TestImageSolidWhiteRight_Final]: ./test_images_output/solidWhiteRight.jpg "Solid White Right - Final image with overlay"

---

**My pipeline consisted of the following steps**:

* Reading the original image.

![][TestImageSolidWhiteRight_Original]

* Gray scaling.

![][TestImageSolidWhiteRight_Gray]

* Gaussian Smoothing.

![][TestImageSolidWhiteRight_Gaussian]

* Canny Edge Detection.

![][TestImageSolidWhiteRight_Canny]

* Region Of Interest Selection.

![][TestImageSolidWhiteRight_ROI]

* Hough Transform Detection.

![][TestImageSolidWhiteRight_Hough]

* Hough Transform Detection - with Solid lines

![][TestImageSolidWhiteRight_Hough_SolidLines]

* Overlaying the detected lane(s) on top of the original image.

![][TestImageSolidWhiteRight_Final]

**Output Video(s)** : 

* solidWhiteRight.mp4

<video width="432" height="288" controls>
  <source src="test_videos_output/solidWhiteRight.mp4" type="video/mp4">
</video>

* solidYellowLeft.mp4

<video width="432" height="288" controls>
  <source src="test_videos_output/solidYellowLeft.mp4" type="video/mp4">
</video>

* challenge.mp4

<video width="432" height="288" controls>
  <source src="test_videos_output/challenge.mp4" type="video/mp4">
</video>


**In order to draw a single line on the left and right lanes, I modified the draw_lines() function by :**
1. Categorizing the line segment(s) from hough transform within the region of interest, into left lane, or right lane, using the slope. ( > 0, or < 0).
2. From the array of left categoried line(s) picking the longest straight line, and using that as a reference left lane line to compute slope.
3. Determining the point of intersection of the above reference line with region of interest's top line and bottom line and then connecting the intersection points to get a single left lane line.
4. From the array of right categoried line(s) picking the longest straight line, and using that as a reference right lane line to compute slope.
5. Determining the point of intersection of the above reference line with region of interest's top line and bottom line and then connecting the intersection points to get a single right lane line.

**Potential shortcomings:**

1. Currently towards the far end of the pipeline if there is a car within the region of interest, it shows up. I need to find a better way of excluding such objects from region of interest.
2. Occasionally the lane on the right bottom corner is not getting detected.


**Potential improvements:**

1. Dynamic region of interest generation - currently this is hard coded.
2. Ensuring code is optimized so that it runs in the most optimal manner.
3. Static analysis of code to improve it.
4. The output can be smoothened further. I am currently applying some level of smoothing but it can obviously be improved.
5. Better handling of curved line(s).


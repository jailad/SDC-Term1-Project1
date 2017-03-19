#**Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[TestImageSolidWhiteRight_Original]: ./test_images_output/solidWhiteRight_1_original_image_copy.jpg "Solid White Right"
[TestImageSolidWhiteRight_Gray]: ./test_images_output/solidWhiteRight_2_grayscale_image.jpg "Solid White Right"
[TestImageSolidWhiteRight_Gaussian]: ./test_images_output/solidWhiteRight_3_gaussian_blurred_image.jpg "Solid White Right"
[TestImageSolidWhiteRight_Canny]: ./test_images_output/solidWhiteRight_4_canny_image.jpg "Solid White Right"
[TestImageSolidWhiteRight_Hough]: ./test_images_output/solidWhiteRight_5_hough_lines_image.jpg "Solid White Right"
[TestImageSolidWhiteRight_ROI]: ./test_images_output/solidWhiteRight_6_region_of_interest_image.jpg "Solid White Right"


[TestImageSolidWhiteRight_Final]: ./test_images_output/solidWhiteRight.jpg "Solid White Right Final"

---

**My pipeline consisted of the following steps**:

* Reading the original image.

![Detected lane(s) in red overlaid over the original image][TestImageSolidWhiteRight_Original]

* Gray scaling.

![Detected lane(s) in red overlaid over the original image][TestImageSolidWhiteRight_Gray]


* Gaussian Smoothing.

![Detected lane(s) in red overlaid over the original image][TestImageSolidWhiteRight_Gaussian]


* Canny Edge Detection.

![Detected lane(s) in red overlaid over the original image][TestImageSolidWhiteRight_Canny]


* Hough Transform Detection.

![Detected lane(s) in red overlaid over the original image][TestImageSolidWhiteRight_Hough]


* Region Of Interest Selection.

![Detected lane(s) in red overlaid over the original image][TestImageSolidWhiteRight_ROI]


* Overlaying the detected lane(s) on top of the original image.

![Detected lane(s) in red overlaid over the original image][TestImageSolidWhiteRight_Final]


**TBD** - In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

**Potential shortcomings:**

1. Currently towards the far end of the pipeline if there is a car within the region of interest, it shows up. I need to find a better way of excluding such objects from region of interest.
2. Occasionally the lane on the right bottom corner is not getting detected.


**Potential improvements:**

1. Dynamic region of interest generation - currently this is hard coded.
2. Ensuring code is optimized so that it runs quickly.

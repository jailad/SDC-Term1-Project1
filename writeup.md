#**Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[TestImageSolidWhiteRight]: ./test_images/solidWhiteRight.jpg "Solid White Right"
[TestImageOutputSolidWhiteRight]: ./test_images_output/solidWhiteRight.jpg "Solid White Right"

---

My pipeline consisted of the following steps:

1. Gray scaling.
2. Gaussian Smoothing.
3. Canny Edge Detection.
4. Hough Transform Detection.
5. Region Of Interest Selection.
6. Overlaying the detected lane(s) on top of the original image.

As an example, this is the original image

![solidWhiteRight.jpg - Original Image][TestImageSolidWhiteRight =432]
![Detected lane(s) in red overlaid over the original image][TestImageOutputSolidWhiteRight]

TBD - In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

Potential shortcomings:

1. Currently towards the far end of the pipeline if there is a car within the region of interest, it shows up. I need to find a better way of excluding such objects from region of interest.
2. Occasionally the lane on the right bottom corner is not getting detected.


###3. Suggest possible improvements to your pipeline

Potential improvements:

1. Dynamic region of interest generation - currently this is hard coded.
2. Ensuring code is optimized so that it runs quickly.

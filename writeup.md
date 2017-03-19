#**Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[TestImageSolidWhiteRight]: ./test_images/solidWhiteRight.jpg "Solid White Right"
[TestImageOutputSolidWhiteRight]: ./test_images_output/solidWhiteRight.jpg "Solid White Right"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of the following steps:

1. Gray scaling.
2. Gaussian Smoothing.
3. Canny Edge Detection.
4. Hough Transform Detection.
5. Region Of Interest Selection.
6. Overlaying the detected lane(s) on top of the original image.

As an example, this is the original image

![solidWhiteRight.jpg - Original Image][TestImageSolidWhiteRight]
![Detected lane(s) in red overlaid over the original image][TestImageOutputSolidWhiteRight]

TBD - In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

###2. Identify potential shortcomings with your current pipeline

Currently towards the far end of the pipeline if there is a car within the region of interest, it shows up. I need to find a better way of excluding such objects from region of interest.

Currently shortcoming is that the region of interest is currently hard coded, so in order to generalize it to real-world situations, this will have to be dynamically generated.

Another shortcoming is dealing with curved roads. One potential technique to address it would be to have an adaptive region of interest polygon. 


###3. Suggest possible improvements to your pipeline

Dynamic region of interest generation.
Ensuring code is optimized so that it runs quickly.

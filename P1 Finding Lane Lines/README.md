# Find Lane Lines

The pipeline of finding lane lines is:
- Convert to gray image.
- Apply gaussian blur with kernel size of 3.
- Apply canny edge detection alogorithm with threshold.
- Clip region of interest: keep the road part.
- Draw lines with hough line transform algorithm 
- Recomine the image and return the final result.

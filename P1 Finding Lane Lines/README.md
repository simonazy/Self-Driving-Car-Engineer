# Find Lane Lines

The pipeline of finding lane lines is:
- Convert to gray image.
- Apply gaussian blur with kernel size of 3.
- Apply canny edge detection alogorithm with threshold.
- Clip region of interest: keep the road part.
- Draw lines with hough line transform algorithm 
- Recomine the image and return the final result.

![image](https://user-images.githubusercontent.com/56880104/127221260-0bbe6d53-269a-4ff7-a69b-5a3382732e30.png)

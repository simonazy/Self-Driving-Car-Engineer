# Apply Sobel 

```python 

def abs_sobel_thresh(img, orient='x',thresh_min=0, thresh_max=255):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
  if orient == 'x':
     abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
  if orient == 'y':
     abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
  # Rescale back to 8 bit integer
  scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
  # Create a copy and apply the threshold
  binary_output = np.zeros_like(scaled_sobel)
  # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
  binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
  
  return binary_output
```

# Magnitude of gradient

```python
# Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values

def mag_thresh(img, sobel_kernel=3, mag_thresh=(100, 155)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    return binary_output
```

# direction of gradient

```python
# Define a function to threshold an image for a given range and Sobel kernel

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output
```

# Combine threshold
```python
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

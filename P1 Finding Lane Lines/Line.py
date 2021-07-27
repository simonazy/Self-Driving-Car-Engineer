import numpy as np 
import cv2 

class Line:
    """
    A Line is defined from two points (x1, y1) and (x2, y2) as follows:
    y - y1 = (y2 - y1) / (x2 - x1) * (x - x1)
    Each line has its own slope and intercept (bias).
    """
    def __init__(self,x1,y1,x2,y2):
        self.x1 = np.float(x1)
        self.y1 = np.float(y1)
        self.x2 = np.float(x2)
        self.y2 = np.float(y2)
        
        self.slope = self.compute_slope()
        self.bias = self.compute_bias()
        
    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1
    
    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    def set_coord(self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def draw(self, img, color=[255,0,0],thickness=10):
        cv2.line(img,(self.x1,self.y1),(self.x2, self.y2),color,thickness)
        
        
    
    
    
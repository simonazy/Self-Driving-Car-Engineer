import PIL
import cv2
from torchvision import transform 

"""The Contrast-limited adaptive histogram equalization (CLAHE for short) algorithm partitions the images into contextual regions and applies the histogram equalization to each one. This evens out the distribution of used grey values and thus makes hidden features of the image more visible. """

class CLAHE_GRAY:
    def __init__(self, clipLimit=2.5, tileGridSize=(8,8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
    def __call__(self,img):
        img_y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:,:,0] #YCrCb Color-Space
        # for Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_y = clahe.apply(img_y)
        img_output = img_y.reshape(img_y.shape + (1,))
       
        return img_output


"""
The 43 classes are not equally represented. The relative frequency of some classes is significantly lower than the mean. So, I built a jittered dataset by geometrically transforming(rotation, translation, shear mapping, scaling) the same sign picture. 
"""
def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transform.RandomRotation(20, resample=PIL.Image.BICUBIC),
                                transforms.RandomAffine(0,translate=(0.2, 0.2), resample=PIL.Image.BICUBIC),
                                transforms.RandomAffine(0, shear=20, resample=PIL.Image.BICUBIC),
                                transforms.RandomAffine(0, scale=(0.8, 1.2), resample=PIL.Image.BICUBIC)]),
                                transforms.ToTensor()
    ])

def get_test_transforms():
    return transforms.Compose([transforms.ToTensor()])

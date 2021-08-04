# Traffic Sign Classifier Pytorch Version

In this repo, I implement the traffic sign classifier with PyTorch. I trained the model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I tested the model on test dataset with accuracy rate of.

## Dataset

The GTSRB dataset (German Traffic Sign Recognition Benchmark) is provided by the Institut für Neuroinformatik group [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). It was published for a competition held in 2011 ([results](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results)). Images are spread across 43 different types of traffic signs and contain a total of 39,209 train examples and 12,630 test ones.

<p align="center"><img src="./images/traffic-signs.png" /></p>

1. [Download the dataset](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which resized the images to 32x32.

2. Unzip the dataset into `./data` directory.


The pickled dataset summary:
- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32)
- Number of classes = 43



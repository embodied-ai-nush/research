# EmbodiedAI
The Public Repository for the project "Embodied AI for computational perception and understanding of spatial designs"

> By: **Zayan Karimi** and **Prannaya Gupta**, with assistance from **Prof Immanuel Koh** (SUTD) and **Mr Ng Chee Loong** (NUS High).



## Abstract

Semantic Segmentation is a computer vision task used to identify specific regions of interest for virtual agents and autonomous robots or vehicles, specifically by assigning a class to every pixel of a given image. These class allocations require computational perception of nearby pixels, which hence requires very highly detailed algorithms and larger segmentation models to be achieved. The purpose of this project is to gain semantic understanding of the architectural design and layout of Housing Development Board (HDB) apartment interiors and of 3D exterior facades of HDB buildings. Transfer learning algorithms on semantic segmentation models have been employed to understand and analyse these two-dimensional images provided. These models require a lot of carefully annotated data, hence transfer learning on pre-trained models is used to improve the accuracy of the model. Ultimately, it is found that transfer learning with semantic segmentation is an effective way to comprehend the design and layouts of these HDB interiors and exteriors.



## Introduction

Understanding the layout and design of apartments and buildings is an important task for interior designers, architects, construction workers and various other professions. Hence, we found it valuable to use machine learning and computer vision for faster and better understanding of these layouts.

For our project, we used semantic segmentation which is a computer vision task for segmenting an image into various classes. It differs from standard object detection and classification as the model tries to find exact boundaries for the given objects in the image and every pixel in a given image belongs to a class.

However, the problem with semantic segmentation is that it requires a lot of data and a lot of computational resources to perform well. Hence, we used transfer learning to optimize the task. Transfer learning is a machine learning technique where knowledge gained in one task is used for a similar second task. In our case, we used a pretrained semantic segmentation model and only trained the weights in the last few layers of the model reducing gradient computation time and decreasing the size of the dataset required for a quality model.

We split the project into two tasks. In the interior task, we perform semantic segmentation on HDB apartment interiors. Since, there are no available datasets with semantic annotations of HDB interiors, we used a portion of the ADE20K dataset which contains interiors of homes. In the exterior task, we perform semantic segmentation on HDB exterior facades. We use a dataset provided by SUTD which contains images of HDB buildings and their annotations.



## Methodology

### Data Collection

#### ADE20K Dataset

The ADE20K dataset~\cite{ade20k} is made up of over 27000 images and their annotations from the SUN and Places databases. We used this dataset for the interior task of our project, Hence, we only used images that contained images of interiors of houses and hotels for training.

#### HDB Facade Dataset

The HDB Facade Dataset has been provided by the Artificial-Architecture Research Laboratory at the Singapore University of Technology and Design (SUTD). The Dataset uses

### Data Preprocessing and Cleaning

#### Color Classification

The HDB Facade Dataset is taken via Screenshots of Specific Facades, which are susceptible to altered shades instead of definite class-based colours of grey, red and green. Hence, a requirement of the Exterior HDB Semantic Segmentation Task is to modify this data to fit to the model required by the pretrained model. 

#### Colour Clustering

The initial approach is to cluster the colours in L*AB space so as to identify specific regions

#### Image Transformation

The images in ADE20K have a very high resolution and there are over 3000 object classes. Since a large number of these classes were unnecessary for the interior task, we lowered the number of classes to 150 and converted the annotated images such that if a given pixel in an image is labelled as an object which has a class number of 56, the colour of that pixel in the transformed annotated image will be (56, 56, 56).

This method is adapted from the MIT Scene Parsing Benchmark

### Training

Our model was written in pytorch. We used Stochastic Gradient Descent with Resnet101~\cite{resnet101_reference} as our encoder net and Upernet~\cite{upernet_reference} as our decoder net with Negative Log Likelihood loss.\ref{Negative Log Likelihood Loss} We used transfer learning to optimize our task. In our case we used MIT's semantic segmentation model as our base model and retrained the last layer with a different number of classes. This model is trained on ADE20K scene parsing dataset which is a slightly different dataset than ADE20K with fewer images and classes.

#### Interior Task

We trained the last convolutional layer of the model with 150 classes for 10 epochs for 120 iterations.

#### Exterior Task

We trained the last convolutional layer of the model with 3 classes for 10 epochs for 120 iterations.



## Results and Discussion

### Validation

Measuring accuracy is difficult for semantic segmentation tasks as pixel accuracy can vary a large amount. To test the model we used two metrics, Intersection over union (IOU) [\ref{Intersection Over Union (IOU)}] and Pixel accuracy [\ref{Pixel accuracy}].

#### Interior Task

The interior task model had a mean pixel accuracy of 78.34\% and a mean IOU score of 46.35\%. As we can see from Figure \ref{fig:sample_image_interior_task}, The model correctly identifies the boundaries between the different objects in the frame. However, there is a high amount of noise in the predicted segmentation's are slightly incorrect. We can also see that the model has a difficult time with partial objects as it fails to correctly identify the tables underneath the lamps and the table in the front of the image.

#### Exterior Task

The exterior task model has a mean pixel accuracy of 95.20\% and a mean IOU score of 59.79\%. As we can see from Figure \ref{fig:sample_image_exterior task}, the Model correctly identifies the building data. However, it has a difficult time discerning between windows and void data. The high accuracy as compared to the interior task is likely due to the lower class count of 4.



## Conclusion

In conclusion, we were able to successfully make 2 models for the interior and exterior tasks. We also gained a better understanding of semantic segmentation and transfer learning. However, more work can be done to improve the accuracy of the both tasks. The model in the exterior task has difficulty distinguishing between void and windows and the model in the interior task has a more poor segmentation.



## Future Work
### Material type detection

### other future

## Acknowledgements
We are sincerely thankful to Mr Ng Chee Leong of NUS High school and Dr Immanuel Koh of SUTD. We would also like to thank SUTD for providing us with resources.



## Appendix

### Negative Log Likelihood Loss

> L(y)=-log(y)

Where,

- y is the output of the network
- L(y) is the Loss with respect to y

### Intersection Over Union (IOU)

Intersection over union is a measure of the overlap between the predicted segmentation and ground truth. The IOU values for every class were averaged to get the final IOU score.

> IOU=A/B

Where,

- A is the area of the intersection between the predicted segmentation and ground truth
- B is the area of the union between the predicted segmentation and ground truth

### Pixel Accuracy

Pixel accuracy is a measure of whether the pixels of the predicted segmentation matched ground truth and whether the pixels that were not part of the predicted segmentation matched ground truth.

> PA=(TP + TN)/(TP + TN + FP + FN)

Where,

- TP are true positives
- TN are true negatives
- FP are false positives
- FN are false negatives




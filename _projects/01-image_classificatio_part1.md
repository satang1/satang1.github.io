---
title: "Image Analysis Classification"
layout: single
classes: wide
author_profile: true


header:
  teaser: /assets/img/image_class/teddybear_logo.jpg
---



**Computer Vision Image Classification EDA**


This classification project is meant to be an individual analysis project (despite the separation between training and test sets). The datasets were downloaded from DS100's graduate project. The primary portion of the following notebook is to explain my exploration process for the training set and consider the potential features that could separate the images into their respective categories. 

Disclaimer, this post is rather long, and comprises a lot of code and my thought processes while looking through the datasets.

# Data Input 


```python
#Import anything you need here
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import skimage
from skimage import data
from skimage import io
import os

#imported below for creating dom_color function
from sklearn.cluster import KMeans
from collections import Counter
```

<h4> Dataframe encoding as listed below. </h4>	
0=Airplanes, 1=Bear, 2=Blimp, 3=Comet, 4=Crab, 5=Dog, 6=Dolphin, 7=Giraffe, 8=Goat, 9=Gorilla, 10=Kangaroo, 11=Killer-Whale, 12=Leopards, 13=Llama, 14= Penguin, 15= Porcupine, 16=Teddy-Bear, 17=Triceratops, 18=Unicorn, 19=Zebra


```python
from os import listdir
from os.path import isfile, join
import cv2

def read_organize_data(file_path):
    ''' Input: file_path 
                Takes in a file path for training data ONLY, 
                cuz file/folder formatting difference as compared to test data
        
        Returns: A list of lists, where each inner list represents each individual folder, 
                and thus each individual folder (or inner list) holds the array matrices 
                for all the images/pictures within that folder
    '''
    
    pic_files = [ f for f in listdir(file_path)] # "list" out all the files/images within the directory    
    collection = []

    for i in range(1, len(pic_files)): #cuz index 0 is DS.store
        
        folder_path = join(file_path, pic_files[i]) 
        full_img_path = [join(folder_path, img) for img in listdir(folder_path)]
        
        temp = []
        for k in range(len(full_img_path)): 
            img_matrix = io.imread(full_img_path[k])
            temp.append(img_matrix)
            
        collection.append(temp)
    
    
    return collection

```


```python
starting_data = read_organize_data("20_categories_training/")
# starting_data

```


```python
def make_dataframe(all_folders):
    ''' Takes in a list of lists where each inner list represents an individual folder 
        which holds all the picture array matrices of that folder
        
        Returns a dataframe where "encoding" represents the specific folder, and 
        each row under the "pictures" column represents an indivdual image belonging to the encoded folder
    '''
    li = []
    
    for k in range(len(all_folders)): 
        folder = all_folders[k]
        
        # NOTE: means that folders with no images are not included in df
        if len(folder) != 0: #folder is not empty 
            for i in range(len(folder)):
#             for i in range(3): #to test small samples
                d = {"pictures": folder[i]}
                encoding = {"encoding": k} 
                
                d.update(encoding)
                li.append(d)
                
    df = pd.DataFrame(li)
    return df
        

# # training_df = make_dataframe(starting_data[:3]) #to test small samples
# training_df = make_dataframe(starting_data)
# training_df.head()
```


```python
training_df = make_dataframe(starting_data)
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[[[75, 93, 79], [34, 52, 38], [27, 48, 33], [2...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[[[231, 205, 188], [228, 180, 144], [217, 179,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>[[[255, 252, 249], [255, 252, 249], [255, 253,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>[[[173, 166, 160], [175, 168, 162], [178, 171,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
  </tbody>
</table>
</div>



From here, I will split the training set into a training and validation set so that I can do some validation testing later. I will reserve 10% of the training set data for the validation test set.


```python
from sklearn.model_selection import train_test_split

#42 to generate the same pseudo-rand sequence everytime it runs
train, val = train_test_split(training_df, test_size=0.1, random_state=42) 
```


```python
len(train), len(val), len(training_df)
```




    (1278, 143, 1421)




```python
training_df = train
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
    </tr>
    <tr>
      <th>358</th>
      <td>6</td>
      <td>[[[150, 202, 226], [150, 202, 226], [150, 202,...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>[[[6, 5, 3], [6, 5, 3], [6, 5, 3], [6, 5, 3], ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
skimage.io.imshow(training_df.loc[400].values[1])
plt.show()

```


![png](/assets/img/image_class/output_13_0.png)
<!-- <img src="/assets/img/image_class/output_13_0.png"> -->

At this point, we can read in the testing data, and below I've defined a different function to read in the test data since the organization of the testing and training set data is slightly different in the arrangement of folders. At the moment, it was more conveinent to write a simple function, but in the future, I'll probabily generalize the above "read_organize_data" function to also include reading in the test set data regardless of the folder organization.  


```python
def read_test_data(test_file_path): 
    

    ''' Takes in the file path of test set ONLY,
    
        Returns a list of image array matrices for each image
        in the directory file
    '''
    onlyfiles = [f for f in listdir(test_file_path) if isfile(join(test_file_path, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(test_file_path, onlyfiles[n]))
        
    return images
```


```python
test_starting_data = read_test_data("20_Validation/")
# test_starting_data

```

Notice that we drop the "encoding" column since we have yet to determine what each image is categorized as. The dataframe was constructed with an encoding column of 0 because of how we defined our "make_dataframe" function, which was mainly constructed to read in the training set data. But in this case, we can simply use the same dataframe creation function and ignore the "encoding" column for convienence.  


```python
test_df = make_dataframe([test_starting_data]) #had to encase test_starting data so that it'll be a list of list 
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[[[74, 79, 80], [50, 55, 56], [48, 53, 54], [4...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[[[85, 173, 213], [85, 173, 213], [85, 173, 21...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[[[30, 36, 19], [25, 31, 14], [21, 26, 11], [2...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>[[[69, 138, 95], [69, 138, 95], [69, 138, 95],...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>[[[242, 204, 162], [242, 203, 164], [242, 203,...</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.drop(columns="encoding", axis=1, inplace=True) #remember to set inplace to True to make this permanent 
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pictures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[[74, 79, 80], [50, 55, 56], [48, 53, 54], [4...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[[85, 173, 213], [85, 173, 213], [85, 173, 21...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[[[30, 36, 19], [25, 31, 14], [21, 26, 11], [2...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[[[69, 138, 95], [69, 138, 95], [69, 138, 95],...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[[[242, 204, 162], [242, 203, 164], [242, 203,...</td>
    </tr>
  </tbody>
</table>
</div>



Let's look at the first image of our test set, it looks to be a penguin! (but sideways--which makes me wonder how this will affect my analysis later...will I need to account for the orientation of the image? Perhaps not, if I'm only considering the pixel colors. Though it may be interesting to see if I could somehow incorporate orientation...maybe more penguin photos are displayed/visualized hotdog style instead of hamburger style)


```python
skimage.io.imshow(test_df.iloc[0].values[0]);
```


![png](/assets/img/image_class/output_21_0.png)


# Exploratory Analysis

Now that we've successfully read in our training and testing set data, and visualized a few of the pictures of both datasets. We're ready to start our exploratory analysis! First, let's gather some graphical summaries of the training set data (e.g. sizes, pixel intensities, and class frequencies).


```python
info = training_df.groupby("encoding").size().to_frame("counts")
info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>counts</th>
    </tr>
    <tr>
      <th>encoding</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
    </tr>
    <tr>
      <th>5</th>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>




```python
info["prop_of_training_set"] = info["counts"] / sum(info["counts"])
info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>counts</th>
      <th>prop_of_training_set</th>
    </tr>
    <tr>
      <th>encoding</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>59</td>
      <td>0.046166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51</td>
      <td>0.039906</td>
    </tr>
    <tr>
      <th>3</th>
      <td>76</td>
      <td>0.059468</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>0.036776</td>
    </tr>
    <tr>
      <th>5</th>
      <td>63</td>
      <td>0.049296</td>
    </tr>
  </tbody>
</table>
</div>




```python
sum(info["counts"])
```




    1278



We've looked briefly at the proportion/frequency of each animal group (which I will now refer to as "class" from now on). So we will now look at some of the pixel summaries for each image. If you take a look at this [link](http://www.whydomath.org/node/wavlets/imagebasics.html) here, it'll give you a general idea of what a pixel is and how bit values relate. But in a nutshell, a pixel (or picture element) is basically a block of an image represented by numbers ranging from 0 (black) to 255 (white). These numbers determine the grey intensity of the image. But for pictures/images in color, the pixels are represented by 3 values (r, g, b) representing red, blue, and green. And each red, blue, or green value has the same range of values (0: none of the respective color, and 255: all or highest intensity of that color). Mixing and matching these three values allows for all colors in an image. And the more pixels there are in an image, the higher the picture resolution.  

For now, let's take a look at two of the images, namely the 302th index and the 1369th index image. 


```python
img_302 = training_df.loc[302].values[1]
skimage.io.imshow(img_302);
```


![png](/assets/img/image_class/output_28_0.png)



```python
img_302.shape 
```




    (737, 568, 3)



The image shape is represented by a 3d array, where the first and second dimension represent the row and column pixel numbers, and the third value represents the color value. Remember that colors are represented by 3 values (r, b, g) and thus, the third dimension of this image will always be 3 (with each of the three values ranging from 0 to 255).

Now, we're going to build an image histogram. And the rest of this exploratory analysis will be based on Adrian Rosebrock's tutorials on pyimagesearch. He has a lot of opencv and computer vision tutorials that are great to follow if anyone is interested. I highly recommend them! 

References for the following code and analysis can be found in the [openCV documentation](https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html) on histograms and on Rosebrock's [tutorials](https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/). 


```python
#let's convert our image into grayscale and look at that first: 
grayscale = cv2.cvtColor(img_302, cv2.COLOR_BGR2GRAY)

fig, axes = plt.subplots(1,2)
axes[0].imshow(grayscale, "gray");
axes[0].set_title("grayscale image of 302")

hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
axes[1].plot(hist)
axes[1].set_xlabel("pixel value bins")
axes[1].set_ylabel("pixel count")
axes[1].set_title("grayscale image histogram for img_302")


plt.tight_layout()
```


![png](/assets/img/image_class/output_31_0.png)


From both the grayscale picture and the histogram, you can obviously tell that there is a lot of white pixels, hence the large peak on the right side of the histogram.

Notice too, what the parameters of the cv2.calcHist() function takes in: 

1. the image as a list


2. the channel (as a list) that we want to compute, which is essentially a list of indices where grayscale is represented by 0, if we want the RGB channels, our indices will be (0, 1, 2) as a list


3. the mask image (which I will go deeper into later), but Rosebrock's post also has a great explanation


4. histogram size/ number of bins to use (as a list) 


5. range of possible pixel values (as a list) 

Now, we will take a look at each individual color scheme, red, blue, and green, and the corresponding image histogram distribution. 


```python
color = ("b", "g", "r")

for channel, color, in enumerate(color): 
    hist_channel = cv2.calcHist([img_302], [channel], None, [256], [0, 256])
    plt.plot(hist_channel)

plt.xlabel("bins")
plt.ylabel("number of pixels")
plt.title("Color histogram for image 302")
```




    Text(0.5, 1.0, 'Color histogram for image 302')




![png](/assets/img/image_class/output_33_1.png)


References: [read this](https://www.cambridgeincolour.com/tutorials/histograms1.htm)


Now this histogram is not very interesting, because remember, our original image was mainly white, except for the small splash of color in the upper right hand corner of the picture. And this small splash of color doesn't show up very well in the histogram because in the context of the entire image, it doesn't contribute as much to the image as the whiter pixels do. However, we can use "masking" to section out a portion of the image and create a small image histogram of the upper right hand corner. 

Masking is basically when you mask everything you don't want, so that you can section out the part of the image that you do want. The masking image has the same dimensions and pixels as the original image, except the pixels in the mask image which are 0, are ignored, and everything above 0 is used to compute the histogram.


```python
mask = np.zeros(img_302.shape[:2], np.uint8) #aka a copy of the 302 image without the 3rd (color) dimension

plt.title("mask image")
mask, skimage.io.imshow(mask) 
#note: mask is all 0's so this is basically a copy of the 302 image pixel dimension but all BLACK (see below)
```




    (array([[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
     <matplotlib.image.AxesImage at 0x1c74126e10>)




![png](/assets/img/image_class/output_35_1.png)


We want to isolate the upper right hand corner of the image, so this means that we need to convert the upper right hand side to be a value greater than 0 (i.e. 255)). You may have to play around with the dimensions to get the right isolated section, visualizing it helps!


```python
#     width ; height
mask[50: 400, 360:600] = 255
skimage.io.imshow(mask)
```




    <matplotlib.image.AxesImage at 0x1c73e95748>




![png](/assets/img/image_class/output_37_1.png)



```python
#bit_wise_and is basically masking the bits (see bit twiddling)
mask_image = cv2.bitwise_and(img_302, img_302, mask=mask) #getting the image that we want 

plt.subplot(121), skimage.io.imshow(mask_image)

#plotting the masked histogram (all colors, so for loop)
color = ("b", "g", "r")

for channel, color, in enumerate(color): 
    masked_hist = cv2.calcHist([img_302], [channel], mask, [256], [0,256])
    plt.subplot(122), plt.plot(masked_hist)

plt.subplot(122), plt.title("Masked image histogram")
plt.subplot(122), plt.xlabel("bins")
plt.subplot(122), plt.ylabel("number of pixels")


plt.tight_layout()
```

    /Users/samanthatang/anaconda3/lib/python3.6/site-packages/matplotlib/figure.py:98: MatplotlibDeprecationWarning: 
    Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      "Adding an axes using the same arguments as a previous axes "



![png](/assets/img/image_class/output_38_1.png)


So it looks like there's a bit more fluctuation of color now that we've masked the colored section out. 

But another way that we can take a look at the colored section is to simply crop out the section we want instead of masking it. Then, we can construct an image histogram on the cropped out image. You'd see that the two image histograms (masked and cropped) are both the same. (see above and below histograms)


```python
cropped_img = img_302[50:400, 360:600]
skimage.io.imshow(cropped_img)

plt.title("cropped image");
```


![png](/assets/img/image_class/output_40_0.png)



```python
color = ("b", "g", "r")

for channel, color, in enumerate(color): 
    hist_channel = cv2.calcHist([cropped_img], [channel], None, [256], [0, 256])
    plt.plot(hist_channel)
    
plt.title("Cropped image histogram")
plt.ylabel("number of pixels")
plt.xlabel("bins");
```


![png](/assets/img/image_class/output_41_0.png)


At this point, we've focused on creating one channel image histograms, but we can also create multi-dimensional channel histograms as well. And this may be helpful in finding correlation/pairs of colors, which could be potentially useful as features for our animal classification. That is, large pixel counts of both black and white could mean that the image is a zebra or a panda. But at this point, I won't be creating any multi-dimensional histograms. Perhaps later in the analysis I will return to this thought. (Though you can take a look at how it's done in the pyimage post on Rosebrock's blog).

Instead, let's take a look at the other image that I've selected--the 1369th index image. 


```python
img_1369 = training_df.loc[1369].values[1]
skimage.io.imshow(img_1369)
```




    <matplotlib.image.AxesImage at 0x1c73963390>




![png](/assets/img/image_class/output_43_1.png)


It's a zebra! Let's quickly perform the same analysis as we did for the 302nd image, but perhaps without all the masking and cropping of images and histograms.


```python
grayscale_1369 = cv2.cvtColor(img_1369, cv2.COLOR_BGR2GRAY)

skimage.io.imshow(grayscale_1369)
plt.show()
```


![png](/assets/img/image_class/output_45_0.png)



```python
hist2 = cv2.calcHist([grayscale_1369], [0], None, [256], [0, 256])
plt.plot(hist2)
plt.title("grayscale histogram of img_1369")
plt.xlabel("bins")
plt.ylabel("number of pixels");
```


![png](/assets/img/image_class/output_46_0.png)



From the looks of it, its seems that my image is quite evenly distributed in that majority of my tonal colors are gray-ish (and not all overly white (left skewed) or black (right skewed)). And this can easily be observed by the even distribution of gray tones in the grayscale zebra image. This is in contrast to the grayscale 302 image which was overexposed (i.e. white) and had a primarily left-skewed histogram. 


References/Posts I've consulted to understand my histogram: 

[How to read image histogram](https://digital-photography-school.com/how-to-read-and-use-histograms/)

https://www.allaboutcircuits.com/technical-articles/image-histogram-characteristics-machine-learning-image-processing/

https://photographylife.com/landscapes/what-are-histograms-a-photographers-guide


```python
plt.figure(figsize=(10, 8))
plt.subplot(121), skimage.io.imshow(img_1369)
color = ("b", "g", "r")

for channel, color, in enumerate(color): 
    hist_channel = cv2.calcHist([img_1369], [channel], None, [256], [0, 256])
    plt.subplot(122), plt.plot(hist_channel)
    
plt.title("img_1369 histogram")
plt.ylabel("number of pixels")
plt.xlabel("bins");

plt.tight_layout()
```


![png](/assets/img/image_class/output_48_0.png)



So you'll see that there's lots of yellowish, "autumny" colors, which is why there are peaks at the green, red, and blue colors. There's also a peak at the end with blue since the color contributes to the stripes of the blackish stripes of the zebra.

At this point, it's fair to say that if I want to look at anymore images, it would be nice to have some functions created so that I don't have to keep rewriting the same code to create these grayscale or colored histograms.


```python
def colored_hist(img): 
    '''Takes in an image and returns the colored rgb image histogram along with the visualized image'''
    
    plt.figure(figsize=(10, 8))
    plt.subplot(121), skimage.io.imshow(img
                                       )
    color = ("b", "g", "r")

    for channel, color, in enumerate(color): 
        hist_channel = cv2.calcHist([img], [channel], None, [256], [0, 256])
        plt.subplot(122), plt.plot(hist_channel)

    plt.title("img histogram")
    plt.ylabel("number of pixels")
    plt.xlabel("bins");

    plt.tight_layout()


def grayscale_hist(img): 
    '''Takes in an image and returns the grayscaled image histogram along with grayscaled image'''
    
    #let's convert our image into grayscale 
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(1,2, figsize=(10, 8))
    axes[0].imshow(grayscale, "gray");
    axes[0].set_title("grayscale image")

    hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    axes[1].plot(hist)
    axes[1].set_xlabel("pixel value bins")
    axes[1].set_ylabel("pixel count")
    axes[1].set_title("grayscale image histogram")


    plt.tight_layout()
```

# Feature Extraction 

Now, that we've explored/interacted with the data a little bit, we'll now be organizing/aggregating some of the following image features: 

1. image size
2. type (i.e. colored or grayscaled)
3. average of red, blue, and green channel intensities
4. dominant color (see [link](https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv))
5. aspect ratio of image (see [link](https://en.wikipedia.org/wiki/Aspect_ratio_(image))), but it's basically the proportional relationship between height and width of image (or width / height)
Note: .shape gives (h, w, channels)


```python
training_df["shape"] = [img.shape for img in training_df["pictures"]]
training_df["size"] = [img.size for img in training_df["pictures"]]
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
      <td>(655, 440, 3)</td>
      <td>864600</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
      <td>(976, 1301, 3)</td>
      <td>3809328</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
      <td>(200, 151, 3)</td>
      <td>90600</td>
    </tr>
    <tr>
      <th>358</th>
      <td>6</td>
      <td>[[[150, 202, 226], [150, 202, 226], [150, 202,...</td>
      <td>(326, 328, 3)</td>
      <td>320784</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>[[[6, 5, 3], [6, 5, 3], [6, 5, 3], [6, 5, 3], ...</td>
      <td>(337, 382, 3)</td>
      <td>386202</td>
    </tr>
  </tbody>
</table>
</div>



While trying to find the average color channel intensities, I got errors saying "IndexError: too many indices for array", and after checking my code and testing it on a smaller subset, I realized that some images given in the training set (and perhaps the test set) weren't color images. Taking this into consideration, I extracted the images that had no color channel (i.e. the length of the shape tuple in the "shape" column was less than 3). As such, I will be adding a column indicating whether the image is colored or not.  


```python
boolean = [3 == length for length in [len(tup) for tup in training_df["shape"].values]]

training_df[[not b for b in boolean]] #not colored
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135</th>
      <td>3</td>
      <td>[[15, 10, 10, 13, 13, 9, 10, 15, 13, 12, 11, 1...</td>
      <td>(480, 640)</td>
      <td>307200</td>
    </tr>
    <tr>
      <th>355</th>
      <td>6</td>
      <td>[[128, 146, 164, 168, 165, 160, 154, 147, 148,...</td>
      <td>(843, 1125)</td>
      <td>948375</td>
    </tr>
    <tr>
      <th>660</th>
      <td>9</td>
      <td>[[73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, ...</td>
      <td>(400, 334)</td>
      <td>133600</td>
    </tr>
    <tr>
      <th>182</th>
      <td>3</td>
      <td>[[47, 44, 37, 34, 46, 59, 51, 33, 35, 31, 29, ...</td>
      <td>(370, 503)</td>
      <td>186110</td>
    </tr>
    <tr>
      <th>137</th>
      <td>3</td>
      <td>[[7, 7, 7, 8, 8, 9, 9, 9, 10, 8, 6, 4, 5, 9, 1...</td>
      <td>(480, 640)</td>
      <td>307200</td>
    </tr>
    <tr>
      <th>165</th>
      <td>3</td>
      <td>[[25, 23, 21, 19, 19, 20, 22, 23, 13, 20, 27, ...</td>
      <td>(387, 472)</td>
      <td>182664</td>
    </tr>
    <tr>
      <th>145</th>
      <td>3</td>
      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...</td>
      <td>(339, 440)</td>
      <td>149160</td>
    </tr>
    <tr>
      <th>250</th>
      <td>4</td>
      <td>[[255, 255, 255, 255, 255, 255, 255, 255, 255,...</td>
      <td>(189, 300)</td>
      <td>56700</td>
    </tr>
    <tr>
      <th>177</th>
      <td>3</td>
      <td>[[14, 0, 11, 0, 0, 0, 0, 1, 0, 5, 7, 2, 0, 0, ...</td>
      <td>(527, 711)</td>
      <td>374697</td>
    </tr>
    <tr>
      <th>181</th>
      <td>3</td>
      <td>[[173, 175, 178, 166, 144, 137, 136, 127, 120,...</td>
      <td>(374, 500)</td>
      <td>187000</td>
    </tr>
    <tr>
      <th>173</th>
      <td>3</td>
      <td>[[27, 31, 31, 28, 31, 39, 44, 42, 43, 42, 44, ...</td>
      <td>(351, 403)</td>
      <td>141453</td>
    </tr>
    <tr>
      <th>162</th>
      <td>3</td>
      <td>[[0, 0, 0, 1, 3, 2, 0, 0, 14, 0, 7, 4, 0, 23, ...</td>
      <td>(333, 500)</td>
      <td>166500</td>
    </tr>
    <tr>
      <th>176</th>
      <td>3</td>
      <td>[[36, 37, 38, 39, 41, 42, 43, 43, 38, 39, 39, ...</td>
      <td>(340, 472)</td>
      <td>160480</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2</td>
      <td>[[191, 191, 191, 192, 192, 193, 193, 193, 191,...</td>
      <td>(477, 595)</td>
      <td>283815</td>
    </tr>
    <tr>
      <th>160</th>
      <td>3</td>
      <td>[[12, 12, 12, 12, 12, 12, 12, 12, 17, 15, 14, ...</td>
      <td>(193, 250)</td>
      <td>48250</td>
    </tr>
    <tr>
      <th>130</th>
      <td>3</td>
      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...</td>
      <td>(262, 432)</td>
      <td>113184</td>
    </tr>
  </tbody>
</table>
</div>




```python
gray_image_indices = training_df[[not b for b in boolean]].index

training_df["type"] = ["color"] * len(training_df) #basically set "type" column to color by default and change values later
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
      <td>(655, 440, 3)</td>
      <td>864600</td>
      <td>color</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
      <td>(976, 1301, 3)</td>
      <td>3809328</td>
      <td>color</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
      <td>(200, 151, 3)</td>
      <td>90600</td>
      <td>color</td>
    </tr>
    <tr>
      <th>358</th>
      <td>6</td>
      <td>[[[150, 202, 226], [150, 202, 226], [150, 202,...</td>
      <td>(326, 328, 3)</td>
      <td>320784</td>
      <td>color</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>[[[6, 5, 3], [6, 5, 3], [6, 5, 3], [6, 5, 3], ...</td>
      <td>(337, 382, 3)</td>
      <td>386202</td>
      <td>color</td>
    </tr>
  </tbody>
</table>
</div>




```python
gray_image_indices
```




    Int64Index([135, 355, 660, 182, 137, 165, 145, 250, 177, 181, 173, 162, 176,
                89, 160, 130],
               dtype='int64')




```python
# https://stackoverflow.com/questions/12307099/modifying-a-subset-of-rows-in-a-pandas-dataframe
#for how to change/replace/modify selected rows 
training_df.loc[training_df.index.isin(gray_image_indices), "type"] = "gray"
```

You can see/select the images, and confirm that these images are in fact grayscale and not colored. 


```python
img_89 = training_df.loc[660].values[1]
skimage.io.imshow(img_89)
```




    <matplotlib.image.AxesImage at 0x1c741e4470>




![png](/assets/img/image_class/output_61_1.png)


Now, we can find the average color channel intensities for each color per image. 

Note: We know that the color channel order is BGR because by default, openCV reads in the channels in this order.


```python
def color_channel_avg(df, color): 
    
    """Takes in a dataframe DF that has a 'type' and 'pictures' column 
    and returns the average color intensity for the COLOR number specified.
    
    Takes: 
    DF: dataframe with 'type' and 'pictures' column
    COLOR: channel number where 0 = blue, 1 = green, and 2 = red
    (bgr order because of openCV)
    
    Returns: 
    A list of averages for that COLOR channel
    
    """
    
    average = []
    
    for index in np.arange(len(df)): 
        if df.iloc[index]["type"] == "color": 
            avg = np.average(df.iloc[index]["pictures"][:, :, color])
            average.append(avg)
        else: 
            average.append(0)
            
    return average
```


```python
# color_channel_avg(training_df, 0)
```


```python
plt.imshow(training_df.loc[1369]["pictures"][:, :, 0]); #only blue channel
```


![png](/assets/img/image_class/output_65_0.png)



```python
print("blue avg: ", np.mean(training_df.loc[1369]["pictures"][:, :, 0]))
print("green avg: ", np.mean(training_df.loc[1369]["pictures"][:, :, 1]))
print("red avg: ", np.mean(training_df.loc[1369]["pictures"][:, :, 2]))
```

    blue avg:  151.8945083618164
    green avg:  136.87965393066406
    red avg:  92.40813954671223



```python
test = training_df.iloc[:3] #simple sanity check
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
      <td>(655, 440, 3)</td>
      <td>864600</td>
      <td>color</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
      <td>(976, 1301, 3)</td>
      <td>3809328</td>
      <td>color</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
      <td>(200, 151, 3)</td>
      <td>90600</td>
      <td>color</td>
    </tr>
  </tbody>
</table>
</div>




```python
[np.mean(img[:, :, 0]) for img in test["pictures"]]
```




    [140.3127550312283, 115.4489122490896, 39.40473509933775]




```python
training_df["avg_blue"] = color_channel_avg(training_df, 0) #blue
training_df["avg_green"] = color_channel_avg(training_df, 1) #green
training_df["avg_red"] = color_channel_avg(training_df, 2) #red
```


```python
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
      <th>type</th>
      <th>avg_blue</th>
      <th>avg_green</th>
      <th>avg_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
      <td>(655, 440, 3)</td>
      <td>864600</td>
      <td>color</td>
      <td>140.312755</td>
      <td>137.023931</td>
      <td>129.614445</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
      <td>(976, 1301, 3)</td>
      <td>3809328</td>
      <td>color</td>
      <td>115.448912</td>
      <td>115.112264</td>
      <td>92.376852</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
      <td>(200, 151, 3)</td>
      <td>90600</td>
      <td>color</td>
      <td>39.404735</td>
      <td>43.188742</td>
      <td>40.581093</td>
    </tr>
    <tr>
      <th>358</th>
      <td>6</td>
      <td>[[[150, 202, 226], [150, 202, 226], [150, 202,...</td>
      <td>(326, 328, 3)</td>
      <td>320784</td>
      <td>color</td>
      <td>132.903776</td>
      <td>172.298388</td>
      <td>196.274727</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>[[[6, 5, 3], [6, 5, 3], [6, 5, 3], [6, 5, 3], ...</td>
      <td>(337, 382, 3)</td>
      <td>386202</td>
      <td>color</td>
      <td>114.541077</td>
      <td>104.983291</td>
      <td>74.854056</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sanity check with 1369 image (as seen above)
training_df.loc[1369]
```




    encoding                                                    19
    pictures     [[[75, 76, 44], [95, 97, 60], [107, 108, 66], ...
    shape                                            (512, 768, 3)
    size                                                   1179648
    type                                                     color
    avg_blue                                               151.895
    avg_green                                               136.88
    avg_red                                                92.4081
    Name: 1369, dtype: object



Now that we have the average intensities, we will be finding the dominant color. That is, which color channel is represented in the picture the most (see [link](https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv)) and also [here](http://www.aishack.in/tutorials/dominant-color/). 

Taking a look at the information row for img_1369 (above), the average blue intensity channel has the highest average and this makes sense if we remember the colored image histogram we saw before where the blue channel seem to have the highest contribution. We'll see if this conclusion stays consistent with what the dominant color of the image will be . . . 


```python


# Note: there is also another notebook "dominant_color_example" that I've written that
# goes through step-by-step what each line does inside the function (will eventually be posted in "Posts" on website)

def get_dom_color(image, k=3):
    '''Takes in an image and k number of clusters 
    to use when finding/calculating the dominant color'''
    
    if len(image.shape) < 3: #if grayscale (then only width, height, no color/rgb)
        return "grayscale"

    else:
#         if resize == True: 

        if image.shape[0] > 250 or image.shape[1] > 250: 
            image = cv2.resize(image, (250, 250), interpolation = cv2.INTER_AREA)

        image = image.reshape((image.shape[0] * image.shape[1], 3)) #reshape into 2D array 

        clt = KMeans(n_clusters=k)
        labels = clt.fit_predict(image)


        label_counts = Counter(labels)

        dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

        return list(dom_color)

```

Now let's take a look at what the calculated dominant color is, and how it compares to the features we've calculated so far. It'll also be helpful to play around with some of the parameters to figure out what to use to get the best/most useful results. 


```python
plt.imshow(img_1369)
```




    <matplotlib.image.AxesImage at 0x1c30140a58>




![png](/assets/img/image_class/output_75_1.png)



```python
import time
```


```python
start = time.time() 
d = get_dom_color(img_1369)
end = time.time()

print("Time it takes: ", end - start)
d
```

    Time it takes:  0.6984419822692871





    [155.8551028979393, 139.06948951930138, 91.80722385552228]



The dominant color generated from the rgb values above can be seen in the following color patch: 

<center><img src='/assets/img/image_class/orig.png'></center>

These rgb values were generated using a shrunken image (by default in the function to make things faster), and it is still evident that this dominant color is very representative of the yellow-autumn/golden color dominant in the original zebra image of 1369.

You'll notice too, that if we gather/generate the dominant color using only one cluster, we simply get the averaged rgb values that we found before. See below: 


```python
start = time.time()
d = get_dom_color(img_1369, 1)
end = time.time()

print("Time it takes: ", end - start)
d
```

    Time it takes:  0.14059901237487793





    [151.89628800000753, 136.88155199999514, 92.40743999999768]



But before we apply this "get_dom_color" function on the training_df data, remember we have some images that are grayscaled, so if we apply it to these images, then the "dominant" color would be gray or black or white. This wouldn't give us any useful information. For instance, take a look below. 


```python
fig, ax = plt.subplots(4, 4, figsize=(10, 8))

row = 0
col = 0
for i in np.arange(len(gray_image_indices)):
    ax[row, col].imshow(training_df.loc[gray_image_indices[i]]["pictures"], cmap="gray")
    col += 1
    if col >=4: 
        col = 0
        row += 1

plt.tight_layout()
```


![png](/assets/img/image_class/output_82_0.png)


Notice, that there are a lot of grayscale images of comets, but in color, comets look something like what's shown below.


```python
skimage.io.imshow(training_df[training_df["encoding"] == 3].iloc[3]["pictures"])
```




    <matplotlib.image.AxesImage at 0x1c7378c470>




![png](/assets/img/image_class/output_84_1.png)



```python
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
      <th>type</th>
      <th>avg_blue</th>
      <th>avg_green</th>
      <th>avg_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
      <td>(655, 440, 3)</td>
      <td>864600</td>
      <td>color</td>
      <td>140.312755</td>
      <td>137.023931</td>
      <td>129.614445</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
      <td>(976, 1301, 3)</td>
      <td>3809328</td>
      <td>color</td>
      <td>115.448912</td>
      <td>115.112264</td>
      <td>92.376852</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
      <td>(200, 151, 3)</td>
      <td>90600</td>
      <td>color</td>
      <td>39.404735</td>
      <td>43.188742</td>
      <td>40.581093</td>
    </tr>
    <tr>
      <th>358</th>
      <td>6</td>
      <td>[[[150, 202, 226], [150, 202, 226], [150, 202,...</td>
      <td>(326, 328, 3)</td>
      <td>320784</td>
      <td>color</td>
      <td>132.903776</td>
      <td>172.298388</td>
      <td>196.274727</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>[[[6, 5, 3], [6, 5, 3], [6, 5, 3], [6, 5, 3], ...</td>
      <td>(337, 382, 3)</td>
      <td>386202</td>
      <td>color</td>
      <td>114.541077</td>
      <td>104.983291</td>
      <td>74.854056</td>
    </tr>
  </tbody>
</table>
</div>




```python
d_color = training_df["pictures"].iloc[:3].apply(get_dom_color)
d_color #small example to test
```




    1043    [236.5683503083357, 229.59880135797755, 191.14...
    1010    [68.13215926491932, 66.12595712098701, 46.1749...
    168     [31.168270137711442, 33.979689366786985, 34.00...
    Name: pictures, dtype: object




```python
d_color = training_df["pictures"].apply(get_dom_color)
d_color #takes forever to run, but it works
```




    1043    [236.29123435939545, 229.34941918580455, 190.9...
    1010    [68.5619909502148, 66.62239819005217, 46.54917...
    168     [31.168270137711442, 33.979689366786985, 34.00...
    358     [166.83636128948234, 206.26625576816235, 224.8...
    30      [122.69572353424392, 111.23224218109345, 78.07...
    254     [24.645368943653594, 19.889265536742073, 27.17...
    1148    [71.63536730717108, 57.68568853743446, 54.3797...
    1110    [132.96867389490154, 126.65584653877629, 120.7...
    339     [32.23801453571749, 125.80041065084907, 166.04...
    316     [240.99902118661305, 224.7881342552754, 205.18...
    259     [30.988868334766217, 18.336383993751866, 21.89...
    952     [40.11713796061386, 43.200471293937184, 33.632...
    944     [93.43391985390049, 114.8864538334549, 90.6034...
    986     [123.2476536184957, 111.95133915432187, 104.82...
    306     [182.4050044616257, 125.82246430695629, 56.991...
    741     [92.46232635982078, 108.5165857740584, 150.222...
    865     [65.15900783290932, 57.469886858137514, 21.136...
    1141    [84.69214058499882, 116.5339178110626, 207.971...
    1321    [26.192328925591692, 26.167117957898526, 30.91...
    1055    [60.78409740071645, 71.9311982291125, 48.57033...
    807     [114.06075007748734, 104.08043186279575, 53.54...
    654     [36.67890428964997, 36.82787668778308, 26.5678...
    1379    [250.71304155615613, 250.71366047748074, 250.7...
    309     [186.2839634941172, 153.19262432483765, 112.23...
    765     [21.405501201701885, 111.15925809035787, 173.5...
    665     [101.22116689279792, 93.62211668929379, 58.914...
    1397    [199.01200911639413, 200.15112201963893, 205.4...
    558     [85.62163351761924, 66.50204946218416, 77.7674...
    342     [169.09160218478073, 69.45668335610347, 22.122...
    1063    [48.916817881895696, 54.118442758172975, 34.62...
                                  ...                        
    856     [38.64917984334859, 45.65435200235622, 19.5806...
    747     [82.82707683164249, 81.94962104322207, 79.9575...
    252     [106.58555144698752, 87.406625309414, 74.84090...
    21      [16.4057366362416, 24.222350530829353, 13.8074...
    1337    [36.21791671865915, 55.02670660176831, 35.7396...
    459     [113.93167657379573, 150.21825538923883, 82.30...
    1184    [247.25343081442168, 238.93240246900737, 212.6...
    276     [55.42331229375625, 26.786773893804884, 17.719...
    955     [224.38537203962915, 232.52431709906477, 234.8...
    1215    [187.62620739930213, 155.87887734645756, 156.0...
    385     [15.900575471473218, 133.41755312523995, 189.0...
    805     [113.09404329576124, 118.03947554591124, 65.45...
    343     [205.96592623267037, 219.07755331418795, 228.3...
    769     [147.30588716578796, 166.77526125925863, 235.5...
    1332    [203.5951522766934, 194.92404351705875, 189.05...
    130                                             grayscale
    871     [27.79963343109009, 40.034897360707575, 19.267...
    1123    [22.97983351728942, 34.24827538634672, 35.5026...
    1396    [192.49585933510488, 142.86785195651188, 115.3...
    87      [55.09646779457767, 36.70433363014034, 26.4178...
    330     [108.41889819134701, 104.89732856138669, 99.66...
    1238    [29.822250479955052, 32.73413618567714, 23.643...
    466     [36.37503308546496, 36.496450828906426, 40.302...
    121     [163.19629210787753, 179.3855248278079, 197.41...
    1044    [18.411423983539763, 22.641165338079446, 20.33...
    1095    [79.88479126230737, 43.56716123303565, 22.8236...
    1130    [49.027123100913954, 66.70919361123188, 50.047...
    1294    [195.04900029343403, 181.8307415014427, 178.52...
    860     [92.45566619250825, 135.90449095712253, 196.34...
    1126    [226.2961902282505, 234.2160027770335, 230.562...
    Name: pictures, Length: 1278, dtype: object




```python
training_df["dom_color"] = d_color
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
      <th>type</th>
      <th>avg_blue</th>
      <th>avg_green</th>
      <th>avg_red</th>
      <th>dom_color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
      <td>(655, 440, 3)</td>
      <td>864600</td>
      <td>color</td>
      <td>140.312755</td>
      <td>137.023931</td>
      <td>129.614445</td>
      <td>[236.29123435939545, 229.34941918580455, 190.9...</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
      <td>(976, 1301, 3)</td>
      <td>3809328</td>
      <td>color</td>
      <td>115.448912</td>
      <td>115.112264</td>
      <td>92.376852</td>
      <td>[68.5619909502148, 66.62239819005217, 46.54917...</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
      <td>(200, 151, 3)</td>
      <td>90600</td>
      <td>color</td>
      <td>39.404735</td>
      <td>43.188742</td>
      <td>40.581093</td>
      <td>[31.168270137711442, 33.979689366786985, 34.00...</td>
    </tr>
    <tr>
      <th>358</th>
      <td>6</td>
      <td>[[[150, 202, 226], [150, 202, 226], [150, 202,...</td>
      <td>(326, 328, 3)</td>
      <td>320784</td>
      <td>color</td>
      <td>132.903776</td>
      <td>172.298388</td>
      <td>196.274727</td>
      <td>[166.83636128948234, 206.26625576816235, 224.8...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>[[[6, 5, 3], [6, 5, 3], [6, 5, 3], [6, 5, 3], ...</td>
      <td>(337, 382, 3)</td>
      <td>386202</td>
      <td>color</td>
      <td>114.541077</td>
      <td>104.983291</td>
      <td>74.854056</td>
      <td>[122.69572353424392, 111.23224218109345, 78.07...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from math import gcd
```


```python
def aspect_ratio(arr): 
    """Given the array of values consisting of shape tuples, return the aspect ratio.
    
    Input:
    Array of shape tuples where shape[0] = h, shape[1] = w and w/h is the aspect ratio
    
    Return: 
    A list of tuple aspect ratios
    
    """
    
    asp_rat = []
    
    for i in arr: 
        w = i[1]
        h = i[0]
        
        common_divisor = gcd(w, h)
        
        w = w/common_divisor
        h = h/common_divisor
        
        asp_rat.append((w, h))
    
    return asp_rat

# aspect_ratio(training_df["shape"].iloc[:3].values) #check
```


```python
a_ratio = aspect_ratio(training_df["shape"].values)
```


```python
training_df["aspect_ratio"] = a_ratio #note: it's (w, h)
training_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encoding</th>
      <th>pictures</th>
      <th>shape</th>
      <th>size</th>
      <th>type</th>
      <th>avg_blue</th>
      <th>avg_green</th>
      <th>avg_red</th>
      <th>dom_color</th>
      <th>aspect_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>14</td>
      <td>[[[26, 19, 50], [26, 19, 50], [29, 20, 51], [3...</td>
      <td>(655, 440, 3)</td>
      <td>864600</td>
      <td>color</td>
      <td>140.312755</td>
      <td>137.023931</td>
      <td>129.614445</td>
      <td>[236.29123435939545, 229.34941918580455, 190.9...</td>
      <td>(88.0, 131.0)</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>14</td>
      <td>[[[53, 89, 28], [62, 102, 42], [52, 91, 34], [...</td>
      <td>(976, 1301, 3)</td>
      <td>3809328</td>
      <td>color</td>
      <td>115.448912</td>
      <td>115.112264</td>
      <td>92.376852</td>
      <td>[68.5619909502148, 66.62239819005217, 46.54917...</td>
      <td>(1301.0, 976.0)</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>[[[50, 54, 53], [54, 58, 57], [58, 62, 61], [5...</td>
      <td>(200, 151, 3)</td>
      <td>90600</td>
      <td>color</td>
      <td>39.404735</td>
      <td>43.188742</td>
      <td>40.581093</td>
      <td>[31.168270137711442, 33.979689366786985, 34.00...</td>
      <td>(151.0, 200.0)</td>
    </tr>
    <tr>
      <th>358</th>
      <td>6</td>
      <td>[[[150, 202, 226], [150, 202, 226], [150, 202,...</td>
      <td>(326, 328, 3)</td>
      <td>320784</td>
      <td>color</td>
      <td>132.903776</td>
      <td>172.298388</td>
      <td>196.274727</td>
      <td>[166.83636128948234, 206.26625576816235, 224.8...</td>
      <td>(164.0, 163.0)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>[[[6, 5, 3], [6, 5, 3], [6, 5, 3], [6, 5, 3], ...</td>
      <td>(337, 382, 3)</td>
      <td>386202</td>
      <td>color</td>
      <td>114.541077</td>
      <td>104.983291</td>
      <td>74.854056</td>
      <td>[122.69572353424392, 111.23224218109345, 78.07...</td>
      <td>(382.0, 337.0)</td>
    </tr>
  </tbody>
</table>
</div>


The outputs below were mainly used to inspect what the arrays and objects looked like.

```python
training_df.dtypes
```




    encoding          int64
    pictures         object
    shape            object
    size              int64
    type             object
    avg_blue        float64
    avg_green       float64
    avg_red         float64
    dom_color        object
    aspect_ratio     object
    dtype: object




```python
test_df["pictures"][0] #numpy.ndarray
```




    array([[[ 74,  79,  80],
            [ 50,  55,  56],
            [ 48,  53,  54],
            ...,
            [ 67,  59,  59],
            [ 58,  51,  48],
            [ 57,  53,  48]],
    
           [[ 95, 100, 101],
            [ 52,  57,  58],
            [ 39,  44,  45],
            ...,
            [ 67,  60,  57],
            [ 61,  54,  51],
            [ 63,  57,  52]],
    
           [[ 84,  90,  89],
            [ 49,  55,  54],
            [ 50,  56,  55],
            ...,
            [ 67,  60,  57],
            [ 62,  56,  51],
            [ 65,  59,  54]],
    
           ...,
    
           [[ 37,  57,  68],
            [ 29,  49,  60],
            [ 20,  40,  51],
            ...,
            [122, 105, 102],
            [159, 142, 139],
            [160, 144, 138]],
    
           [[ 40,  63,  71],
            [ 36,  59,  67],
            [ 28,  51,  59],
            ...,
            [155, 140, 138],
            [188, 173, 170],
            [179, 164, 161]],
    
           [[ 46,  70,  76],
            [ 45,  69,  75],
            [ 39,  63,  69],
            ...,
            [184, 168, 169],
            [194, 178, 179],
            [189, 174, 172]]], dtype=uint8)


The following commented block was used to expore my training and testing dataframes into a csv for later use in a separate notebook. 


```python
## exporting training and testing dataframe as csv's for later use
# training_df.to_csv(r"/Users/samanthatang/Desktop/exported_training.csv", index=False, header=True)
# val.to_csv(r"/Users/samanthatang/Desktop/exported_validation.csv", index=False, header=True)
# test_df.to_csv(r"/Users/samanthatang/Desktop/exported_testing.csv", index = False, header=True)
```


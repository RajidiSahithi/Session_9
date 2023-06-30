# DESIGN A CNN USING DILATION CONVOLUTION AND DEPTH WISE CONVOLUTION

## TARGET :
* The architecture must have 4 blocks (C1,C2,C3,C40) (No MaxPooling,  3 convolutions per block, where the last one has a stride of 2 )
* RF must be more than 44
* one of the layers must use Depthwise Separable Convolution
* one of the layers must use Dilated Convolution
* use albumentation library (horizontal flip, shiftScaleRotate, coarseDropout)
* Total Params to be less than 200k
* Acheive 85% accuracy

## CONTENTS :
- [DATASET](#dataset)
- [IMPORTING_LIBRARIES](#importing_libraries)
- [SET_THE_ALBUMENTATIONS](#set_the_albumentations)
- [DATA_AUGMENTATIONS](#data_augmentations)
- [SET_DATA_LOADER](#Set_Data_Loader)
- [CNN_MODEL](#cnn_model)
- [TRAINING_THE_MODEL](training_the_model)
- [RESULTS](results)


## DATASET 
### CIFAR DATASET
CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color (RGB) images containing one of 10 object classes, with 6000 images per class.

## IMPORTING_LIBRARIES
Import the required libraries. 
* NumPy is used for numerical operations. The torch library is used to import Pytorch.
* Pytorch has an nn component that is used for the abstraction of machine learning operations. 
* The torchvision library is used so that to import the CIFAR-10 dataset. This library has many image datasets and is widely used for research. The transforms can be imported to resize the image to an equal size for all the images. 
* The optim is used to train the neural Networks.
* MATLAB libraries are imported to plot the graphs and arrange the figures with labeling
* Albumenations are imported for Middle Man's Data Augmentation Strategy
* cv2 is imported 

## SET_THE_ALBUMENTATIONS
* cv2.setNumThreads(0) sets the number of threads used by OpenCV to 0. This is done to avoid a deadlock when using OpenCV’s resize method with PyTorch’s dataloader1.

* cv2.ocl.setUseOpenCL(False) disables the usage of OpenCL in OpenCV2 and is used when you want to disable the usage of OpenCL.

* The  class is inherited from torchvision.datasets.CIFAR10. It overrides the __init__ and __getitem__ methods of the parent class. The __getitem__ method returns an image and its label after transforming the image3. (This is to be done while using Albumenations)


## DATA_AUGMENTATIONS
For this import albumentations as A

Middle-Class Man's Data Augmentation Strategy is used. Like
### HorizontalFlip :
<pre>
Syntax : A.HorizontalFlip(p=0.5)
Flip the input horizontally around the y-axis.
Args:     
p (float): the probability of applying the transform. Default: 0.5.
Applied only to Training data
</pre>
### ShiftScaleRotate :
<pre>
Syntax:
A.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
Randomly apply affine transforms: translate, scale, and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in the range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        p(float): the probability of applying the transform. Default: 0.5.

Applied only to Training data
</pre>
### Cutout
<pre>
Syntax:
 A.CoarseDropout(max_holes = 32, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=((0.4914, 0.4822, 0.4465)))
 It is similar to a cutout

    Args:
        max_holes(int): The maximum number of rectangular regions to be masked. (for CIFAR10 Dataset its 32X32)
        max_height(int): The maximum height of the rectangular regions. 
        max_width(int): The maximum width of the rectangular regions.
        min_holes(int): The minimum number of rectangular regions to be masked.
        min_height(int): The minimum height of the rectangular regions.
        min_width(int): The minimum width of the rectangular regions.
        fill_value(float): The value to be filled in the masked region. It can be a tuple or a single value . 
            It is usually equal to the mean of dataset for CIFAR10 its (0.4914, 0.4822, 0.4465)
       
Applied only to Training data 
</pre>
### RandomBrightnessContrast
<pre>
    A.RandomBrightnessContrast(p=0.5)
    Brightness and contrast are adjusted with a probability of 0.5
</pre>
### Normalize
<pre>
Syntax:
    A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

Normalization is a common technique used in deep learning to scale the pixel values of an image to a standard range. This is done to ensure that the input features have similar ranges and are centered around zero. 
Normalization is done with respect to mean and standard Deviation.
For CIFAR10 (RGB) will have 3 means and 3 standard deviation that is equal to 
(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

Applied to training and test data

</pre>

### ToTensorV2
<pre>
Syntax:
    ToTensorV2()

To make this function work we need to ToTensorV2 from albumentations.pytorch.transforms
It is a class in the PyTorch library that converts an image to a PyTorch tensor. It is part of the torchvision.transforms module and is used to preprocess images before feeding them into a neural network. 

Applied to training and test data
</pre>
 #### PRINTED TRAIN_TRANSFORMS and TEST_TRANSFORMS 
 <pre>
 Train_Transfoms:
 Compose([
  HorizontalFlip(always_apply=False, p=0.5),
  ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.09999999999999998, 0.10000000000000009), rotate_limit=(-45, 45), interpolation=1, border_mode=4, value=None, mask_value=None),
  RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
  CoarseDropout(always_apply=False, p=0.5, max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16),
  Normalize(always_apply=False, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616), max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
Compose([
  Normalize(always_apply=False, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201), max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
</pre>

## SET_DATA_LOADER
* Batch Size = 128
* Number of Workers = 2
* CUDA is used

#### PRINTED TRAIN and TEST LOADER:
<pre>
Files already downloaded and verified
Files already downloaded and verified
<torch.utils.data.dataloader.DataLoader object at 0x000002D4A245B710>
length of train_loader 391
<torch.utils.data.dataloader.DataLoader object at 0x000002D4A244DA50>
length of test_loader 79

</pre>
#### SAMPLE IMAGES IN TRAIN LOADER
![alt text](https://github.com/RajidiSahithi/Session_9/blob/main/Images%20S9/Sample_training_data.png)

## CNN_MODEL

#### MODEL
<pre>
BLOCK1 (C1)
 This Block Contains 3 Convolution layers with Kernal Size 3X3 and Stride is 1 . (each layer has Batch Normalization, Activation Function (Relu) and droput of 1%)
 Deptwise convolution is implemented in this block. As we know Depthwise convolution is to be followed by point wise convolution.(number of groups=4)
  (conv11): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (conv12): Sequential(
    (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), groups=4)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (conv13): Sequential(
    (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
  )
</pre>
<pre>
BLOCK2 (C2)
  This Block Contains 3 Convolution layers with Kernal Size 3X3 and Stride is 1 for all layers. (each layer has Batch Normalization, Activation Function (Relu) and droput of 1%)
  In the last layer of this block dialation Convolution is used.
  For using dilation Covolution we need to maintain Receptive fields and the number of output channels. Padding is added based on this constraint. Stride is 1 all layers. 
  Dilation is to be followed by a skip connection. Even the kernel size is 3 the Effective kernel size is 7 (when dilation = 3).
  
  (conv21): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), dilation=(3, 3))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), dilation=(3, 3))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout(p=0.01, inplace=False)
    (8): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(9, 9), dilation=(3, 3))
    (9): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (relu2): ReLU(inplace=True)
  (drop2): Dropout(p=0.01, inplace=False)
  </pre>
  <pre>
  BLOCK3 (C3)

  This Block Contains 3 Convolution layers with Kernal Size 3X3 and Stride is 1 for all layers. (each layer has Batch Normalization, Activation Function (Relu), and droput of 1%)

  (conv31): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (conv32): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (conv33): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  </pre>
  <pre>
  BLOCK4 / OUTPUT BLOCK (C4O)
  This Block Contains 3 Convolution layers with Kernal Size 3X3 and Stride is 1 for all layers. (each layer has Batch Normalization, Activation Function (Relu), and droput of 1%)
  In the last layer of this block dialation Convolution is used.
  For using dilation Covolution we need to maintain Receptive fields and number of output channels. Padding is added based on this constraint. Stride is 1 all layers. 
  Dilation is to be followed by a skip connection. Even the kernel size is 3 the Effective kernel size is 7 (when dilation = 3).
  
  (conv41): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), dilation=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
    (4): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), dilation=(3, 3))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout(p=0.01, inplace=False)
    (8): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(9, 9), dilation=(3, 3))
    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (relu1): ReLU(inplace=True)
  (drop1): Dropout(p=0.01, inplace=False)

</pre>
<pre>
At last Adaptive Average Pooling (to reduce to 10) and Fully Connected are used.
(gap): AdaptiveAvgPool2d(output_size=1)
  (conv5): Sequential(
    (0): Conv2d(64, 10, kernel_size=(1, 1), stride=(1, 1))
  )
</pre>



## TRAINING_THE_MODEL
The train function takes the model, device, train_loader, optimizer, and epoch as inputs. It performs the following steps:

* Sets the model to train mode, which enables some layers and operations that are only used during training, such as dropout and batch normalization.
* Creates a progress bar object from the train_loader, which is an iterator that yields batches of data and labels from the training set.
* Initializes two variables to keep track of the number of correct predictions and the number of processed samples.
* Loops over the batches of data and labels, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Calls optimizer.zero_grad() to reset the gradients of the model parameters to zero, because PyTorch accumulates them on subsequent backward passes.
* Passes the data through the model and obtains the predictions (y_pred).
* Calculates the loss between the predictions and the labels using the negative log-likelihood loss function (F.nll_loss).
* Appends the loss to the train_losses list for later analysis.
* Performs backpropagation by calling loss.backward(), which computes the gradients of the loss with respect to the model parameters.
* Performs optimization by calling optimizer.step(), which updates the model parameters using the gradients and the chosen optimization algorithm (such as SGD or Adam).
* Updates the progress bar with the current loss, batch index, and accuracy. The accuracy is computed by comparing the predicted class (the index of the max log-probability) with the true class, and summing up the correct predictions and processed samples.
* Appends the accuracy to the train_acc list for later analysis.

The test function takes the model, device, and test_loader as inputs. It performs the following steps:

* Sets the model to eval mode, which disables some layers and operations that are only used during training, such as dropout and batch normalization.
* Initializes two variables to keep track of the total test_loss and the number of correct predictions.
* Uses a torch.no_grad() context manager to disable gradient computation, because we don’t need it during testing and it saves memory and time.
* Loops over the batches of data and labels from the test set, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Passes the data through the model and obtains the output (predictions).
* Adds up the batch loss to the total test loss using the negative log-likelihood loss function (F.nll_loss) with reduction=‘sum’, which means it returns a scalar instead of a vector.
* Compares the predicted class (the index of the max log-probability) with the true class, and sums up the correct predictions.
* Divides the total test loss by the number of samples in the test set to get the average test loss, and appends it to the test_losses list for later analysis.

* creates an instance of the Adam optimizer, which is a popular algorithm that adapts the learning rate for each parameter based on the gradient history and the current gradient. You pass the model parameters, the initial learning rate (lr), and some other hyperparameters to the optimizer constructor. 
* creates an instance of the OneCycleLR scheduler, which is a learning rate policy that cycles the learning rate between two boundaries with a constant frequency. You pass the optimizer, the maximum learning rate (0.01), the number of epochs (30), and the number of steps per epoch (len(train_loader)) to the scheduler constructor.
* Defines a constant for the number of epochs = 30, which is the number of times you iterate over the entire training set.
* Prints out a summary of the average test loss, accuracy, and number of samples in the test set. 


## [RESULTS]

I achieved 85% of accuracy at the 50th epoch. Initially model is underfitting. As the number of epochs increased it worked well.

I missed the screenshot of the result.


#### MODEL SUMMARY
<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 30, 30]             448
       BatchNorm2d-2           [-1, 16, 30, 30]              32
              ReLU-3           [-1, 16, 30, 30]               0
           Dropout-4           [-1, 16, 30, 30]               0
            Conv2d-5           [-1, 16, 28, 28]             592
       BatchNorm2d-6           [-1, 16, 28, 28]              32
              ReLU-7           [-1, 16, 28, 28]               0
           Dropout-8           [-1, 16, 28, 28]               0
            Conv2d-9           [-1, 16, 28, 28]             272
      BatchNorm2d-10           [-1, 16, 28, 28]              32
             ReLU-11           [-1, 16, 28, 28]               0
          Dropout-12           [-1, 16, 28, 28]               0
           Conv2d-13           [-1, 32, 22, 22]           4,640
      BatchNorm2d-14           [-1, 32, 22, 22]              64
             ReLU-15           [-1, 32, 22, 22]               0
          Dropout-16           [-1, 32, 22, 22]               0
           Conv2d-17           [-1, 32, 16, 16]           9,248
      BatchNorm2d-18           [-1, 32, 16, 16]              64
             ReLU-19           [-1, 32, 16, 16]               0
          Dropout-20           [-1, 32, 16, 16]               0
           Conv2d-21           [-1, 16, 28, 28]           4,624
      BatchNorm2d-22           [-1, 16, 28, 28]              32
             ReLU-23           [-1, 16, 28, 28]               0
          Dropout-24           [-1, 16, 28, 28]               0
           Conv2d-25           [-1, 32, 26, 26]           4,640
      BatchNorm2d-26           [-1, 32, 26, 26]              64
             ReLU-27           [-1, 32, 26, 26]               0
          Dropout-28           [-1, 32, 26, 26]               0
           Conv2d-29           [-1, 32, 24, 24]           9,248
      BatchNorm2d-30           [-1, 32, 24, 24]              64
             ReLU-31           [-1, 32, 24, 24]               0
          Dropout-32           [-1, 32, 24, 24]               0
           Conv2d-33           [-1, 64, 22, 22]          18,496
      BatchNorm2d-34           [-1, 64, 22, 22]             128
             ReLU-35           [-1, 64, 22, 22]               0
          Dropout-36           [-1, 64, 22, 22]               0
           Conv2d-37           [-1, 64, 16, 16]          36,928
      BatchNorm2d-38           [-1, 64, 16, 16]             128
             ReLU-39           [-1, 64, 16, 16]               0
          Dropout-40           [-1, 64, 16, 16]               0
           Conv2d-41           [-1, 32, 10, 10]          18,464
      BatchNorm2d-42           [-1, 32, 10, 10]              64
             ReLU-43           [-1, 32, 10, 10]               0
          Dropout-44           [-1, 32, 10, 10]               0
           Conv2d-45           [-1, 64, 22, 22]          18,496         
      BatchNorm2d-46           [-1, 64, 22, 22]             128
             ReLU-47           [-1, 64, 22, 22]               0
          Dropout-48           [-1, 64, 22, 22]               0
AdaptiveAvgPool2d-49             [-1, 64, 1, 1]               0
           Conv2d-50             [-1, 10, 1, 1]             650
================================================================
Total params: 127,578
Trainable params: 127,578
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.02
Params size (MB): 0.49
Estimated Total Size (MB): 6.52
----------------------------------------------------------------
</pre>

* Total number of Convolution Layers = 13

* Total Receptive Field is greater than 44 (49 in this case)
![alt text](https://github.com/RajidiSahithi/Session_9/blob/main/Images%20S9/RF.png)

# ANALYSIS:
* Usage of albumenations is understood.
* This model took more time/epoch compared to previous models.
* Deptwise Convoltions, Dilation is used with max-pooling and striding.
* This model can be trained further.
* Achieved No of parameters < 200K
* Accuracy > 85%
* RF > 44




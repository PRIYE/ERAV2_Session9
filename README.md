# Classification CIFAR-10 dataset using Depthwise & Dilated Convolution
## Objective
1. Model has the architecture to C1C2C3C40
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP
6. use albumentation library
   * horizontal flip
   * shiftScaleRotate
   * coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
8. achieve 85% accuracy with total params to be less than 200k

## Model Architecture
![image](https://github.com/PRIYE/ERAV2_Session9/assets/7592375/4d738c87-6307-4392-8f18-f988be5af4a9)

1. Total Paramters : 191,760
2. Used Dilated Convolution at end of C1, C2 and C3 block
3. Used Depthwise Separable Convolution at C2, C3 and C4
4. Took 55 iterations to reach 85% accuracy.
   
## Image Augumentation using Albumentation library
![image](https://github.com/PRIYE/ERAV2_Session9/assets/7592375/f5ef7248-78a7-4f99-a2d4-3446b4be22f7)


## Result
![image](https://github.com/PRIYE/ERAV2_Session9/assets/7592375/34356cb4-2704-46c0-b78f-e11477ed01c7)

## Performance Metric
![image](https://github.com/PRIYE/ERAV2_Session9/assets/7592375/b4baf515-255a-4fbc-89e9-0865817e3acf)



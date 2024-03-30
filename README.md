# Classification CIFAR-10 dataset using Depthwise & Dilated Convolution
## Objective
1. Model has the architecture to C1C2C3C40
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP
6. use albumentation library
   6.1. horizontal flip
   6.2. shiftScaleRotate
   6.3. coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
8. achieve 85% accuracy with total params to be less than 200k

## Model Architecture
![image](https://github.com/PRIYE/ERAV2_Session9/assets/7592375/400c4ede-74d6-448f-b5e4-a8004ad1fa40)

1. Total Paramters : 190,608
2. Used Dilated Convolution at end of C1, C2 and C3 block
3. Used Depthwise Separable Convolution at C2, C3 and C4
4. Used following augumentation from Albumentation library
  4.1. HorizontalFlip
  4.2. ShiftScaleRotate
  4.3. CoarseDropout
  4.4. ColorJitter

## Result
![image](https://github.com/PRIYE/ERAV2_Session9/assets/7592375/34356cb4-2704-46c0-b78f-e11477ed01c7)

## Performance Metric
![image](https://github.com/PRIYE/ERAV2_Session9/assets/7592375/b4baf515-255a-4fbc-89e9-0865817e3acf)



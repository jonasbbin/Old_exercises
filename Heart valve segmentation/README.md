# Disclaimer:
This was an old exercise. Therefore it might be possible that current approaches largely differ from this approach. Also keep in mind that this task was solved under potential restrictions (solution approaches, model usage, hardware, time).
# Heart Valve Segmentation
In this task we need to automatically segment the heart valve in some ultrasound videos. We have 195 training images, where the heart valves are segmented by an expert. 

![image](ultrasound.jpg)

*Ultrasound image* 


![image](solution.jpg)

*Expert Segmentation*
## Our Approach:
We train our own CNN using a U-Net structure. Due to computational constraints we only use a small network and compute only on very small images sizes (64x64). But note that this approach can be scaled in size. Due to the small training set we use data augmentation methods, like small rotations, gaussian noise and image flips. This could be even more extended by using crops, salt and pepper noise etc. Our output image is thus more like a proof of concept than the best possible output

![image](detection.jpg)

*Our Output using just a small model*
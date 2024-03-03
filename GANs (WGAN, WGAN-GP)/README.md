
# GAN 
This notebook has three versions of GANs implemented on the MNIST Dataset. There is the normal GAN using the BCE Losse, a Wasserstein GAN using the original proposed parameter clipping and a Wasserstein GAN using gradient penalty. The Wasserstein GAN benefits from improved stability and therefore less mode colapse. However, the cost of training also incrases. Feel free to play around with the paramaters and see the changes for yourself. Note, that training can get quite expensive.
Here are some example outputs:

![image](GANoutput.jpg)

![image](GAN_smalleroutput.jpg)
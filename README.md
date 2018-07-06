# pytorch-unet-segnet

These architectures have shown good results in semantic segmentation, image reconstruction (denoising, super-resolution).

[Unet](https://arxiv.org/abs/1505.04597) (unet.py)
![Unet](https://github.com/trypag/pytorch-unet-segnet/blob/master/docs/unet/u-net-architecture.png)

[SegNet](https://arxiv.org/abs/1511.00561) (segnet.py)
![SegNet](https://github.com/trypag/pytorch-unet-segnet/blob/master/docs/segnet/segnet.png)

ModSegNet (modsegnet.py)
![ModSegNet](https://github.com/trypag/pytorch-unet-segnet/blob/master/docs/modsegnet/1.png)

**I would encourage you to use SegNet if you don't see any major performance decrease with Unet**. SegNet uses maximum unpooling during the upsampling step, reusing the maximum pooling indices from the encoding step. Making the upsampling procedure parameter free, where Unet makes use of transpose convolution (filters) to learn how to upsample. **SegNet will be lighter and faster**. I would also recommend to tune SegNet by replacing 7x7 convolution by 2 3x3 convolutions, reducing the number of dropout layers and without batchnorm (see modsegnet.py)

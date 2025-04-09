# Image_Resolution-SRGAN
SRGAN (Super-Resolution Generative Adversarial Network) is a deep learning model designed to generate high-resolution images from low-resolution inputs, and I applied it to the DIV2K dataset for training and evaluation.

Results (Testing example)
(Comparison between Original, Bicubic, SRResNet, SRGAN-5epochs, SRGAN-10epochs)

General Metrics Compared: 
PSNR (Peak Signal-to-Noise Ratio) – Higher is better. 
SSIM (Structural Similarity Index) – Higher is better. 
MSE (Mean Squared Error) – Lower is better.

![image](https://github.com/user-attachments/assets/e4c55a1f-8adb-4d60-a79a-4ed416dfdc42)


Overall Observations:
● SRResNet consistently gives the best PSNR, SSIM, and lowest MSE.
● SRGAN (14 epochs) usually performs better than SRGAN (5 epochs) as well as Bicubic.
● Bicubic interpolation is always the weakest performer or similar to SRGAN (5 epochs).



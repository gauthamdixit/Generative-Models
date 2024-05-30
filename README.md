# Generative-Models

Stable Diffusion:

Uses VAE encoder-decoder architecture to reduce image size for computational purposes during training and inference. After the encoder, the data is passed to a UNET which determines how much noise needs to be removed from the image to generate a new image. Attention mechanism is used to correlate text prompts to images. demo.ipynb has a variable "cfg_scale" which determines how much the generated images should follow the prompt. To run, change the prompt and negative prompt variable in demo.ipynb and make sure the prompt has less than the max 77 tokens.

NOTE:

The ckpt file which containes the saved weights is NOT included as it is too large. In order to run this, you will need the ckpt file (which I can provide) or train the model from scratch which is extremely resource intesive to say the least.

Algorithm:

![alt text](images/image.png)

Examples:

![alt text](images/Astronaut_on_Dino_Mars.png) ![alt text](images/naruto_ganesha.png) ![alt text](images/Pikachu_Spiderman.png)


Style Transfer:

Implements algorithm from paper: A Neural Algorithm of Artistic Style https://arxiv.org/pdf/1508.06576v2

Takes a source image and recreates it in a specific artistic style. 

Algorithm:
![alt text](images/image-1.png)

Examples:

Content image:

![alt text](images/Taj_Mahal.jpg)

Style image:

![alt text](images/style.jpg)

Style Transfer:

![alt text](images/image_style_w=0.01__content_w=10000.0__steps=1000.png)

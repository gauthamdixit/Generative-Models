import os
import tensorflow as tf
#print(tf.__version__)
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor,dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def save_image(image, style_w,content_w,steps):
    save_path = "C:\\Users\\gauth\\OneDrive\\Desktop\\AI\\Generative\\StyleTransfer\\results\\image_style_w=" +str(style_w) + "__content_w="+ str(content_w)+ "__steps="+ str(steps)+".png"
    tensor_to_image(image).save(save_path)

#images to be used for content and style
# content_path = "C:\\Users\\gauth\\OneDrive\\Desktop\\AI\\Generative\\StyleTransfer\\style_content_images\\naruto.jpg"
# style_path = "C:\\Users\\gauth\\OneDrive\\Desktop\\AI\\Generative\\StyleTransfer\\style_content_images\\one_piece.jpg"

style_path = "C:\\Users\\gauth\\OneDrive\\Desktop\\AI\\Generative\\StyleTransfer\\style_content_images\\style.jpg"
content_path = "C:\\Users\\gauth\\OneDrive\\Desktop\\AI\\Generative\\StyleTransfer\\style_content_images\\Taj_Mahal.jpg"


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img,channels = 3)
    img = tf.image.convert_image_dtype(img,tf.float32)

    shape = tf.cast(tf.shape(img)[:-1],tf.float32)
    long_dim = max(shape)
    scale = max_dim/long_dim
    new_shape = tf.cast(shape*scale,tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis,:]
    return img

def imshow(image,title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image,axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)


x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x,(224,224))

#VGG19 is a nn architecture used for image classification. It contains 5 blocks of CNNs where each block has 2-3 Conv2D layers that have 64,128,256,512,512 
#input channels respectively. The activation function between each layer is the RELU(). Each block is completed with a Max Pooling with 2x2 pool size and stride 2.
#The output of these blocks is then flattened and then passed to a final fully connected layer which contains 2 linear layers of size 4096. 
#The final output layer is of size 1000 and ends on a softmax activation. 

vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
#The VGG nn is used for classification. To generate images, we can choose intermediate layers within the network to use for generation.
#TODO: experiment with different layers being chosen.
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
#get the specified layers from the network
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)
#to get the style of an image we can calculate the gram matrix which is the inner product of the filters of x amount of layers. Feature maps represent the 
#activations of neurons in a certain layer. These activations correspond to different filters or channels in that layer. By capturing correlations between different features,
#we can get the style of an image and this is done by computing the inner product of the vectorized feature maps aka the gram matrix.
#When the Gram matrix is calculated, each element (i, j) in the matrix represents the correlation between the i-th and j-th features. 
#High values indicate that these features are correlated in terms of style.
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd',input_tensor,input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2],tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self,style_layers,content_layers):
        super(StyleContentModel,self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def call(self,inputs):
        inputs = inputs *255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs,content_outputs = (outputs[:self.num_style_layers],outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name:value for content_name,value in zip(self.content_layers,content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}
    
extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)
def clip_0_1(image):
    return tf.clip_by_value(image,clip_value_min=0.0,clip_value_max=1.0)
opt = tf.keras.optimizers.Adam(learning_rate = 0.02,beta_1 = 0.99, epsilon = 1e-1)
# total loss of the style transfer alg is alpha * content loss + beta * style loss
def style_content_loss(outputs,style_w,content_w):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    style_loss *= style_w/num_style_layers
    content_loss *= content_w/num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image,style_w,content_w):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs,style_w,content_w)
    
    grad = tape.gradient(loss,image)
    opt.apply_gradients([(grad,image)])
    image.assign(clip_0_1(image))

start = time.time()
epochs = 10
steps_per_epoch = 100
step = 0
style_weights = [1e-2]
content_weights = [1e4]
for i in range(len(style_weights)):
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step+=1
            train_step(image,style_weights[i],content_weights[i])
            #print(".",end = '',flush= True)
        print("completed epoch: ",n+1)
    save_image(image,style_weights[i],content_weights[i],step)
    step = 0
    

        

end = time.time()
print("Total time: {:.1f}".format(end-start))










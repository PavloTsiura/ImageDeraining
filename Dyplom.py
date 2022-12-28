#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt


# In[2]:


path = 'D:\\.Files\\AI\\data'


# In[2]:


did_train = keras.preprocessing.image_dataset_from_directory(
    'D:\.Files\AI\data\DID-MDN-datasets\DID-MDN-training',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 512),
    shuffle=True,
    interpolation='bilinear')


# In[3]:


def split_x_y(x):
    return (x[:, :, :256, :] / 255., x[:, :, 256:, :] / 255.)


# In[4]:


did_train_splitted = did_train.map(split_x_y)


# In[22]:


plt.figure(figsize=(40, 40))
for X, Y in did_train_splitted.take(10):
    for i in range(4):
        #ax = plt.subplot(3, 3, i + 1)
        plt.imshow(X[i].numpy())
        plt.show()
        plt.imshow(Y[i].numpy())
        plt.show()
        #plt.title(int(labels[i]))
        plt.axis("off")


# In[23]:


rain800 = keras.preprocessing.image_dataset_from_directory(
    'D:\.Files\AI\data\\rain800_idcgan\\train',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 512),
    shuffle=False,
    interpolation='bilinear')


# In[24]:


rain800_splited = rain800.map(split_x_y)


# In[ ]:





# In[25]:


rain1400_X = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\rainy_image_dataset\\training\\rainy_image',
    #labels='inferred',
    label_mode=None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    interpolation='bilinear')


# from PIL import Image
# import glob
# 
# 
# images = glob.glob('D:\\.Files\\AI\\data\\rainy_image_dataset\\training\\ground_truth\\ground_truth\\*.jpg')
# 
# for image in images:
#     with open(image, 'rb') as file:
#         img = Image.open(file)
#         for i in range(1,15):
#             img.save(image.replace('ground_truth\\ground_truth', 'ground_truth2')[:-4] + '_' + str(i) + '.jpg')

# In[26]:


rain1400_Y = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\rainy_image_dataset\\training\\ground_truth2',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    interpolation='bilinear')


# In[27]:


rain1400 =  tf.data.Dataset.zip((rain1400_X.map(lambda x: x / 255.), rain1400_Y.map(lambda x: x / 255.)))


# In[28]:


rain1400


# In[29]:


train = did_train_splitted.concatenate(rain1400).concatenate(rain800_splited)


# In[31]:


plt.figure(figsize=(40, 40))
for images, labels in train.take(10):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        #plt.title(int(labels[i]))
        plt.axis("off")


# In[5]:


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    #x = keras.layers.LeakyReLU()(x)
    #x = Activation('relu')(x)
    x = keras.activations.gelu()(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    #x = keras.layers.LeakyReLU()(x)
    #x = Activation('relu')(x)
    x = keras.activations.gelu()(x)
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    
    """ Bridge """
    b1 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)


    """ Decoder """
    d2 = decoder_block(b1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(3, 1, padding="same", activation="relu")(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model


# In[6]:


model = build_resnet50_unet((256,256,3))


# In[32]:


model = tf.keras.models.load_model('dyplom1.h5')


# In[18]:


model.summary()


# In[18]:


@tf.function
def ssim_loss(y_true, y_pred):
    return 1 - tf.image.ssim_multiscale(y_true, y_pred, 1)


# In[38]:


model.save_weights('weights.h5')


# In[41]:


model.compile(optimizer= 'adam', #keras.optimizers.SGD(momentum = 0.01),
              loss= 'mse',
              metrics=['MSE'])


# In[44]:


model.load_weights('weights.h5')


# In[92]:


history1 = model.fit(train, epochs = 1)


# In[93]:


history1 = model.fit(train, epochs = 1)


# In[94]:


history1 = model.fit(train, epochs = 1)


# In[21]:


plt.plot(history.history['loss'])


# In[82]:


model.save('dyplom!.h5')


# In[7]:


model = tf.keras.models.load_model('dyplom!.h5')


# In[154]:


plt.figure(figsize=(40, 40))
for X, Y in rain800_splited_t.take(10):
    for i in range(4):
        ax = plt.subplot(3, 3, i + 1)
        plt.figure(figsize=(40, 40))
        plt.imshow(model.predict(X)[i])
        plt.show()
        #plt.figure(figsize=(40, 40))
        #plt.imshow(Y[i])
        #plt.show()
        plt.figure(figsize=(40, 40))
        plt.imshow(X[i])
        plt.show()
        #plt.title(int(labels[i]))
        plt.axis("off")


# In[54]:


history1 = model.fit(train, epochs = 1)


# In[58]:


rain800_test = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\rain800_idcgan\\test',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 512),
    shuffle=False,
    interpolation='bilinear')


# In[72]:


did_test = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\DID-MDN-datasets\\test',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 512),
    shuffle=True,
    interpolation='bilinear')


# In[170]:


did_test_2 = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\DID-MDN-datasets\\testing_fu',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 512),
    shuffle=True,
    interpolation='bilinear')


# In[ ]:


did_test = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\DID-MDN-datasets\\test',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 512),
    shuffle=True,
    interpolation='bilinear')


# In[191]:


rain1400_t_X = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\rainy_image_dataset\\training\\rainy_image',
    #labels='inferred',
    label_mode=None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    interpolation='bilinear')


# In[192]:


rain1400_t_Y = keras.preprocessing.image_dataset_from_directory(
    'D:\\.Files\\AI\\data\\rainy_image_dataset\\training\\ground_truth2',
    #labels='inferred',
    label_mode= None,
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    interpolation='bilinear')


# In[75]:


did_splited_t = did_test.map(split_x_y)


# In[171]:


did_test_2_s = did_test_2.map(split_x_y)


# In[76]:


rain800_splited_t = rain800_test.map(split_x_y)


# In[193]:


rain1400_t =  tf.data.Dataset.zip((rain1400_t_X.map(lambda x: x / 255.), rain1400_t_Y.map(lambda x: x / 255.)))


# In[84]:


test = did_splited_t.concatenate(rain800_splited_t)#.concatenate(rain1400_t)


# In[89]:


train_t = did_train_splitted.concatenate(rain800_splited)#.concatenate(rain1400_t)


# In[95]:


def split_x(x):
    return x[:, :, :256, :] / 255.

def split_y(x):
    return x[:, :, 256:, :] / 255.


# In[96]:


rain800_x = rain800_test.map(split_x)
rain800_y = rain800_test.map(split_y)
did_x = did_test.map(split_x)
did_y = did_test.map(split_y)


# In[97]:


test_x = did_x.concatenate(rain800_x)
test_y = did_y.concatenate(rain800_y)


# In[90]:


model.evaluate(train_t)


# In[86]:


model.evaluate(test)


# In[166]:


a = []
for X, Y in rain800_splited_t:
    #pred= model.predict(X)
    for p in range(pred.shape[0]):
        a.append(tf.image.ssim(X[p], Y[p], 1))
print(a)


# In[ ]:


tf.reduce_mean(tf.stack(a))


# In[180]:


a = []
for X, Y in did_test_2_s:
    pred= model.predict(X)
    for p in range(pred.shape[0]):
        a.append(tf.image.ssim(pred[p], Y[p], 1))
print(a)


# In[173]:


tf.reduce_mean(tf.stack(a))


# rain1400 - 0.83481604
# did_mdn - 0.83768666
# did_test_2 - 0.58992624
# rain800 - 0.6411672

# In[8]:


for X, Y in  did_train_splitted:
    #a = model.predict(X)
    plt.figure(figsize=(25, 25))
    plt.imshow(model.predict(X)[30])
    plt.show()
    plt.figure(figsize=(25, 25))
    plt.imshow(model.predict(Y)[30])
    plt.show()
    plt.figure(figsize=(25, 25))
    plt.imshow(X[30])
    plt.figure(figsize=(25, 25))
    plt.show()
    plt.figure(figsize=(25, 25))
    plt.imshow(Y[30])
    plt.show()
    break


# In[175]:


tf.keras.utils.plot_model(model)


# In[ ]:





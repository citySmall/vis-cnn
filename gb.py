import PIL.Image
from matplotlib import pylab as plt
import numpy as np
import keras.backend as K
from keras.layers import Input
from models.vgg16 import VGG16
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from saliency import GradientSaliency
from integrated_gradients import IntegratedGradients
from visual_backprop import VisualBackprop
from guided_backprop import GuidedBackprop
from utils import *

MEAN_TEST = np.load('./mean/mean_224_binary_all.npy')

# model define
NB_CLASSES = 5
model = VGG16(include_top=True, weights=None, input_tensor=Input(shape=(224, 224, 1)), classes=NB_CLASSES)
model.load_weights('./weights/vgg16_sgd_lr0001_last5layer_0831_patient_benign_malignant-' + str(7) + '.h5')
sgd = SGD(lr=0.0001, decay=0.0002, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# load image
image = smooth_50center_crop('./examples/1_SHR_C91S1M1_OD_U-3D_13x13_R02_518.jpg',is_color=False).astype("float")
img = image-MEAN_TEST
x = np.expand_dims(img, axis=0)

#predict
preds = model.predict(x)
label = np.argmax(preds)
print("predicted label:	{}".format(label))


#guided backpropagation
show_image(np.squeeze(image), ax=plt.subplot('131'), title='raw grey image')
guided_bprop = GuidedBackprop(model)
mask = guided_bprop.get_mask(img)
mask = np.squeeze(mask)
print(mask.shape)
mask -= np.min(mask)
mask /= mask.max()
mask *= 255
show_image(mask, ax=plt.subplot('132'), title='guided backprop', grayscale = False)
mask = guided_bprop.get_smoothed_mask(x[0])
mask = np.squeeze(mask)
mask -= np.min(mask)
mask /= mask.max()
mask *= 255
show_image(mask, ax=plt.subplot('133'), title='smoothed guided backprop', grayscale = False)

plt.show()
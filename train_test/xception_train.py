import numpy as np
import glob
import os
from keras.models import load_model,Sequential
from keras.layers import advanced_activations
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten, BatchNormalization,Conv2D
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import inception_resnet_v2
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.initializers import Orthogonal
from keras.preprocessing.image import  ImageDataGenerator
from keras.utils import to_categorical
from generators import DataGenerator
from scipy.misc import imresize
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import multi_gpu_model
import time
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
img_name = 'LIV057003 STZ+HFD #123'
img_size = (72,72,3)
model_name = '54-57003-'
base = Xception(weights="imagenet", include_top=False, pooling="max",input_shape=img_size)
# base = MobileNetV2(weights="imagenet", include_top=False, input_shape=img_size)
# base = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=img_size)
# base = load_model('/cptjack/totem/barrylee/codes/Xception_train_3_2.hdf5')
# base = NASNetLarge(weights="imagenet",include_top=False,input_shape=img_size)
# base = load_model('/cptjack/totem/yatong/imagenet/NASNetLarge(54).h5')
top_model = Sequential()
top_model.add(base)
# top_model.add(MaxPooling2D(pool_size=(2, 2)))#,strides=2,padding="same"
# top_model.add(Flatten())
top_model.add(Dropout(0.5))
top_model.add(Dense(96,name="dense",kernel_initializer=Orthogonal()))
top_model.add(advanced_activations.PReLU(alpha_initializer='zeros'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='softmax', kernel_initializer=Orthogonal()))
# parallel_model = multi_gpu_model(top_model, 2)
top_model.summary()
# top_model.load_weights('./Xception_decay.hdf5')

for layer in top_model.layers:
    layer.trainable = True

LearningRate = 0.001
decay = 0.0001
n_epochs = 30
sgd = optimizers.SGD(lr=LearningRate, decay=LearningRate / n_epochs, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=LearningRate,decay=LearningRate/n_epochs,amsgrad=True)
top_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.trainable_weights)]))

non_trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.non_trainable_weights)]))

print("\nModel Status")
print("=" * 40)
print("Total Parameters: {:,}".format((trainable_params + non_trainable_params)))
print("Non-Trainable Parameters: {:,}".format(non_trainable_params))
print("Trainable Parameters: {:,}\n".format(trainable_params))
train_path = '/cptjack/totem/barrylee/cell_raw_imgs/cell_tri-classification/train54-'+img_name
train_folders = [train_path+"/hepatocyte/",train_path+"/immune_cells/",train_path+"/other_cells/"]#,"/cptjack/totem/barrylee/cut_small_cell/train/other_cells/"
#,train_path+"/other_cells/"
population_sizes = []

print("\nImages for Training:")
print("=" * 40)
def imagenet_processing(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        image[:,:,i] -= mean[i]
        image[:,:,i] /= std[i]
    return image
for folder in train_folders:
    files = glob.glob(folder + "*.png")
    n = len(files)
    print("Class: %s. " % (folder.split("/")[-2]), "Size: {:,}".format(n))
    population_sizes.append(n)

MAX = max(population_sizes)

train_images = []
train_labels = []
val_path = '/cptjack/totem/barrylee/cut_small_cell/cell_tri-classification/val54-'+img_name
for index, folder in enumerate(train_folders):
    files = glob.glob(folder + "*.png")
    # sample = list(np.random.choice(files, MAX))
    images = io.imread_collection(files)
    # images = io.imread_collection(sample)
    images = [np.array(Image.fromarray(image).resize((img_size[0],img_size[1]))) for image in images]  # .resize((72,72))
    # images = [imresize(image, (139, 139)) for image in images] ### Reshape to (299, 299, 3) ###
    labels = [index] * len(images)
    train_images = train_images + images
    train_labels = train_labels + labels

train_images = np.stack(train_images)
train_images = (train_images / 255).astype(np.float32)  ### Standardise into the interval [0, 1] ###

train_labels = np.array(train_labels).astype(np.int32)
Y_train = to_categorical(train_labels, num_classes=np.unique(train_labels).shape[0])

valid_folders = [val_path+"/hepatocyte/",val_path+"/immune_cells/",val_path+"/other_cells/"]#,"/cptjack/totem/barrylee/cut_small_cell/val/other_cells/"
#,val_path+'/other_cells/'
print("\nImages for Validation")
print("=" * 40)

valid_images = []
valid_labels = []

for index, folder in enumerate(valid_folders):
    files = glob.glob(folder + "*.png")
    images = io.imread_collection(files)
    # images = files
    images = [np.array(Image.fromarray(image).resize((img_size[0],img_size[1]))) for image in images]
    # images = [imresize(image, (139, 139)) for image in images] ### Reshape to (299, 299, 3) ###
    labels = [index] * len(images)
    valid_images = valid_images + images
    valid_labels = valid_labels + labels
    print("Class: %s. Size: %d" % (folder.split("/")[-2], len(images)))

valid_images = np.stack(valid_images)
valid_images = (valid_images / 255).astype(np.float32)  ### Standardise

valid_labels = np.array(valid_labels).astype(np.int32)
Y_valid = to_categorical(valid_labels, num_classes=np.unique(valid_labels).shape[0])

print("\nBootstrapping to Balance - Training set size: %d (%d X %d)" % (
train_labels.shape[0], MAX, np.unique(train_labels).shape[0]))
print("=" * 40, "\n")
batch_size_for_generators =128
train_datagen = DataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True, shear_range=0.6,
                              stain_transformation=True)

train_gen = train_datagen.flow(train_images, Y_train, batch_size=batch_size_for_generators)

# VALIDATION

valid_datagen = DataGenerator()

valid_gen = valid_datagen.flow(valid_images, Y_valid, batch_size=batch_size_for_generators)
start = time.time()


class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk, self).__init__(filepath, monitor, save_best_only, mode)

    def set_model(self, model):
        super(Mycbk, self).set_model(self.single_model)


def get_callbacks(filepath, model, patience=8):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, './'+model_name+'/' + filepath)
    file_dir = './'+model_name+'/log/' + time.strftime('%Y_%m_%d', time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2, verbose=0, mode='min', min_delta=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./'+model_name+'/' + time.strftime('%Y_%m_%d', time.localtime(time.time())) + img_name +'-128-0.001'+'_log.csv',
                       separator=',', append=True)
    return [es, msave, reduce_lr, tb_log, log_cv]
import time
file_path = "xception-57003-.h5"
callbacks_s = get_callbacks(file_path, top_model, patience=10)

train_steps = train_images.shape[0] // batch_size_for_generators

valid_steps = valid_images.shape[0] // batch_size_for_generators

start_time = time.time()
# save_name = 'chongxin'+model_name+'_train_48-2'+img_name+'-54.hdf5'
# top_model.save(save_name)
# top_model.save_weights('x-3.h5')
top_model.fit_generator(generator=train_gen, epochs=n_epochs, steps_per_epoch=train_steps, validation_data=valid_gen,
                        validation_steps=valid_steps, callbacks=callbacks_s, verbose=1)
fin_time = time.time()-start_time
print(fin_time)

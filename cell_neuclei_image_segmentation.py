# %%
import os
import cv2
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
from tensorflow import keras
from keras import Model
from keras.utils import plot_model
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Layer, RandomZoom, Concatenate, Conv2DTranspose, Input
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow_examples.models.pix2pix import pix2pix

# %% 1. Data loading
DATASET_PATH = os.path.join(os.getcwd(), 'datasets', 'data-science-bowl-2018-2')

train_inputs_path = glob.glob(os.path.join(DATASET_PATH, 'train', 'inputs','*.png'))
train_masks_path = glob.glob(os.path.join(DATASET_PATH, 'train', 'masks','*.png'))
test_inputs_path = glob.glob(os.path.join(DATASET_PATH, 'test', 'inputs','*.png'))
test_masks_path = glob.glob(os.path.join(DATASET_PATH, 'test', 'masks','*.png'))

# %% 2. Data cleaning and visualization
IMAGE_SIZE = (128, 128)

train_inputs = [cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), IMAGE_SIZE) for image in train_inputs_path]
train_masks = [cv2.resize(cv2.imread(mask, cv2.IMREAD_GRAYSCALE), IMAGE_SIZE) for mask in train_masks_path]
test_inputs = [cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), IMAGE_SIZE) for image in test_inputs_path]
test_masks = [cv2.resize(cv2.imread(mask, cv2.IMREAD_GRAYSCALE), IMAGE_SIZE) for mask in test_masks_path]

train_inputs = np.array(train_inputs)
train_masks = np.array(train_masks)
test_inputs = np.array(test_inputs)
test_masks = np.array(test_masks)

# Visualize an input and its mask to make sure image is loaded properly
def display(image_list):
    plt.figure(figsize=(10,10))
    title = ['Input image', 'True mask', 'Predicted mask']

    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i+1)
        plt.title(title[i])
        plt.axis(False)
        plt.imshow(image_list[i])
    plt.show()

display([train_inputs[0], train_masks[0]])

# %% 3. Data preparation
# Expand the dimension of masks
train_masks_exp = np.expand_dims(train_masks, -1)
test_masks_exp = np.expand_dims(test_masks, -1)

# Normalize inputs and masks
train_inputs_norm = train_inputs / 255.0
test_inputs_norm = test_inputs / 255.0

train_masks_norm = (train_masks_exp > 128) * 1
test_masks_norm = (test_masks_exp > 128) * 1

# Do train-val split
SEED = 12345
X_train, X_val, y_train, y_val = train_test_split(train_inputs_norm, train_masks_norm, random_state=SEED)

# Convert numpy array to tensor
train_inputs_tensor = tf.data.Dataset.from_tensor_slices(X_train)
train_masks_tensor = tf.data.Dataset.from_tensor_slices(y_train)
validation_inputs_tensor = tf.data.Dataset.from_tensor_slices(X_val)
validation_masks_tensor = tf.data.Dataset.from_tensor_slices(y_val)
test_inputs_tensor = tf.data.Dataset.from_tensor_slices(test_inputs_norm)
test_masks_tensor = tf.data.Dataset.from_tensor_slices(test_masks_norm)

# Combine images and maks into zip dataset
train = tf.data.Dataset.zip((train_inputs_tensor, train_masks_tensor))
val = tf.data.Dataset.zip((validation_inputs_tensor, validation_masks_tensor))
test = tf.data.Dataset.zip((test_inputs_tensor, test_masks_tensor))

# Define an augmentation layer using class
class ImageAugmentation(Layer):
    def __init__(self, seed=SEED):
        super().__init__()
        self.augment_images = RandomZoom(0.2, seed=seed)
        self.augment_masks = RandomZoom(0.2, seed=seed)

    def call(self, images, masks):
        image = self.augment_images(images)
        mask = self.augment_masks(masks)
        return image, mask

# Convert zip dataset to PrefetchDataset
TRAIN_SIZE= len(train)
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.AUTOTUNE

train_pf =  (
    train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(ImageAugmentation())
    .prefetch(buffer_size=AUTOTUNE)
)

val_pf = val.batch(BATCH_SIZE)
test_pf = test.batch(BATCH_SIZE)

# Visualize the image in the PrefetchDataset
for images,masks in train_pf.take(1):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

# %% Model develoment
INPUT_SHAPE = list(IMAGE_SIZE) + [3,]
OUTPUT_CLASSES = 2

# Define pretrained model
base_model = keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)

# Outputs from activation layers for feature extraction
extractor_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu', 
    'block_16_project'
]

base_model_outputs = [base_model.get_layer(name).output for name in extractor_names]

# Feature extractor
down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Upsampling path
up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3)
]

# Build the U-Net model
# Downsampling layers
inputs = Input(shape=INPUT_SHAPE)
skips = down_stack(inputs)
x = skips[-1]
skips = reversed(skips[:-1])

# Upsampling layers
for up,skip in zip(up_stack, skips):
    x = up(x)
    x = Concatenate()([x,skip])

# Transpose conv layer
outputs = Conv2DTranspose(filters=OUTPUT_CLASSES, kernel_size=3, strides=2, padding='same')(x)

# Instantiate the model
model = Model(inputs=inputs, outputs=outputs)

# Model summary
if 'resources' not in os.listdir():
    os.mkdir('resources')

model.summary()
plot_model(model, to_file=os.path.join(os.getcwd(), 'resources', 'model.png'), show_layer_names=True, show_shapes=True)

# Model compilation
iou = keras.metrics.IoU(num_classes=2, target_class_ids=[1], sparse_y_pred=False)
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['acc', iou])

# Function for displaying model prediction
def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(tf.expand_dims(sample_image, axis=0)))])

# Create a callback function with the show_predictions function
class ShowPredictionCallback(Callback):
    def on_epoch_end(self,epoch,logs=None):
        # clear_output(wait=True)
        show_predictions()
        print('Sample predictions after epoch {}\n'.format(epoch+1))
        
LOG_DIR = os.path.join(os.getcwd(), "logs", datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=LOG_DIR)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.001)
es = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
mc = ModelCheckpoint(filepath=os.path.join(os.getcwd(), 'temp', 'checkpoint'), monitor='val_acc', save_best_only=True)

# %%
# Model training
EPOCHS = 20
VAL_SUBSPLITS = 2
VALIDATION_STEPS = len(val) // BATCH_SIZE // VAL_SUBSPLITS

model_history = model.fit(
    train_pf, 
    validation_data=val_pf, 
    validation_steps=VALIDATION_STEPS, 
    epochs=EPOCHS, 
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[ShowPredictionCallback(), tb, reduce_lr, es]
)

# %% Model evaluation
# Doing predictions with the model
show_predictions(test_pf, 5)

# Evaluating the model with test data
eval = model.evaluate(test_pf)

print(f'Prediction loss: {eval[0]:.2f}\nPrediction accuracy: {eval[1]:.2f}\nIoU: {eval[2]:.2f}')

# %% Model saving
model.save('model.h5')

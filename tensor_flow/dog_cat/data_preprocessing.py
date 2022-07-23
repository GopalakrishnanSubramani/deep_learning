from keras.preprocessing.image import ImageDataGenerator


train_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/train'
validation_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/validation'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'       
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'       
)


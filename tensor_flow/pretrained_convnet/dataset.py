import os, shutil

original_dataset_dir = '/home/krish/Documents/PyTorch/dogs-vs-cats/dataset'

base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'

train_cats_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/train/cats'
validation_cats_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/validation/cats'
test_cats_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/test/cats'
train_dogs_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/train/dogs'
validation_dogs_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/validation/dogs'
test_dogs_dir = '/home/krish/Documents/TF/dog_cat/cats_and_dogs_small/test/dogs'

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname os.path.join(original_dataset_dir, fname)
    dst in fnames:
    src = = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst) 

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
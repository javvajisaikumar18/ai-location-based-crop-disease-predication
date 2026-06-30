"""
Training script for plant disease detection using transfer learning.
Saves model to backend/model.h5 with classes order:
['Leaf Spot','Rust','Healthy','Powdery Mildew']

Usage:
    python train.py --data_dir ../data --epochs 10 --batch_size 32

Prepare data directory with structure:
  data/train/<class_name>/*.jpg
  data/val/<class_name>/*.jpg

This script uses MobileNetV2 and fine-tunes the top layers.
"""
import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

CLASS_NAMES = ['Leaf Spot','Rust','Healthy','Powdery Mildew']


def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    # freeze base
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base


def make_generators(data_dir, batch_size):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        train_dir, target_size=(224,224), batch_size=batch_size, classes=CLASS_NAMES, class_mode='categorical'
    )
    val_gen = val_aug.flow_from_directory(
        val_dir, target_size=(224,224), batch_size=batch_size, classes=CLASS_NAMES, class_mode='categorical'
    )
    return train_gen, val_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', default='model.h5', help='Output model path (saved in backend/)')
    args = parser.parse_args()

    train_gen, val_gen = make_generators(args.data_dir, args.batch_size)
    model, base = build_model(num_classes=len(CLASS_NAMES))

    # callbacks
    os.makedirs('.', exist_ok=True)
    checkpoint = ModelCheckpoint(args.output, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

    # initial training
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[checkpoint, early]
    )

    # fine-tune: unfreeze some top layers
    for layer in base.layers[-40:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=[checkpoint, early])

    # final save
    model.save(args.output)
    print('Saved model to', args.output)


if __name__ == '__main__':
    main()

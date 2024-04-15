import yaml
from tensorflow import keras
import tensorflow as tf
from keras import layers, models, optimizers, callbacks

# some of pre defined augmentation in tensorflow
pre_defined_augs = {
    'Rescaling'      : lambda x : keras.layers.Rescaling(x),
    'RandomRotation' : lambda x : keras.layers.RandomRotation(x),
    'RandomZoom'     : lambda x : keras.layers.RandomZoom(x),
    'RandomFlip'     : lambda x : keras.layers.RandomFlip("horizontal_and_vertical"),
}

# pre defined models
pre_defined_models = {
    'ConvNeXtBase'     : lambda x : tf.keras.applications.ConvNeXtBase(include_top=False, input_shape=x, weights=None),
    'ConvNeXtLarge'    : lambda x : tf.keras.applications.ConvNeXtLarge(include_top=False, input_shape=x, weights=None),
    'ConvNeXtSmall'    : lambda x : tf.keras.applications.ConvNeXtSmall(include_top=False, input_shape=x, weights=None),
    'ConvNeXtTiny'     : lambda x : tf.keras.applications.ConvNeXtTiny(include_top=False, input_shape=x, weights=None),
    'ConvNeXtXLarge'   : lambda x : tf.keras.applications.ConvNeXtXLarge(include_top=False, input_shape=x, weights=None),

    'EfficientNetV2B0' : lambda x : tf.keras.applications.EfficientNetV2B0(include_top=False, input_shape=x, weights=None),
    'EfficientNetV2B1' : lambda x : tf.keras.applications.EfficientNetV2B1(include_top=False, input_shape=x, weights=None),
    'EfficientNetV2B2' : lambda x : tf.keras.applications.EfficientNetV2B2(include_top=False, input_shape=x, weights=None),
    'EfficientNetV2B3' : lambda x : tf.keras.applications.EfficientNetV2B3(include_top=False, input_shape=x, weights=None),
    'EfficientNetV2L'  : lambda x : tf.keras.applications.EfficientNetV2L(include_top=False, input_shape=x, weights=None),
    'EfficientNetV2M'  : lambda x : tf.keras.applications.EfficientNetV2M(include_top=False, input_shape=x, weights=None),
    'EfficientNetV2S'  : lambda x : tf.keras.applications.EfficientNetV2S(include_top=False, input_shape=x, weights=None),

}


# get sequential of augmentations in config file
def parse_config(config_path='./config/config.yaml'):
    # read config file
    with open('./config/config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # save all augmentations
    list_of_augmentation = []
    for aug in data['augs']:
        aug_name = list(aug.keys())[0]
        list_of_augmentation.append(pre_defined_augs[aug_name](aug[aug_name]))

    # data augmentation and normalization
    data_augmentation = tf.keras.Sequential(list_of_augmentation)

    # get model name
    model_name = data['model_name']

    return data_augmentation, model_name

# get model based on config file
def make_model(input_shape, num_classes, config_path):
    # get augmentation and model name
    data_augmentation, model_name = parse_config(config_path)

    # Load model without the top layer
    base_model = pre_defined_models[model_name](input_shape)
    base_model.trainable = True

    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
import os
import pickle

np.set_printoptions(threshold=np.nan)
img_dir = 'images/'
print("Loading model")
base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)


def extract_features(dir, img_name):
    print("Loading image %s" % img_name)
    img = image.load_img(dir + img_name, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print("Extracting image features")
    features = model.predict(x)

    return img_name, features[0]


def extract_all(dir):
    all_imgs = os.listdir(dir)
    all_embeddings = {}
    print(all_imgs)

    for img in all_imgs:
        img_id, embedding = extract_features(img_dir, img)
        all_embeddings[img_id] = embedding

    return all_embeddings


def save_embeddings(emb_dict):
    print("Saving\n", emb_dict)
    with open("emb_dict.p", "wb") as f:
        pickle.dump(emb_dict, f)


def load_embeddings(file_name):
    print("Opening %s" % file_name)
    with open(file_name, "rb") as f:
        emb_dict = pickle.load(f)

    return emb_dict


def extract_and_save_features(dir):
    embeddings = extract_all(dir)
    save_embeddings(embeddings)


# print(extract_features('images/', 'brown_bear.png'))

# print(extract_all(img_dir))

# extract_and_save_features(img_dir)

print(load_embeddings("emb_dict.p"))
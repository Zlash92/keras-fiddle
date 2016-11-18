from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
import os

img_dir = 'images/'
print("Loading model")
base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)


def extract_features(img_path):
    print("Loading image %s" % img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print("Predicting image")
    features = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Top 3:', decode_predictions(preds, top=3)[0])
    # id, label, prob = decode_predictions(features, top=1)[0][0]

    # print("Top prediction: %s, Prob: %f" % (label, prob))

    return features


def predict_all(dir):
    all_imgs = os.listdir(dir)
    results = []
    print(all_imgs)

    for img in all_imgs:
        results.append(extract_features(img_dir + img))

    return results

# print(predict_all(img_dir))
# print(predict_batch(paths))

print(extract_features('images/brown_bear.png'))

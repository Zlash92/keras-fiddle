from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')


def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    #print('Top 3:', decode_predictions(preds, top=3)[0])
    id, label, prob = decode_predictions(preds, top=1)[0][0]

    #print("Top prediction: %s, Prob: %f" % (label, prob))

    return [img_path, label, prob]

def predict_batch(list_of_paths):
    results = []
    for img_path in list_of_paths:
        results.append(predict(img_path))

    return results

paths = ['images/brown_bear.png', 'images/dog_beagle.png']
print(predict_batch(paths))

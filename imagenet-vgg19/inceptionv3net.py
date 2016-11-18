from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np

print("Loading model")
base_model = InceptionV3(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)


img_path = 'images/brown_bear.png'
# img_path = 'images/bb2.jpg'
print("Loading image %s" % img_path)
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print("Predicting image")
preds = model.predict(x)[0]
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('Top 3:', decode_predictions(preds, top=3)[0])
# id, label, prob = decode_predictions(preds, top=1)[0][0]


print(preds)

# print("Top prediction: %s, Prob: %f" % (label, prob))
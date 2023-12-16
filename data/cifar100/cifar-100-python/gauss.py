import pickle as pkl
import random
import numpy as np

data = pkl.load(open('train', 'rb'), encoding='latin1')

print(data['data'])
print(data['data'].shape)

image_list = []

for image in data['data']:
    image = image / 255
    # print(image)
    severity = random.choice(range(1,6))
    c = [0.04, 0.06, .08, .09, .10][severity - 1]
    image = image + np.random.normal(size=image.shape, scale=c)
    # print(image)
    image = np.clip(image, 0, 1) * 255
    # print(image)
    image = np.uint8(image)
    # print(image)
    image_list.append(image)

# new_image_array = np.array(image_list)
# data['data'] = new_image_array
# print(data['data'])
# pkl.dump(data, open('train', 'wb'))


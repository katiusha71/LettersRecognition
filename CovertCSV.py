import numpy
import pandas as pd
import matplotlib
import matplotlib.image


data = pd.read_csv('../data/A_Z Handwritten Data.csv')
data=data.sample(frac=1)
pd_cut_data = data.head(65000)
cut_data = pd_cut_data.values
data_images = cut_data[:, 1:]
data_labels = cut_data[:, 0]


matplotlib.image.imsave('letters_images.png', data_images)




def _dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

labels_one_hot = _dense_to_one_hot(data_labels, 26)
# print(labels_one_hot)


with open('labels_uint8', 'w') as fo:
    for k in range(len(labels_one_hot)):
        fo.write(str(labels_one_hot[k]))




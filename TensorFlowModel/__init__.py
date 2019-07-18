import logging

import azure.functions as func

import numpy
import tensorflow

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from PIL import Image
from azure.storage.blob import BlockBlobService, PublicAccess


# We keep model as global variable so we don't have to reload it in case of warm invocations
model = None

class CustomModel(Model):
  def __init__(self):
    super(CustomModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

def download_blob(container_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    block_blob_service = BlockBlobService(account_name='jehollanpy12', account_key='')

    block_blob_service.get_blob_to_path(container_name, source_blob_name, destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    global model
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Model load which only happens during cold starts
    if model is None:
        download_blob('google', 'fashion_mnist_weights.index', '/tmp/fashion_mnist_weights.index')
        download_blob('google', 'fashion_mnist_weights.data-00000-of-00001', '/tmp/fashion_mnist_weights.data-00000-of-00001')
        model = CustomModel()
        model.load_weights('/tmp/fashion_mnist_weights')
    
    download_blob('google', 'test.png', '/tmp/test.png')
    image = numpy.array(Image.open('/tmp/test.png'))
    input_np = (numpy.array(Image.open('/tmp/test.png'))/255)[numpy.newaxis,:,:,numpy.newaxis]
    predictions = model.call(input_np)
    logging.info(predictions)
    logging.info("Image is "+class_names[numpy.argmax(predictions)])
    
    return func.HttpResponse(class_names[numpy.argmax(predictions)])

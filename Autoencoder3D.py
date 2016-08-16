import os
import numpy as np
import tensorflow as tf

#Future : Denoising option

class BasicAutoencoder3D(object):
  """ 
    The super class of LinearAutoencoder3D and ConvAutoencoder3D.

    Methods :
      train - Train one step and return loss value.
      reconstruct - Return reconstructed input array.
      save - Save variables.
      load - Load variables.
      sub_conv - Construct lowrank ConvAutoencdoer3D based on
                its latent variable shape (z_shape).
      sub_linear - Construct lowrank LinearAutoencdoer3D based on
                its latent variable shape (z_shape).
  """
  def train(self, data):
    _, loss = self._sess.run([self._train, self._loss], {self._x:data})
    return loss

  def reconstruct(self, data):
    return self._sess.run(self._y, {self._x:data})

  def get_latent(self, data):
    return self._sess.run(self._z, {self._x:data})

  def save(self, filename):
    self._saver.save(self._sess, filename)
    
  def load(self, filename):
    self._saver.restore(self._sess, filename)

  def sub_conv(self, kernel, stride, 
              padding='SAME', activation=None):
    return ConvAutoencoder3D(
      self._sess, self.z_shape, kernel, stride, padding, activation)

  def sub_linear(self, z_dim, activation=None, keep_prob=None):
    return LinearAutoencoder3D(
      self._sess, self.z_shape, z_dim, activation, keep_prob)
  

class LinearAutoencoder3D(BasicAutoencoder3D):
  def __init__(self, sess, input_shape, z_dim, activation=None, keep_prob=None):
    """ 
    Args:
      sess - tf session.
      input_shape - n-d integer list.
      z-dim - An integer indicating the dimensionality of latent code.
      activation - Tensorflow activation function
      keep_prob - if None, Dropout is not used
                  <1.0 for training, ==1.0 for test
    """

    """ private """
    self._sess = sess
    self._x = tf.placeholder(tf.float32, input_shape)
    self._z_dim = z_dim

    # Future : Dropout
    
    """ public """
    self.input_shape = input_shape
    self.activation = activation
    self.keep_prob = keep_prob

    t_dim = np.prod(input_shape[1:])
    curr = tf.reshape(self._x, [input_shape[0], -1])
    
    """ Encoder """
    self.lin_W1 = tf.Variable(tf.random_normal([t_dim, z_dim]))
    self.lin_b1 = tf.Variable(tf.fill([z_dim], 0.01), name="lin_b1")
    curr = tf.nn.xw_plus_b(curr, self.lin_W1, self.lin_b1)
    curr = activation(curr)

    """ Latent code """
    self._z = curr
    self.z_shape = self._z.get_shape().as_list()

    """ Decoder """
    self.lin_W2 = tf.Variable(tf.random_normal([z_dim, t_dim]))
    self.lin_b2 = tf.Variable(tf.fill([t_dim], 0.01), name="lin_b1")
    curr = tf.nn.xw_plus_b(curr, self.lin_W2, self.lin_b2)
    curr = activation(curr)

    self._var_list = [self.lin_W1, self.lin_b1, self.lin_W2, self.lin_b2]

    # Future : self.init() in BasicAutoencoder3D
    self._y = tf.reshape(curr, input_shape)
    self._reconstruct_loss = tf.reduce_mean(tf.square(self._x - self._y))
    self._loss = self._reconstruct_loss # More
    self._train = tf.train.AdamOptimizer().minimize(self._loss)

    self._saver = tf.train.Saver(self._var_list)
    self._sess.run(tf.initialize_variables(self._var_list))

class ConvAutoencoder3D(BasicAutoencoder3D):
  def __init__(self, sess, input_shape, kernel, stride, 
               padding='SAME', activation=None, keep_prob=None):
    """ 
    Args:
      sess - tf session.
      input_shape : 5d list. (N x D x H x W x C)
      kernel - 5d list. (D x H x W x I x O)
      stride - 5d list. (N x D x H x W x C)
      padding - 'SAME', 'VALID' (default : 'SAME')
      activation - Tensorflow activation function
      keep_prob - if None, Dropout is not used
                  <1.0 for training, ==1.0 for test
    """

    """ private members """
    self._sess = sess
    self._kernel = kernel
    self._x = tf.placeholder(tf.float32, input_shape)

    """ public members """
    self.input_shape = input_shape
    self.stride = stride
    self.padding = padding
    self.activation = activation
    self.keep_prob = keep_prob

    # Future : Dropout

    """ Encoder """
    curr = self._x
    self.enc_W = tf.Variable(tf.random_normal(kernel), name="enc_W")
    curr = tf.nn.conv3d(curr, self.enc_W, strides=stride, padding=padding)
    b_shape = curr.get_shape().as_list()[1:]
    self.enc_b = tf.Variable(tf.fill(b_shape, 0.01), name="enc_b")
    curr += self.enc_b
    if activation: 
      curr = activation(curr)

    """ Latent code """
    self._z = curr
    self.z_shape = self._z.get_shape().as_list()

    """ Decoder """
    self.dec_W = tf.Variable(tf.random_normal(kernel),name="dec_W")
    curr = tf.nn.conv3d_transpose(
      curr, self.dec_W, output_shape=input_shape, 
      strides=stride, padding=padding)
    b_shape = curr.get_shape().as_list()[1:]
    self.dec_b = tf.Variable(tf.fill(b_shape, 0.01), name="dec_b")
    curr += self.dec_b
    if activation: 
      curr = activation(curr)

    self._var_list = [self.enc_W, self.enc_b, self.dec_W, self.dec_b]

    # Future : self.init(self._var_list) in BasicAutoencoder3D
    self._y = curr
    self._reconstruct_loss = tf.reduce_mean(tf.square(self._x - self._y))
    self._loss = self._reconstruct_loss # More
    self._train = tf.train.AdamOptimizer().minimize(self._loss)

    self._saver = tf.train.Saver(self._var_list)
    self._sess.run(tf.initialize_variables(self._var_list))

class StackedConvAutoencoder3D(object):
  def __init__(self, sess, ae_list, z_dim=None, activation=tf.nn.tanh): 
    """ 
    Args:
      sess - tf session.
      ae_list - a list of ConvAutoencoder3D objects.
    """
    self._sess = sess
    self._ae_list = ae_list
    self._x = tf.placeholder(tf.float32, self._ae_list[0].input_shape)

    curr = self._x

    """ Encoder """
    for ae in ae_list:
      if ae.__class__ is ConvAutoencoder3D:
        curr = tf.nn.conv3d(curr, ae.enc_W, 
          strides=ae.stride, padding=ae.padding)
        curr += ae.enc_b
        if ae.activation:
          curr = ae.activation(curr)
      elif ae.__class__ is LinearAutoencoder3D:
        curr = tf.reshape(curr, [ae.input_shape[0], -1])
        curr = tf.nn.xw_plus_b(curr, ae.lin_W1, ae.lin_b1)
        if ae.activation:
          curr = ae.activation(curr)

    """ Latent code """
    self._z = curr
    self.z_shape = self._z.get_shape().as_list()
    
    """ Decoder """
    for ae in ae_list[::-1]:
      if ae.__class__ is ConvAutoencoder3D:
        curr = tf.nn.conv3d_transpose(
          curr, ae.dec_W, output_shape=ae.input_shape, 
          strides=ae.stride, padding=ae.padding)
        curr += ae.dec_b
        if ae.activation:
          curr = ae.activation(curr)
      elif ae.__class__ is LinearAutoencoder3D:
        curr = tf.nn.xw_plus_b(curr, ae.lin_W2, ae.lin_b2)
        if ae.activation:
          curr = ae.activation(curr)
        curr = tf.reshape(curr, ae.input_shape)

    self._y = curr 

    self._reconstruct_loss = tf.reduce_mean(tf.square(self._x - self._y))
    self._loss = self._reconstruct_loss # More.
    self._train = tf.train.AdamOptimizer().minimize(self._loss)

    self._sess.run(tf.initialize_all_variables())
  
  def train_each_layer(self, data_generator, iteration=10000):
    if not os.path.isdir("ckpt/"): 
      os.makedirs("ckpt/")

    for l, ae in enumerate(self._ae_list):
      dir_path = "ckpt/ckpt_{}/".format(l+1)
      if not os.path.isdir(dir_path): 
        os.makedirs(dir_path)

      file_path = dir_path+"model.ckpt"
      if os.path.exists(file_path):
        ae.load(file_path) 
      else :
        for i in range(iteration):
          data = data_generator.gen()
          data = self._get_nth_latent(l, data)
          loss = ae.train(data)
          print "{} layer / {} iter / {}".format(l+1, i+1, loss)
        ae.save(file_path)

  def fine_tuning(self, dg, iteration=10000):
    for i in range(iteration):
      data = dg.gen()
      _, loss = self._sess.run([self._train, self._loss], {self._x:data})
      print "{} iter / {}".format(i+1, loss)
      
    self.save()

  def save(self):
    """ Save all AE variables """
    if not os.path.isdir("ckpt/"): 
      os.makedirs("ckpt/")
    for l, ae in enumerate(self._ae_list):
      dir_path = "ckpt/ckpt_{}/".format(l+1)
      if not os.path.isdir(dir_path): 
        os.makedirs(dir_path)
      file_path = dir_path+"model.ckpt"
      ae.save(file_path)

  def load(self):
    for l, ae in enumerate(self._ae_list):
      dir_path = "ckpt/ckpt_{}/".format(l+1)
      file_path = dir_path+"model.ckpt"
      if os.path.exists(file_path):
        ae.load(file_path)
      else : 
        raise IOError("No ckpt_{}/model.ckpt file".format(l+1))

  def reconstruct(self, data):
    return self._sess.run(self._y, {self._x:data})

  def _get_nth_latent(self, nth, data):
    if nth == 0: return data
    else :
      latent = data
      for l, ae in enumerate(self._ae_list):
        latent = ae.get_latent(latent)
        if l+1 == nth: return latent

class DataGenerator(object):
  def __init__(self, filename, data_shape, output_shape, method='random', slide_step=1):
    """ 
    Args :
      filename : the name of raw volume file
      data_shape : a 3-d list of integers. (D x H x W)
      output_shape : Output shape of data from the generator.
                    a 5-d list of integers. (N x D x H x W x 1)
      method : Sampling method. 'random' or 'slide'
    """

    with open(filename, 'rb') as f:
      self.volume = np.fromstring(f.read(), dtype=np.uint8)
    self.volume = self.volume.reshape(data_shape).astype(np.float32)

    # normalization from 0 to 1
    v_min, v_max = np.min(self.volume), np.max(self.volume)
    self.volume = (self.volume - v_min) / (v_max - v_min)

    self._output_shape = output_shape
    self._data_shape = data_shape
    self._method = method
    if self._method == 'slide':
      self._slide_d = 0
      self._slide_h = 0
      self._slide_w = 0
      self._slide_step = slide_step
  
  def gen(self):
    sample = np.empty(self.output_shape)

    if self._method == 'random':
      index = np.empty([3], dtype=np.int32)
      for i in range(self.output_shape[0]):
        index[0] = np.random.randint(
            self._data_shape[0]-self.output_shape[1]+1)
        index[1] = np.random.randint(
            self._data_shape[1]-self.output_shape[2]+1)
        index[2] = np.random.randint(
            self._data_shape[2]-self.output_shape[3]+1)
        sample[i,:,:,:,0] = self.volume[
            index[0]:index[0]+self.output_shape[1],
            index[1]:index[1]+self.output_shape[2],
            index[2]:index[2]+self.output_shape[3]]

    elif self._method == 'slide':
      for i in range(self.output_shape[0]):
        sample[i,:,:,:,0] = self.volume[
            self._slide_d:self._slide_d+self.output_shape[1],
            self._slide_h:self._slide_h+self.output_shape[2],
            self._slide_w:self._slide_w+self.output_shape[3]]
        self._slide_w += self._slide_step
        if self._slide_w + self.output_shape[3] >= self._data_shape[2]:
          self._slide_w = 0
          self._slide_h += self._slide_step
        if self._slide_h + self.output_shape[2] >= self._data_shape[1]:
          self._slide_h = 0
          self._slide_d += self._slide_step
        if self._slide_d + self.output_shape[1] >= self._data_shape[0]:
          self._slide_d = 0
          self._slide_h = 0
          self._slide_w = 0
    return sample

  def crop(self, indices):
    """ 
    Args :
      indices : a list of indices. 
                Each index is a 3-d tuple (d, h, w).
    """
    sample = np.empty(self.output_shape)
    for i, index in enumerate(indices):
      sample[i,:,:,:,0] = self.volume[
          index[0]:index[0]+self.output_shape[1],
          index[1]:index[1]+self.output_shape[2],
          index[2]:index[2]+self.output_shape[3]]
    return sample


  @property
  def output_shape(self):
    return self._output_shape
  @output_shape.setter
  def output_shape(self, shape):
    self._output_shape = shape
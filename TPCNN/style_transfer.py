import tensorflow as tf
import numpy as np 
import scipy.io  
import time                       
import cv2
import os
import matplotlib.pyplot as plt
'''
  parsing and configuration
'''  
def build_model(input_img):
  print('\nBUILDING VGG-19 NETWORK')
  net = {}
  _, h, w, d     = input_img.shape
  
  print('loading model weights...')
  vgg_rawnet     = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
  vgg_layers     = vgg_rawnet['layers'][0]
  print('constructing layers...')
  net['input']   = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

  print('LAYER GROUP 1')
  net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
  net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

  net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
  net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))
  
  net['pool1']   = pool_layer('pool1', net['relu1_2'])

  print('LAYER GROUP 2')  
  net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
  net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))
  
  net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
  net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))
  
  net['pool2']   = pool_layer('pool2', net['relu2_2'])
  
  print('LAYER GROUP 3')
  net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
  net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

  net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
  net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

  net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
  net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

  net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
  net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

  net['pool3']   = pool_layer('pool3', net['relu3_4'])

  print('LAYER GROUP 4')
  net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
  net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

  net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
  net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

  net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
  net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

  net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
  net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

  net['pool4']   = pool_layer('pool4', net['relu4_4'])

  print('LAYER GROUP 5')
  net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
  net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

  net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
  net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

  net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
  net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

  net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
  net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

  net['pool5']   = pool_layer('pool5', net['relu5_4'])

  return net

def conv_layer(layer_name, layer_input, W):
  conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
  print('--{} | shape={} | weights_shape={}'.format(layer_name, conv.get_shape(), W.get_shape()))
  return conv

def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), b.get_shape()))
    return relu

def pool_layer(layer_name, layer_input):
    pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
    return pool

def get_weights(vgg_layers, i):
  weights = vgg_layers[i][0][0][2][0][0]
  W = tf.constant(weights)
  return W

def get_bias(vgg_layers, i):
  bias = vgg_layers[i][0][0][2][0][1]
  b = tf.constant(np.reshape(bias, (bias.size)))
  return b


def write_image(path, img):
  img += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  # shape (1, h, w, d) to (h, w, d)
  img = img[0]
  img = np.clip(img, 0, 255).astype('uint8')
  # rgb to bgr
  img = img[...,::-1]
  cv2.imwrite(path, img)



def write_image_output(output_img, content_img, init_img):
  out_dir = os.path.join('C:/reaearch/neural-style-tf-master/image_output')
  img_path = os.path.join(out_dir, 'result.png')
  content_path = os.path.join(out_dir, 'content.png')
  init_path = os.path.join(out_dir, 'init.png')
  style_path = os.path.join(out_dir, 'style.png')
  write_image(img_path, output_img)
  write_image(content_path, content_img)
  write_image(init_path, init_img)
  write_image(style_path, content_img)
  
  
def get_conv_output(sess,layer_output):

    imgs_out = sess.run(layer_output)
    
    img_num,height,width, depth = imgs_out.shape
    
    for img_out in imgs_out:
        
        for channel in range(depth):
            
            channel_output =img_out[:, :,channel]
            
            channel_output = channel_output.astype(np.uint8)
            
            plt.imshow(channel_output,cmap='gray')
            
            plt.show()
  
  # save the configuration settings
#  out_file = os.path.join(out_dir, 'meta_data.txt')
#  f = open(out_file, 'w')
##  f.write('image_name: {}\n'.format(args.img_name))
##  f.write('content: {}\n'.format(args.content_img))
##  index = 0
##  for style_img, weight in zip(args.style_imgs, args.style_imgs_weights):
##    f.write('styles['+str(index)+']: {} * {}\n'.format(weight, style_img))
##    index += 1
##  index = 0
##  if args.style_mask_imgs is not None:
##    for mask in args.style_mask_imgs:
##      f.write('style_masks['+str(index)+']: {}\n'.format(mask))
##      index += 1
##  f.write('init_type: {}\n'.format(args.init_img_type))
##  f.write('content_weight: {}\n'.format(args.content_weight))
##  f.write('style_weight: {}\n'.format(args.style_weight))
##  f.write('tv_weight: {}\n'.format(args.tv_weight))
##  f.write('content_layers: {}\n'.format(args.content_layers))
##  f.write('style_layers: {}\n'.format(args.style_layers))
##  f.write('optimizer_type: {}\n'.format(args.optimizer))
##  f.write('max_iterations: {}\n'.format(args.max_iterations))
##  f.write('max_image_size: {}\n'.format(args.max_size))
#  f.close()

def convert_to_original_colors(content_img, stylized_img):
  content_img += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  # shape (1, h, w, d) to (h, w, d)
  content_img = content_img[0]
  content_img = np.clip(content_img, 0, 255).astype('uint8')
  # rgb to bgr
  content_img = content_img[...,::-1]
  stylized_img += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  # shape (1, h, w, d) to (h, w, d)
  stylized_img = stylized_img[0]
  stylized_img = np.clip(stylized_img, 0, 255).astype('uint8')
  # rgb to bgr
  stylized_img = stylized_img[...,::-1]
  cvt_type = cv2.COLOR_BGR2YUV
  inv_cvt_type = cv2.COLOR_YUV2BGR
  content_cvt = cv2.cvtColor(content_img, cvt_type)
  stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
  c1, _, _ = cv2.split(stylized_cvt)
  _, c2, c3 = cv2.split(content_cvt)
  merged = cv2.merge((c1, c2, c3))
  dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
  dst = dst[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
  dst = dst[np.newaxis,:,:,:]
  dst -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  return dst


path = os.path.join('C:/reaearch/neural-style-tf-master/image_input', 'tubingen.jpg')
   # bgr image
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = img.astype(np.float32)
h, w, d = img.shape
mx = 512
  # resize if > max size
if h > w and h > mx:
  w = (float(mx) / float(h)) * w
  img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
if w > mx:
  h = (float(mx) / float(w)) * h
  img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
      # bgr to rgb
content_img = img[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
content_img = content_img[np.newaxis,:,:,:]
content_img -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  
_, ch, cw, cd = content_img.shape
path = os.path.join('C:/reaearch/neural-style-tf-master/styles', 'hei3.jpg')
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = img.astype(np.float32)
img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    # bgr to rgb
style_imgs1 = img[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
style_imgs1 = style_imgs1[np.newaxis,:,:,:]
style_imgs1 -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

#_, ch, cw, cd = content_img.shape
#path = os.path.join('C:/reaearch/neural-style-tf-master/styles', 'kandinsky.jpg')
#img = cv2.imread(path, cv2.IMREAD_COLOR)
#img = img.astype(np.float32)
#img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
#    # bgr to rgb
#style_imgs2 = img[...,::-1]
#  # shape (h, w, d) to (1, h, w, d)
#style_imgs2 = style_imgs2[np.newaxis,:,:,:]
#style_imgs2 -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  
path = os.path.join('C:/reaearch/neural-style-tf-master/image_input', 'tubingen.jpg')
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = img.astype(np.float32)
img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    # bgr to rgb
bie_imgs = img[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
bie_imgs = bie_imgs[np.newaxis,:,:,:]
bie_imgs -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

with tf.Graph().as_default():
  print('\n---- RENDERING SINGLE IMAGE ----\n')
  init_img = content_img
  tick = time.time()
 
  with tf.device('/gpu:0'), tf.Session() as sess:
   # setup network
      net = build_model(content_img)
      L_style = 0.
      # Only one style image so the weight is 1
      weights = [0.5]
      style = [style_imgs1]
      for styleimg, img_weight in zip(style, weights):
          sess.run(net['input'].assign(styleimg))
          style_loss = 0.
          style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
          style_layer_weights = [0.05, 0.05, 0.2, 0.3, 0.4]
          for layer, weight in zip(style_layers, style_layer_weights):
              a = sess.run(net[layer])
              x = net[layer]
              a = tf.convert_to_tensor(a)
              _, h, w, d = a.get_shape()
              M = h.value * w.value
              N = d.value
              F = tf.reshape(a, (M, N))
              A = tf.matmul(tf.transpose(F), F)        
              F = tf.reshape(x, (M, N))
              G = tf.matmul(tf.transpose(F), F)        
              loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))
              style_loss += loss * weight
          style_loss /= float(len(style_layers))
          L_style += (style_loss * img_weight)
      L_style /= float(len(style))  
          
  # content loss
      sess.run(net['input'].assign(content_img))
      content_loss = 0.
      for layer, weight in zip(['conv4_2'], [1.0]):
          p = sess.run(net[layer])
          x = net[layer]
          p = tf.convert_to_tensor(p)
          _, h, w, d = p.get_shape()
          M = h.value * w.value
          N = d.value
          K = 1. / (2. * N**0.5 * M**0.5)/3 
          
          loss = K * tf.reduce_sum(tf.pow((x - p), 2))
          content_loss += loss * weight
          content_loss /= float(len(['conv4_2'])) 
      L_content = content_loss
    # denoising loss
      L_tv = tf.image.total_variation(net['input'])
    
    # loss weights
      alpha = 3
      beta  = 10000
      theta = 0.001
    
    # total loss
      L_total  = alpha * L_content
      L_total += beta  * L_style
      L_total += theta * L_tv

    # optimization algorithm
      print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
      optimizer = tf.contrib.opt.ScipyOptimizerInterface(L_total, method='L-BFGS-B',options={'maxiter': 1000,'disp': 50, 'epsilon': 1e-4})
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      sess.run(net['input'].assign(init_img))
      optimizer.minimize(sess)
      output_img = sess.run(net['input']) 
#      nihaohao = x.eval()
#      iterations = 0
#      while (iterations < 5):
#        sess.run(train_op)
#        curr_loss = L_total.eval()
#        print("At iterate {}\tf=  {}".format(iterations, curr_loss))
#        iterations += 1
  
      
#      output_img = convert_to_original_colors(np.copy(content_img), output_img)
      write_image_output(output_img, content_img, style_imgs1)
  tock = time.time()
  print('Single image elapsed time: {}'.format(tock - tick))

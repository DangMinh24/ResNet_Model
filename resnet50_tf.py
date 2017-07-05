import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras.api.keras.layers import Input,Dense,Convolution2D,MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import Sequential
inputs=np.random.ranf(size=(224,224,3))
x=tf.placeholder(shape=(None,224,224,3),dtype=tf.float32)
y=tf.placeholder(shape=(None,1000),dtype=tf.float32)

# Layer 1:
w1_c=tf.Variable(tf.random_normal(shape=[7,7,3,64],dtype=tf.float32))
o1=tf.nn.conv2d(x,w1_c,strides=[1,2,2,1],padding="SAME")
o1_act=tf.nn.relu(o1)
mp1=tf.nn.max_pool(o1_act,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")

# Layer 2:
#   *Block 1:
in2_b1=mp1
w2_s=tf.Variable(tf.random_normal(shape=[1,1,64,256],dtype=tf.float32))
shortcut2_b1=tf.nn.conv2d(in2_b1,w2_s,strides=[1,1,1,1],padding="SAME")
w2_b1_v1=tf.Variable(tf.random_normal(shape=[1,1,64,64],dtype=tf.float32))
o2_b1_v1=tf.nn.conv2d(mp1,w2_b1_v1,strides=[1,1,1,1],padding="SAME")
w2_b1_v2=tf.Variable(tf.random_normal(shape=[3,3,64,64],dtype=tf.float32))
o2_b1_v2=tf.nn.conv2d(o2_b1_v1,w2_b1_v2,strides=[1,1,1,1],padding="SAME")
w2_b1_v3=tf.Variable(tf.random_normal(shape=[1,1,64,256],dtype=tf.float32))
o2_b1_v3=tf.nn.conv2d(o2_b1_v2,w2_b1_v3,strides=[1,1,1,1],padding="SAME")
o2_b1=tf.nn.relu(shortcut2_b1+o2_b1_v3)

#   *Block 2:
in2_b2=o2_b1
shortcut2_b2=in2_b2
w2_b2_v1=tf.Variable(tf.random_normal(shape=[1,1,256,64],dtype=tf.float32))
o2_b2_v1=tf.nn.conv2d(o2_b1,w2_b2_v1,strides=[1,1,1,1],padding="SAME")
w2_b2_v2=tf.Variable(tf.random_normal(shape=[3,3,64,64],dtype=tf.float32))
o2_b2_v2=tf.nn.conv2d(o2_b2_v1,w2_b2_v2,strides=[1,1,1,1],padding="SAME")
w2_b2_v3=tf.Variable(tf.random_normal(shape=[1,1,64,256],dtype=tf.float32))
o2_b2_v3=tf.nn.conv2d(o2_b2_v2,w2_b2_v3,strides=[1,1,1,1],padding="SAME")
o2_b2=tf.nn.relu(shortcut2_b2+o2_b2_v3)

#   *Block3:
in2_b3=o2_b2
shortcut2_b3=tf.nn.max_pool(in2_b3,ksize=[1,1,1,1],strides=[1,2,2,1],padding="VALID")
w2_b3_v1=tf.Variable(tf.random_normal(shape=[1,1,256,64],dtype=tf.float32))
o2_b3_v1=tf.nn.conv2d(o2_b2,w2_b3_v1,strides=[1,1,1,1],padding="SAME")
w2_b3_v2=tf.Variable(tf.random_normal(shape=[3,3,64,64],dtype=tf.float32))
o2_b3_v2=tf.nn.conv2d(o2_b3_v1,w2_b3_v2,strides=[1,2,2,1],padding="SAME")
w2_b3_v3=tf.Variable(tf.random_normal(shape=[1,1,64,256],dtype=tf.float32))
o2_b3_v3=tf.nn.conv2d(o2_b3_v2,w2_b3_v3,strides=[1,1,1,1],padding="SAME")
o2_b3=tf.nn.relu(shortcut2_b3+o2_b3_v3)

# Layer 3:
#   *Block1:
in3_b1=o2_b3
w3_s=tf.Variable(tf.random_normal(shape=(1,1,256,512),dtype=tf.float32))
shortcut3_b1=tf.nn.conv2d(in3_b1,w3_s,strides=[1,1,1,1],padding="SAME")
w3_b1_v1=tf.Variable(tf.random_normal(shape=[1,1,256,128],dtype=tf.float32))
o3_b1_v1=tf.nn.conv2d(o2_b3,w3_b1_v1,strides=[1,1,1,1],padding="SAME")
w3_b1_v2=tf.Variable(tf.random_normal(shape=[3,3,128,128],dtype=tf.float32))
o3_b1_v2=tf.nn.conv2d(o3_b1_v1,w3_b1_v2,strides=[1,1,1,1],padding="SAME")
w3_b1_v3=tf.Variable(tf.random_normal(shape=[1,1,128,512],dtype=tf.float32))
o3_b1_v3=tf.nn.conv2d(o3_b1_v2,w3_b1_v3,strides=[1,1,1,1],padding="SAME")
o3_b1=tf.nn.relu(shortcut3_b1+o3_b1_v3)

#   *Block2:
in3_b2=o3_b1
shortcut3_b2=in3_b2
w3_b2_v1=tf.Variable(tf.random_normal(shape=[1,1,512,128],dtype=tf.float32))
o3_b2_v1=tf.nn.conv2d(o3_b1,w3_b2_v1,strides=[1,1,1,1],padding="SAME")
w3_b2_v2=tf.Variable(tf.random_normal(shape=[3,3,128,128],dtype=tf.float32))
o3_b2_v2=tf.nn.conv2d(o3_b2_v1,w3_b2_v2,strides=[1,1,1,1],padding="SAME")
w3_b2_v3=tf.Variable(tf.random_normal(shape=[1,1,128,512],dtype=tf.float32))
o3_b2_v3=tf.nn.conv2d(o3_b2_v2,w3_b2_v3,strides=[1,1,1,1],padding="SAME")
o3_b2=tf.nn.relu(shortcut3_b2+o3_b2_v3)

#   *Block3:
in3_b3=o3_b2
shortcut3_b3=in3_b3
w3_b3_v1=tf.Variable(tf.random_normal(shape=[1,1,512,128],dtype=tf.float32))
o3_b3_v1=tf.nn.conv2d(o3_b2,w3_b3_v1,strides=[1,1,1,1],padding="SAME")
w3_b3_v2=tf.Variable(tf.random_normal(shape=[3,3,128,128],dtype=tf.float32))
o3_b3_v2=tf.nn.conv2d(o3_b3_v1,w3_b3_v2,strides=[1,1,1,1],padding="SAME")
w3_b3_v3=tf.Variable(tf.random_normal(shape=[1,1,128,512],dtype=tf.float32))
o3_b3_v3=tf.nn.conv2d(o3_b3_v2,w3_b3_v3,strides=[1,1,1,1],padding="SAME")
o3_b3=tf.nn.relu(shortcut3_b3+o3_b3_v3)

#   *Block4:
in3_b4=o3_b3
shortcut3_b4=tf.nn.max_pool(in3_b4,ksize=[1,1,1,1],strides=[1,2,2,1],padding="VALID")
w3_b4_v1=tf.Variable(tf.random_normal(shape=[1,1,512,128],dtype=tf.float32))
o3_b4_v1=tf.nn.conv2d(o3_b3,w3_b4_v1,strides=[1,1,1,1],padding="SAME")
w3_b4_v2=tf.Variable(tf.random_normal(shape=[3,3,128,128],dtype=tf.float32))
o3_b4_v2=tf.nn.conv2d(o3_b4_v1,w3_b4_v2,strides=[1,2,2,1],padding="SAME")
w3_b4_v3=tf.Variable(tf.random_normal(shape=[1,1,128,512],dtype=tf.float32))
o3_b4_v3=tf.nn.conv2d(o3_b4_v2,w3_b4_v3,strides=[1,1,1,1],padding="SAME")
o3_b4=tf.nn.relu(shortcut3_b4+o3_b4_v3)

# Layer 4:
#   *Block1:
in4_b1=o3_b4
w4_s=tf.Variable(tf.random_normal(shape=[1,1,512,1024],dtype=tf.float32))
shortcut4_b1=tf.nn.conv2d(in4_b1,w4_s,strides=[1,1,1,1],padding="SAME")
w4_b1_v1=tf.Variable(tf.random_normal(shape=[1,1,512,256],dtype=tf.float32))
o4_b1_v1=tf.nn.conv2d(o3_b4,w4_b1_v1,strides=[1,1,1,1],padding="SAME")
w4_b1_v2=tf.Variable(tf.random_normal(shape=[3,3,256,256],dtype=tf.float32))
o4_b1_v2=tf.nn.conv2d(o4_b1_v1,w4_b1_v2,strides=[1,1,1,1],padding="SAME")
w4_b1_v3=tf.Variable(tf.random_normal(shape=[1,1,256,1024],dtype=tf.float32))
o4_b1_v3=tf.nn.conv2d(o4_b1_v2,w4_b1_v3,strides=[1,1,1,1],padding="SAME")
o4_b1=tf.nn.relu(shortcut4_b1+o4_b1_v3)

#   *Block2:
in4_b2=o4_b1
shortcut4_b2=in4_b2
w4_b2_v1=tf.Variable(tf.random_normal(shape=[1,1,1024,256],dtype=tf.float32))
o4_b2_v1=tf.nn.conv2d(o4_b1,w4_b2_v1,strides=[1,1,1,1],padding="SAME")
w4_b2_v2=tf.Variable(tf.random_normal(shape=[3,3,256,256],dtype=tf.float32))
o4_b2_v2=tf.nn.conv2d(o4_b2_v1,w4_b2_v2,strides=[1,1,1,1],padding="SAME")
w4_b2_v3=tf.Variable(tf.random_normal(shape=[1,1,256,1024],dtype=tf.float32))
o4_b2_v3=tf.nn.conv2d(o4_b2_v2,w4_b2_v3,strides=[1,1,1,1],padding="SAME")
o4_b2=tf.nn.relu(shortcut4_b2+o4_b2_v3)

#   *Block3:
in4_b3=o4_b2
shortcut4_b3=in4_b3
w4_b3_v1=tf.Variable(tf.random_normal(shape=[1,1,1024,256],dtype=tf.float32))
o4_b3_v1=tf.nn.conv2d(o4_b2,w4_b3_v1,strides=[1,1,1,1],padding="SAME")
w4_b3_v2=tf.Variable(tf.random_normal(shape=[3,3,256,256],dtype=tf.float32))
o4_b3_v2=tf.nn.conv2d(o4_b3_v1,w4_b3_v2,strides=[1,1,1,1],padding="SAME")
w4_b3_v3=tf.Variable(tf.random_normal(shape=[1,1,256,1024],dtype=tf.float32))
o4_b3_v3=tf.nn.conv2d(o4_b3_v2,w4_b3_v3,strides=[1,1,1,1],padding="SAME")
o4_b3=tf.nn.relu(shortcut4_b3+o4_b3_v3)

#   *Block4:
in4_b4=o4_b3
shortcut4_b4=in4_b4
w4_b4_v1=tf.Variable(tf.random_normal(shape=[1,1,1024,256],dtype=tf.float32))
o4_b4_v1=tf.nn.conv2d(o4_b3,w4_b4_v1,strides=[1,1,1,1],padding="SAME")
w4_b4_v2=tf.Variable(tf.random_normal(shape=[3,3,256,256],dtype=tf.float32))
o4_b4_v2=tf.nn.conv2d(o4_b4_v1,w4_b4_v2,strides=[1,1,1,1],padding="SAME")
w4_b4_v3=tf.Variable(tf.random_normal(shape=[1,1,256,1024],dtype=tf.float32))
o4_b4_v3=tf.nn.conv2d(o4_b4_v2,w4_b4_v3,strides=[1,1,1,1],padding="SAME")
o4_b4=tf.nn.relu(shortcut4_b4+o4_b4_v3)

#   *Block5:
in4_b5=o4_b4
shortcut4_b5=in4_b5
w4_b5_v1=tf.Variable(tf.random_normal(shape=[1,1,1024,256],dtype=tf.float32))
o4_b5_v1=tf.nn.conv2d(o4_b4,w4_b5_v1,strides=[1,1,1,1],padding="SAME")
w4_b5_v2=tf.Variable(tf.random_normal(shape=[3,3,256,256],dtype=tf.float32))
o4_b5_v2=tf.nn.conv2d(o4_b5_v1,w4_b5_v2,strides=[1,1,1,1],padding="SAME")
w4_b5_v3=tf.Variable(tf.random_normal(shape=[1,1,256,1024],dtype=tf.float32))
o4_b5_v3=tf.nn.conv2d(o4_b5_v2,w4_b5_v3,strides=[1,1,1,1],padding="SAME")
o4_b5=tf.nn.relu(shortcut4_b5+o4_b5_v3)


#   *Block6:
in4_b6=o4_b5
shortcut4_b6=tf.nn.max_pool(in4_b6,ksize=[1,1,1,1],strides=[1,2,2,1],padding="VALID")
w4_b6_v1=tf.Variable(tf.random_normal(shape=[1,1,1024,256],dtype=tf.float32))
o4_b6_v1=tf.nn.conv2d(o4_b5,w4_b6_v1,strides=[1,1,1,1],padding="SAME")
w4_b6_v2=tf.Variable(tf.random_normal(shape=[3,3,256,256],dtype=tf.float32))
o4_b6_v2=tf.nn.conv2d(o4_b6_v1,w4_b6_v2,strides=[1,2,2,1],padding="SAME")
w4_b6_v3=tf.Variable(tf.random_normal(shape=[1,1,256,1024],dtype=tf.float32))
o4_b6_v3=tf.nn.conv2d(o4_b6_v2,w4_b6_v3,strides=[1,1,1,1],padding="SAME")
o4_b6=tf.nn.relu(shortcut4_b6+o4_b6_v3)

# Layer 5:
#   *Block1:
in5_b1=o4_b6
w5_s=tf.Variable(tf.random_normal(shape=[1,1,1024,2048],dtype=tf.float32))
shortcut5_b1=tf.nn.conv2d(in5_b1,w5_s,strides=[1,1,1,1],padding="SAME")
w5_b1_v1=tf.Variable(tf.random_normal(shape=[1,1,1024,512],dtype=tf.float32))
o5_b1_v1=tf.nn.conv2d(o4_b6,w5_b1_v1,strides=[1,1,1,1],padding="SAME")
w5_b1_v2=tf.Variable(tf.random_normal(shape=[3,3,512,512],dtype=tf.float32))
o5_b1_v2=tf.nn.conv2d(o5_b1_v1,w5_b1_v2,strides=[1,1,1,1],padding="SAME")
w5_b1_v3=tf.Variable(tf.random_normal(shape=[1,1,512,2048],dtype=tf.float32))
o5_b1_v3=tf.nn.conv2d(o5_b1_v2,w5_b1_v3,strides=[1,1,1,1],padding="SAME")
o5_b1=tf.nn.relu(shortcut5_b1+o5_b1_v3)

#   *Block2:
in5_b2=o5_b1
shortcut5_b2=in5_b2
w5_b2_v1=tf.Variable(tf.random_normal(shape=[1,1,2048,512],dtype=tf.float32))
o5_b2_v1=tf.nn.conv2d(o5_b1,w5_b2_v1,strides=[1,1,1,1],padding="SAME")
w5_b2_v2=tf.Variable(tf.random_normal(shape=[3,3,512,512],dtype=tf.float32))
o5_b2_v2=tf.nn.conv2d(o5_b2_v1,w5_b2_v2,strides=[1,1,1,1],padding="SAME")
w5_b2_v3=tf.Variable(tf.random_normal(shape=[1,1,512,2048],dtype=tf.float32))
o5_b2_v3=tf.nn.conv2d(o5_b2_v2,w5_b2_v3,strides=[1,1,1,1],padding="SAME")
o5_b2=tf.nn.relu(shortcut5_b2+o5_b2_v3)

#   *Block3:
in5_b3=o5_b2
shortcut5_b3=in5_b3
w5_b3_v1=tf.Variable(tf.random_normal(shape=[1,1,2048,512],dtype=tf.float32))
o5_b3_v1=tf.nn.conv2d(o5_b2,w5_b3_v1,strides=[1,1,1,1],padding="SAME")
w5_b3_v2=tf.Variable(tf.random_normal(shape=[3,3,512,512],dtype=tf.float32))
o5_b3_v2=tf.nn.conv2d(o5_b3_v1,w5_b3_v2,strides=[1,1,1,1],padding="SAME")
w5_b3_v3=tf.Variable(tf.random_normal(shape=[1,1,512,2048],dtype=tf.float32))
o5_b3_v3=tf.nn.conv2d(o5_b3_v2,w5_b3_v3,strides=[1,1,1,1],padding="SAME")
o5_b3=tf.nn.relu(shortcut5_b3+o5_b3_v3)

# Average Pooling
o6=tf.reduce_mean(o5_b3,[1,2],keep_dims=True)

# FC 1000d:
w7_=tf.Variable(tf.random_normal(shape=[1,1,2048,1000]))
o7=tf.nn.conv2d(o6,w7_,strides=[1,1,1,1],padding="SAME")

# Dense
dense=tf.squeeze(o6,[1,2])

# Softmax
prob=tf.nn.softmax(dense)

loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=y))


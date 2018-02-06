# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''


import tensorflow as tf

'''
# tf.get_variable()与tf.Variable()的区别
# 在 tf.name_scope下时，tf.get_variable()创建的变量名不受 name_scope 的影响，而且在未指定共享变量时，
# 如果重名会报错，tf.Variable()会自动检测有没有变量重名，如果有则会自行处理。
with tf.name_scope('name_scope_x'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var4 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var3.name, sess.run(var3))
    print(var4.name, sess.run(var4))
# 输出结果：
# var1:0 [-0.30036557]   可以看到前面不含有指定的'name_scope_x'
# name_scope_x/var2:0 [ 2.]
# name_scope_x/var2_1:0 [ 2.]  可以看到变量名自行变成了'var2_1'，避免了和'var2'冲突
'''



'''
# 如果使用tf.get_variable()创建变量，且没有设置共享变量，重名时会报错
with tf.name_scope('name_scope_1'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var2.name, sess.run(var2))
# ValueError: Variable var1 already exists, disallowed. Did you mean
# to set reuse=True in VarScope? Originally defined at:
# var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)

'''


# 所以要共享变量，需要使用tf.variable_scope()
with tf.variable_scope('variable_scope_y') as scope:
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    scope.reuse_variables()  # 设置共享变量
    var1_reuse = tf.get_variable(name='var1')
    var2 = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
    var2_reuse = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var1_reuse.name, sess.run(var1_reuse))
    print(var2.name, sess.run(var2))
    print(var2_reuse.name, sess.run(var2_reuse))
# 输出结果：
# variable_scope_y/var1:0 [-1.59682846]
# variable_scope_y/var1:0 [-1.59682846]   可以看到变量var1_reuse重复使用了var1
# variable_scope_y/var2:0 [ 2.]
# variable_scope_y/var2_1:0 [ 2.]

import tensorflow as tf
task_index=0
cluster_spec = tf.train.ClusterSpec({'ps' : ['localhost:2222'],'worker' : ['localhost:2223','localhost:2224']})
server = tf.train.Server(cluster_spec,job_name='worker',task_index=task_index)
tf.reset_default_graph()
tf.reset_default_graph()

#create local graph like normal specifying the local device
with tf.device('/job:worker/task:%d'%task_index):
    a = tf.Variable([0.],name='a',collections=[tf.GraphKeys.LOCAL_VARIABLES])
    b = tf.constant([100.])
    loss = tf.abs(a-b)
    
    optimizer = tf.train.GradientDescentOptimizer(.1)
    grads,local_vars = zip(*optimizer.compute_gradients(loss,var_list=tf.local_variables()))
    local_update = optimizer.apply_gradients(zip(grads,local_vars))
    
    
    init_local = tf.local_variables_initializer()

#create the globabl copies on the ps
with tf.device('/job:ps/task:0'):
    for v in tf.local_variables():
        v_g = tf.get_variable('g/'+v.op.name,
                            shape = v.shape,
                            dtype = v.dtype,
                            trainable=True,
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.TRAINABLE_VARIABLES])


#gloabl updates
with tf.device('/job:worker/task:%d'%task_index):
    #this needs to be updated.  Clearly not robust for any graph more complext
    global_vars = tf.global_variables()
    global_update = optimizer.apply_gradients(zip(grads,global_vars))

#create init op on the chief node
with tf.device('/job:worker/task:%d'%task_index):
    init_global = tf.global_variables_initializer()

sess = tf.Session(target=server.target)
sess.run([init_local])
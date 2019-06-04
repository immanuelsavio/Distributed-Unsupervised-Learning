import tensorflow as tf
cluster_spec = tf.train.ClusterSpec({'ps' : ['localhost:2222'],'worker' : ['localhost:2223','localhost:2224']})
ps = tf.train.Server(cluster_spec,job_name='ps')
ps.join()
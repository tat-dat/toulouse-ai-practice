import tensorflow as tf

with tf.device("/gpu:0"):
    a = tf.constant([1.,2.])
    b = tf.constant([3.,4.])

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    print(session.run(a))
    print(session.run(b))


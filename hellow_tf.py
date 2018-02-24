import tensorflow as tf

with tf.device("/gpu:0"):
    a = tf.constant([1.,2.])

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    print(session.run(a))


import time
import numpy as np
import tensorflow as tf

from inputs import *

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp/',
                    'Directory to write event logs and checkpoint.')
flags.DEFINE_string('train_data', '/Users/thomasfu/Downloads/adult.data',
                    'File pattern for training data.')
flags.DEFINE_integer('steps', 100, 'Number of batches to train.')


# def train():
#   """Train Criteo Deepcross Network for a number of steps."""
#
#   # Create a model.
#   model = Train(FLAGS.train_data)
#
#   # Create a supervisor.
#   supervisor = tf.train.Supervisor(graph=model.graph, logdir=FLAGS.train_dir)
#
#   with supervisor.managed_session() as sess:
#     while not supervisor.should_stop():
#       start_time = time.time()
#       _, loss, step = sess.run([model.train, model.loss, model.global_step])
#       #print('step:', step)
#       duration = time.time() - start_time
#
#       assert not np.isnan(loss), 'Model diverged with loss = NaN'
#
#       model.log_step(step, duration, loss, FLAGS.batch_size)
#       if step % 100 == 0:
#         print('setp = %d, loss = %f\n' % (step, loss))
#         supervisor.saver.save(sess, supervisor.save_path, global_step=step)
#
#       if step >= FLAGS.steps:
#         # Save the last checkpoint.
#         supervisor.request_stop()

def test():
    features, labels = inputs(FLAGS.train_data)

    # Create a supervisor.
    supervisor = tf.train.Supervisor(logdir=FLAGS.train_dir)

    step = 0
    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            step = step + 1
            print("step: ", step, sess.run([features, labels]))
            if step >= FLAGS.steps:
                supervisor.request_stop()


def main(_):
    test()


if __name__ == '__main__':
  tf.app.run()

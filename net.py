import tensorflow as tf
import os
import time


LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95


feature_len = 8
out_n = 20


logs_path = "logs"


class Net(object):
    def __init__(self, scope, copy=None, layers=2):
        self.scope = scope
        self.build_model(layers)

        if copy is not None:
            self.copy_model(copy)
            self.session = copy.session
        else:
            self.session = tf.Session()

        self.session.run(
            tf.variables_initializer(tf.global_variables(scope=self.scope)))

        if copy is None:
            self.train_writer = tf.summary.FileWriter(
                logs_path + "/{}_train".format(int(time.time())),
                graph=tf.get_default_graph()
            )
            self.i = 0

    def build_model(self, layers):
        with tf.variable_scope(self.scope):
            self.target = tf.placeholder(tf.float32, shape=[None, out_n])
            self.target_mask = tf.placeholder(tf.float32, shape=[None, out_n])

            self.input = tf.placeholder(
                tf.float32, shape=[None, feature_len])

            if layers == 1:
                self.out = tf.layers.dense(
                    self.input,
                    out_n
                )
            else:
                self.dense1 = tf.layers.dense(
                    self.input,
                    20,
                    activation=tf.nn.relu
                )
                self.out = tf.layers.dense(
                    self.dense1,
                    out_n
                )

            self.action = tf.argmax(self.out, axis=1)
            self.value = tf.reduce_max(self.out, axis=1)

            self.delta = tf.subtract(
                self.out * self.target_mask,
                self.target * self.target_mask
            )
            self.loss = tf.reduce_mean(tf.square(self.delta))
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=LEARNING_RATE,
                momentum=GRADIENT_MOMENTUM,
            ).minimize(self.loss)

            tf.summary.scalar("loss", self.loss)
            self.merged_summary = tf.summary.merge_all()

            self.saver = tf.train.Saver(
                var_list=tf.global_variables(scope=self.scope)
            )

    def copy_model(self, copy):
        self.assigners = []
        for var in tf.global_variables(scope=self.scope):
            copy_name = var.name.replace(self.scope, copy.scope)
            assigner = var.assign(tf.global_variables(scope=copy_name)[0])
            self.assigners.append(assigner)

    def __del__(self):
        try:
            self.session.close()
        except AttributeError:
            pass

    def optimize(self, X_batch, y_batch, mask):
        feed_dict = {
            self.input: X_batch,
            self.target: y_batch,
            self.target_mask: mask
        }

        _, summary, loss = self.session.run(
            [self.optimizer, self.merged_summary, self.loss], feed_dict=feed_dict)

        self.train_writer.add_summary(summary, self.i)
        self.i += 1
        return loss

    def get_action(self, sequence):
        feed_dict = {
            self.input: sequence
        }

        action = self.session.run(self.action, feed_dict=feed_dict)
        return action

    def get_value(self, sequence):
        feed_dict = {
            self.input: sequence
        }

        value = self.session.run(self.value, feed_dict=feed_dict)
        return value

    def copy(self):
        self.session.run(self.assigners)

    def save(self, model_name):
        if not os.path.isdir("models"):
            os.makedirs("models")
        save_path = "models/{}.checkpoint".format(model_name)
        save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))
        self.saver.save(self.session, save_path)
        print("Saved session to {}".format(save_path))

    def load(self, model_name):
        load_path = "models/{}.checkpoint".format(model_name)
        load_path = os.path.abspath(os.path.join(os.getcwd(), load_path))
        self.saver.restore(self.session, load_path)
        print("Restored session from {}".format(load_path))


if __name__ == '__main__':
    net = Net('net1')
    net2 = Net('net2', copy=net)

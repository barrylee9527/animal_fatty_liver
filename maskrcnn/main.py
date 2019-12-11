# @Time    : 2018.10.17
# @Author  : kawa Yeung
# @Licence : bio-totem


import tensorflow as tf

from utils.probmat_utils import walk_file

flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_integer("patch_R", 598, "patch row")
# flags.DEFINE_integer("patch_C", 299, "patch column")
# flags.DEFINE_integer("patch_size", 54, "patch size")
#flags.DEFINE_float("cc_prob_threshold", 0.5, "cancer cell probability")
def main(unused_argv):
    # patch_R = FLAGS.patch_R
    # patch_C = FLAGS.patch_C
#    cc_prob_threshold = FLAGS.cc_prob_threshold
#     patch_size = FLAGS.patch_size
    walk_file()
if __name__ == "__main__":
    tf.app.run()
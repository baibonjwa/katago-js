import sys
import argparse
import json
import numpy as np
import tensorflow as tf
sys.path.append("../python")
import common
from model import Model

if __name__ == "__main__":
    pos_len = 19
    description = """
    Play go with a trained neural net!
    Implements a basic GTP engine that uses the neural net directly to play moves.
    """

    parser = argparse.ArgumentParser(description=description)
    common.add_model_load_args(parser)
    parser.add_argument("-name-scope", help="Name scope for model variables", required=False)
    args = vars(parser.parse_args())

    (model_variables_prefix, model_config_json) = common.load_model_paths(args)
    name_scope = args["name_scope"]
    with open(model_config_json) as f:
        model_config = json.load(f)

    if name_scope is not None:
        with tf.name_scope(name_scope):
            model = Model(model_config,pos_len,{
                  "is_training": tf.constant(False,dtype=tf.bool),
                  #"symmetries": tf.constant(False, shape=[3], dtype=tf.bool),
                  "include_history": tf.constant(1.0, shape=[1,5], dtype=tf.float32)
            })
    else:
        model = Model(model_config,pos_len,{
                  "is_training": tf.constant(False,dtype=tf.bool),
                  #"symmetries": tf.constant(False, shape=[3], dtype=tf.bool),
                  "include_history": tf.constant(1.0, shape=[1,5], dtype=tf.float32)
        })

    saver = tf.train.Saver(
        max_to_keep = 10000,
        save_relative_paths = True,
    )

    with tf.Session() as session:
        saver.restore(session, model_variables_prefix)
        tf.train.write_graph(session.graph,"./tmp","graph.pbtxt", as_text=True)

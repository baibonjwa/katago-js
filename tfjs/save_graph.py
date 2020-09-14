import sys
import argparse
import json
import numpy as np
import tensorflow as tf
sys.path.append("../python")
import common
from model import Model

if __name__ == "__main__":
    description = """
    save a model in saved model format
    """

    parser = argparse.ArgumentParser(description=description)
    common.add_model_load_args(parser)
    parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
    parser.add_argument('-board-size', help='Board size', required=True)
    parser.add_argument('-disable-symmetries', help='Disable symmetries', required=True)
    args = vars(parser.parse_args())

    (model_variables_prefix, model_config_json) = common.load_model_paths(args)
    pos_len = int(args["board_size"])
    name_scope = args["name_scope"]
    disable_symmetries = bool(args["disable_symmetries"])
    with open(model_config_json) as f:
        model_config = json.load(f)

    if disable_symmetries:
        constants = {
            "is_training": tf.constant(False,dtype=tf.bool),
            "include_history": np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype="float32"),
            "symmetries": np.array([False, False, False], dtype="bool"),
        }
    else:
        constants = {
            "is_training": tf.constant(False,dtype=tf.bool),
            "include_history": np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype="float32")
        }

    if name_scope is not None:
        with tf.compat.v1.variable_scope(name_scope):
            model = Model(model_config,pos_len, constants)
    else:
        model = Model(model_config,pos_len, constants)

    saver = tf.compat.v1.train.Saver(
        max_to_keep = 10000,
        save_relative_paths = True,
    )

    with tf.compat.v1.Session() as session:
        saver.restore(session, model_variables_prefix)
        if disable_symmetries:
            inputs = {
                "swa_model/bin_inputs": model.bin_inputs,
                "swa_model/global_inputs": model.global_inputs,
            }
        else:
            inputs = {
                "swa_model/bin_inputs": model.bin_inputs,
                "swa_model/global_inputs": model.global_inputs,
                "symmetries": model.symmetries,
            },
        tf.compat.v1.saved_model.simple_save(
            session,
            './saved_model',
            inputs=inputs,
            outputs={
                'swa_model/policy_output': model.policy_output,
                'swa_model/value_output': model.value_output,
                'swa_model/miscvalues_output': model.miscvalues_output,
                'swa_model/ownership_output': model.ownership_output,
            }
        )

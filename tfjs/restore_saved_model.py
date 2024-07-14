import sys
import tensorflow as tf
from tensorflow.python import ops

def get_graph_def_from_file(graph_filepath):
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def convert_graph_def_to_saved_model(export_dir, graph_filepath):
  if tf.gfile.Exists(export_dir):
    tf.gfile.DeleteRecursively(export_dir)
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.saved_model.simple_save(
        session,
        export_dir,
        inputs={ node.name: session.graph.get_tensor_by_name('{}:0'.format(node.name)) for node in graph_def.node if node.op=='Placeholder' },
        outputs={
            "swa_model/policy_output": session.graph.get_tensor_by_name("swa_model/policy_output:0"),
            "swa_model/value_output": session.graph.get_tensor_by_name("swa_model/value_output:0"),
            "swa_model/miscvalues_output": session.graph.get_tensor_by_name("swa_model/miscvalues_output:0"),
            "swa_model/ownership_output": session.graph.get_tensor_by_name("swa_model/ownership_output:0"),
        }
    )
    print('Optimized graph converted to SavedModel!')

if __name__ == "__main__":
    convert_graph_def_to_saved_model( sys.argv[2], sys.argv[1])
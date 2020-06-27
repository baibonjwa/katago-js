MODEL = g170-b6c96-s175395328-d26788732
# Specify target board size below.
#SIZE = 19
SIZE = 7
TMP_DIR = tmp
SAVED_MODEL_DIR = saved_model
SAVED_MODEL = $(SAVED_MODEL_DIR)/saved_model.pb
FROZEN_MODEL = $(TMP_DIR)/frozen_model.pb
OPTIMIZED_MODEL = $(TMP_DIR)/optimized.pb
SAVED_MODEL2_DIR = saved_model2
SAVED_MODEL2 = $(SAVED_MODEL2_DIR)/saved_model.pb

ifeq "$(DISABLE_SYMMETRIES)" "false"
	INPUT_NODES = swa_model/bin_inputs,swa_model/global_inputs,swa_model/symmetries
else
	INPUT_NODES = swa_model/bin_inputs,swa_model/global_inputs
endif
OUTPUT_NODES = swa_model/policy_output,swa_model/value_output,swa_model/miscvalues_output,swa_model/ownership_output
TARGET_DIR = web_model
TARGET = $(TARGET_DIR)/model.json

all: $(TARGET)

$(SAVED_MODEL):
	python save_graph.py -name-scope swa_model -model-variables-prefix ../models/$(MODEL)/saved_model/variables/variables -model-config-json ../models/$(MODEL)/model.config.json -board-size $(SIZE)

$(FROZEN_MODEL): $(SAVED_MODEL)
	mkdir -p $(TMP_DIR)
	freeze_graph --input_saved_model_dir=$(SAVED_MODEL_DIR) --output_graph=$@ --output_node_names=$(OUTPUT_NODES)

$(OPTIMIZED_MODEL): $(FROZEN_MODEL)
	$(HOME)/OpenSources/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
		--in_graph=$< \
		--out_graph=$@ \
		--inputs="$(INPUT_NODES)" \
		--outputs="$(OUTPUT_NODES)" \
		--transforms='strip_unused_nodes fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'  

$(SAVED_MODEL2): $(OPTIMIZED_MODEL)
	python restore_saved_model.py $(OPTIMIZED_MODEL) $(SAVED_MODEL2_DIR)

$(TARGET): $(SAVED_MODEL2)
	tensorflowjs_converter \
	    --input_format=tf_saved_model \
	    --saved_model_tags=serve \
	    $(SAVED_MODEL2_DIR) \
	    $(TARGET_DIR)
	echo "{\"version\":8}" > $(TARGET_DIR)/metadata.json

clean:
	rm -rf $(SAVED_MODEL_DIR) $(TMP_DIR) $(SAVED_MODEL2_DIR) $(TARGET_DIR)
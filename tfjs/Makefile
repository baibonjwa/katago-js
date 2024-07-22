DISABLE_SYMMETRIES = true
# MODEL = kata1-b20c256-s241116160-d150094945
# MODEL = b10c128-s1141046784-d204142634
MODEL = kata1-b18c384nbt-s9996604416-d4316597426
# Specify target board size below.
SIZE = 19
#SIZE = 7
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
TARGET_DIR = web_model/${MODEL}
TARGET = $(TARGET_DIR)/model.json

all: $(TARGET)

$(SAVED_MODEL): Makefile
	rm -rf $(SAVED_MODEL_DIR)
	python save_graph.py -name-scope swa_model -model-variables-prefix ../models/$(MODEL)/saved_model/variables/variables -model-config-json ../models/$(MODEL)/model.config.json -board-size $(SIZE) -disable-symmetries=$(DISABLE_SYMMETRIES)

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

$(TARGET): $(SAVED_MODEL)
	tensorflowjs_converter \
	    --input_format=tf_saved_model \
	    --saved_model_tags=serve \
	    $(SAVED_MODEL_DIR) \
	    $(TARGET_DIR)
	echo "{\"version\":8}" > $(TARGET_DIR)/metadata.json

clean:
	rm -rf $(SAVED_MODEL_DIR) $(TMP_DIR) $(SAVED_MODEL2_DIR) $(TARGET_DIR)
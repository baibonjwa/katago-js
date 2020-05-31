MODEL = g104-b6c96-s97778688-d23397744
# Specify SIZE except for 19x19 board 
# SIZE = 7

all: web_model

saved_model/saved_model.pb:
	python save_graph.py -name-scope swa_model -model-variables-prefix ../models/$(MODEL)/saved_model/variables/variables -model-config-json ../models/$(MODEL)/model.config.json -board-size $(SIZE)

web_model: saved_model/saved_model.pb
	tensorflowjs_converter \
	    --input_format=tf_saved_model \
	    --saved_model_tags=serve \
	    saved_model \
	    web_model

clean:
	rm -rf saved_model web_model
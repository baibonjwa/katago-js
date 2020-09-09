// The reason why ES5 is https://github.com/emscripten-core/emscripten/issues/9190

function loadJSON(url) {
    return new Promise(function(res, rej) {
        const request = new XMLHttpRequest();
        request.responseType = "json";
        request.open("GET", url);
        request.addEventListener("load", function() {
            console.log(request);
            res(request.response);
        });
        request.addEventListener("error", function() {
            rej(request.statusText);
        });
        request.addEventListener("abort", function() {
            rej(request.statusText);
        });
        request.send();
    });
}

var GraphModelWrapper = function() {
    this.model = null;
    // TODO - modelのメタデータ対応
    this.version = 8;
};

GraphModelWrapper.prototype.AUTO = 0;
GraphModelWrapper.prototype.CPU = 1;
GraphModelWrapper.prototype.WEBGL = 2;

GraphModelWrapper.prototype.getBackend = function() {
    switch (tf.getBackend()) {
        case "cpu":
        return this.CPU;
        case "webgl":
        return this.WEBGL;
        default:
        return 0;
    }
}

GraphModelWrapper.prototype.setBackend = function(backend) {
    var be;
    switch (backend) {
        case this.AUTO:
        be = typeof OffscreenCanvas !== 'undefined' ? "webgl" : "cpu";
        break;
        case this.CPU:
        be = "cpu";
        break;
        case this.WEBGL:
        be = "webgl";
        break;
        default:
        return;
    }
    tf.setBackend(be);
};

GraphModelWrapper.prototype.downloadModel = function(charp) {
    return Asyncify.handleSleep((function(wakeUp) {
        const model = UTF8ToString(charp);
        Promise.all([
            loadJSON(model + "/metadata.json")
                .then((function(json) {
                    this.version = json.version;
                }).bind(this)),
            tf.loadGraphModel(model + "/model.json")
                .then((function(model) {
                    this.model = model;
                }).bind(this))
        ]).then(function() {
            wakeUp(1);
        }).catch(function(errors) {
            console.log(errors);
            wakeUp(0);
        });
    }).bind(this));
};

GraphModelWrapper.prototype.removeModel = function() {
    this.model = null;
};

GraphModelWrapper.prototype.predict = function(
    batches,
    inputBuffer, boardWxH, inputBufferChannels,
    inputGlobalBuffer, inputGlobalBufferChannels,
    symmetriesBuffer, symmetriesBufferLength,
    values, miscvalues, ownerships, policies) {
    return Asyncify.handleSleep(function(wakeUp) {
        try {
            const bin_inputs = new Float32Array(Module.HEAPF32.buffer, inputBuffer, batches * boardWxH * inputBufferChannels);
            const global_inputs = new Float32Array(Module.HEAPF32.buffer, inputGlobalBuffer, batches * inputGlobalBufferChannels);
            const _symmetries = new Int8Array(Module.HEAP8.buffer, symmetriesBuffer, symmetriesBufferLength);
            const symmetries = new Array(symmetriesBufferLength);
            for (let i = 0; i < symmetriesBufferLength; i++) {
                symmetries[i] = _symmetries[i] != 0
            }
            this.model.executeAsync({
                "swa_model/bin_inputs": tf.tensor(bin_inputs, [batches, boardWxH, inputBufferChannels], 'float32'),
                "swa_model/global_inputs": tf.tensor(global_inputs, [batches, inputGlobalBufferChannels], 'float32'),
                "swa_model/symmetries": tf.tensor(symmetries, [symmetriesBufferLength], 'bool'),
            }).then(function(results) {
                var i;
                const miscvaluesSize = this.version === 8 ? 10 : 6;
                for (i = 0; i < results.length; i++) {
                    const result = results[i];
                    const data = result.dataSync();
                    switch (result.size) {
                        case 3: //value
                        Module.HEAPF32.set(data, values / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case miscvaluesSize: // miscvalues
                        Module.HEAPF32.set(data, miscvalues / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case boardWxH: // ownership
                        Module.HEAPF32.set(data, ownerships / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                        case (boardWxH + 1) * 2: // policy
                        Module.HEAPF32.set(data, policies / Module.HEAPF32.BYTES_PER_ELEMENT);
                        break;
                    }
                }
                wakeUp(1);
            });
        } catch (e) {
            console.error(e);
            wakeUp(0);
        }
    }.bind(this));
};

GraphModelWrapper.prototype.getModelVersion = function() {
    return this.version;
};

if (Module['ENVIRONMENT_IS_PTHREAD']) {
    importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.3.0/dist/tf.min.js");
    if (typeof OffscreenCanvas !== 'undefined') {
        self.document = {
            createElement: function() {
                return new OffscreenCanvas(640, 480);
            }
        };
        self.window = self;
        self.screen = {
            width: 640,
            height: 480
        };
        self.HTMLVideoElement = function() {};
        self.HTMLImageElement = function() {};
        self.HTMLCanvasElement = OffscreenCanvas;
    } else {
        console.error("no offscreen canvas");
    }
}
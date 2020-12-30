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
GraphModelWrapper.prototype.WASM = 3;

GraphModelWrapper.prototype.getBackend = function() {
    switch (tf.getBackend()) {
        case "cpu":
        return this.CPU;
        case "webgl":
        return this.WEBGL;
        case "wasm":
        return this.WASM;
        default:
        return 0;
    }
}

GraphModelWrapper.prototype.setBackend = function(backend) {
    var be;
    switch (backend) {
        case this.AUTO:
        be = typeof OffscreenCanvas !== 'undefined' ? "webgl" : "wasm";
        break;
        case this.CPU:
        be = "cpu";
        break;
        case this.WEBGL:
        be = "webgl";
        break;
        case this.WASM:
        be = "wasm";
        break;
        default:
        return;
    }
    return Asyncify.handleSleep(wakeUp => {
        tf.setBackend(be).then(s => {
            console.log("setBackend", be, s);
            if (s) {
                wakeUp(1);
            } else if (backend === this.AUTO && be === "webgl") {
                // OffscreenCanvasが存在してもsetBackendが失敗するケースがあるのでwasmにフォールバックさせる
                console.log("try wasm for setBackend");
                tf.setBackend("wasm").then(s => {
                    wakeUp(s ? 1 : 0);
                });
            } else {
                wakeUp(0);
            }
        });
    });
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
    values, miscvalues, ownerships, policies) {
    return Asyncify.handleSleep(function(wakeUp) {
        try {
            const bin_inputs = new Float32Array(Module.HEAPF32.buffer, inputBuffer, batches * boardWxH * inputBufferChannels);
            const global_inputs = new Float32Array(Module.HEAPF32.buffer, inputGlobalBuffer, batches * inputGlobalBufferChannels);
            const start = Date.now();
            this.model.executeAsync({
                "swa_model/bin_inputs": tf.tensor(bin_inputs, [batches, boardWxH, inputBufferChannels], 'float32'),
                "swa_model/global_inputs": tf.tensor(global_inputs, [batches, inputGlobalBufferChannels], 'float32'),
            }).then(function(results) {
                console.log("executeAsync", Date.now() - start);
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
    const version ="2.8.2"
    importScripts(
        `//cdn.jsdelivr.net/npm/@tensorflow/tfjs@${version}/dist/tf.min.js`,
        `//cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${version}/dist/tf-backend-wasm.min.js`    );
    tf.wasm.setWasmPaths(`//cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${version}/dist/`);
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
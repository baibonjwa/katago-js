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

class GraphModelWrapper {
    constructor() {
        this.AUTO = 0;
        this.CPU = 1;
        this.WEBGL = 2;
        this.WASM = 3;
        this.WEBGPU = 4;
        
        this.device = null;
        this.model = null;
        // TODO - modelのメタデータ対応
        this.version = 8;
    }

    getBackend() {
        switch (tf.getBackend()) {
            case "cpu":
            return this.CPU;
            case "webgl":
            return this.WEBGL;
            case "wasm":
            return this.WASM;
            case "webgpu":
            return this.WEBGPU;
            default:
            return 0;
        }
    }

    setBackend(backend) {
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
            case this.WEBGPU:
            be = "webgpu";
            break;
            default:
            return;
        }
        return Asyncify.handleSleep(async wakeUp => {
            const s = await tf.setBackend(be);
            console.log("setBackend", be, s);
            if (s) {
                if (be === "webgpu") {
                    const adapter = await navigator.gpu.requestAdapter();
                    this.device = await adapter.requestDevice();
                }
                wakeUp(1);
            } else if (backend === this.AUTO && be === "webgl") {
                // OffscreenCanvasが存在してもsetBackendが失敗するケースがあるのでwasmにフォールバックさせる
                console.log("try wasm for setBackend");
                const s = await tf.setBackend("wasm");
                wakeUp(s ? 1 : 0);
            } else {
                wakeUp(0);
            }
        });
    }

    downloadMetadata(charp) {
        return Asyncify.handleSleep(async wakeUp => {
            const model = UTF8ToString(charp);
            try {
                let json = await loadJSON(model + "/metadata.json");
                this.version = json.version;
                wakeUp(1);
            } catch (error) {
                console.error(error);
                wakeUp(0);
            }
        });
    }

    downloadModel(charp) {
        return Asyncify.handleSleep(async wakeUp => {
            const modelName = UTF8ToString(charp);
            try {
                let model = await tf.loadGraphModel(modelName + "/model.json");
                this.model = model;
                wakeUp(1);
            } catch (errors) {
                console.error(errors);
                wakeUp(0);
            }
        });
    }

    removeModel() {
        this.model = null;
    }

    predict(
        batches,
        inputBuffer, boardWxH, inputBufferChannels,
        inputGlobalBuffer, inputGlobalBufferChannels,
        values, miscvalues, ownerships, policies
    ) {
        return Asyncify.handleSleep(async wakeUp => {
            try {
                let bin_inputs = new Float32Array(Module.HEAPF32.buffer, inputBuffer, batches * boardWxH * inputBufferChannels);
                let global_inputs = new Float32Array(Module.HEAPF32.buffer, inputGlobalBuffer, batches * inputGlobalBufferChannels);
                const start = Date.now();
                let results = await this.model.executeAsync({
                    "swa_model/bin_inputs": tf.tensor(bin_inputs, [batches, boardWxH, inputBufferChannels], 'float32'),
                    "swa_model/global_inputs": tf.tensor(global_inputs, [batches, inputGlobalBufferChannels], 'float32'),
                });
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
            } catch (e) {
                console.error(e);
                wakeUp(0);
            }
        });
    }

    getModelVersion() {
        return this.version;
    }
}

if (Module['ENVIRONMENT_IS_PTHREAD']) {
    if (location.protocol === "http:") {
        importScripts(
            "tf.min.js",
            "tf-backend-cpu.min.js",
            "tf-backend-webgl.min.js");
            // wasmを使うにはmin.jsだけじゃなくwasmもローカルにコピーしないといけない
    } else {
        const version ="4.6.0";
        importScripts(
            `//cdn.jsdelivr.net/npm/@tensorflow/tfjs@${version}/dist/tf.min.js`,
            `//cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@${version}/dist/tf-backend-cpu.min.js`,
            `//cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@${version}/dist/tf-backend-webgl.min.js`,
            `//cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${version}/dist/tf-backend-wasm.min.js`);
        tf.wasm.setWasmPaths(`//cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${version}/dist/`);
    }
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
#ifdef USE_TFJS_BACKEND

#include "../core/config_parser.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/desc.h"

extern "C" {
  extern int getBackend();
  extern int setBackend(int);
  extern int downloadModel(int);
  extern void removeModel();
  extern int predict(int, int, int, int, int, int, int, int, int, int);
  extern int jsGetModelVersion();
}

using namespace std;

struct ComputeContext {
  int backend = 0; // auto: 0, cpu: 1, webgl: 2, wasm: 3
  int nnXLen;
  int nnYLen;

  ComputeContext(ConfigParser& cfg, Logger* logger) {
    if(cfg.contains("tfjsBackend")) {
      string tfjsBackend = cfg.getString("tfjsBackend");
      logger->write(string("backend: ") + tfjsBackend);
      if(tfjsBackend == "cpu") {
        backend = 1;
      } else if(tfjsBackend == "webgl") {
        backend = 2;
      } else if(tfjsBackend == "wasm") {
        backend = 3;
      }
    } else {
      logger->write("backend: auto");
    }
  }
};

void NeuralNet::globalInitialize() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

void NeuralNet::globalCleanup() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

// A handle to the loaded neural network model.
struct LoadedModel {
  ModelDesc modelDesc;
  string name;

  LoadedModel(const string& fileName) {
    /* emscripten note:
       This constructor is called in main thread.
       But the instance is actually used in NNEvaluator thread.
       So you need to load the tfjs model in NNEvaluator thread.
    */
    name = fileName;
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

// A handle to the local compute backend. Not thread-safe, each handle should
// only be used by one thread.
struct ComputeHandle {
  const LoadedModel* model;
  int policySize;
  int nnXLen;
  int nnYLen;

  ComputeHandle(const LoadedModel* loadedModel, int nnX, int nnY) {
    model = loadedModel;
    nnXLen = nnX;
    nnYLen = nnY;
    policySize = NNPos::getPolicySize(nnXLen, nnYLen);
  }
};

// The interface for the input buffers for the neural network. The MCTS code
// uses this interface to pass data into the neural network for computation.
struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer; //Host pointer
  float* userInputGlobalBuffer; //Host pointer

  float* policyResults; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * xSize * ySize;
    singleInputBytes = (size_t)m.numInputChannels * xSize * ySize * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = (size_t)m.numInputGlobalChannels * sizeof(float);
    singlePolicyResultElts = (size_t)(1 + xSize * ySize);
    singlePolicyResultBytes = (size_t)(1 + xSize * ySize) * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = (size_t)m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = (size_t)m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * xSize * ySize;
    singleOwnershipResultBytes = (size_t)m.numOwnershipChannels * xSize * ySize * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSize * xSize * ySize * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSize * (1 + xSize * ySize) * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSize * xSize * ySize * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[(size_t)m.numInputChannels * maxBatchSize * xSize * ySize];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];

    policyResults = new float[(size_t)maxBatchSize * (1 + xSize * ySize)];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * xSize * ySize * m.numOwnershipChannels];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  ConfigParser& cfg,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const std::string& openCLTunerFile,
  const std::string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  ComputeContext* context = new ComputeContext(cfg, logger);
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  (void)computeContext;
}

LoadedModel* NeuralNet::loadModelFile(const string& file, const std::string& expectedSha256) {
  return new LoadedModel(file);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  removeModel();
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return jsGetModelVersion();
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)maxBatchSize;
  (void)requireExactNNLen;
  (void)inputsUseNHWC;
  (void)gpuIdxForThisThread;
  (void)serverThreadIdx;
  if (setBackend(context->backend) == 1) {
    logger->write("backend was initialized");
  } else {
    logger->write("backend initialization failed");
  }
  auto backend = getBackend();
  switch (backend) {
    case 1:
    logger->write("backend: cpu");
    break;
    case 2:
    logger->write("backend: webgl");
    break;
    case 3:
    logger->write("backend: wasm");
    break;
    default:
    logger->write("backend: unkown");
  }
  if (downloadModel((int)loadedModel->name.c_str()) == 1) {
    loadedModel->modelDesc.version = getModelVersion(loadedModel);
    if (loadedModel->modelDesc.version >= 9) {
      loadedModel->modelDesc.numInputChannels = 22;
      loadedModel->modelDesc.numInputGlobalChannels = 19;
      loadedModel->modelDesc.numValueChannels = 3;
      loadedModel->modelDesc.numOwnershipChannels = 1;
      loadedModel->modelDesc.numScoreValueChannels = 6;
    } else if (loadedModel->modelDesc.version == 8) {
      loadedModel->modelDesc.numInputChannels = 22;
      loadedModel->modelDesc.numInputGlobalChannels = 19;
      loadedModel->modelDesc.numValueChannels = 3;
      loadedModel->modelDesc.numOwnershipChannels = 1;
      loadedModel->modelDesc.numScoreValueChannels = 4;
    } else if (loadedModel->modelDesc.version == 5) {
      loadedModel->modelDesc.numInputChannels = 22;
      loadedModel->modelDesc.numInputGlobalChannels = 14;
      loadedModel->modelDesc.numValueChannels = 3;
      loadedModel->modelDesc.numOwnershipChannels = 1;
      loadedModel->modelDesc.numScoreValueChannels = 2;
    }
    return new ComputeHandle(loadedModel, context->nnXLen, context->nnYLen);
  } else {
    logger->write("Failed downloadModel");
    return NULL;
  }
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  (void)maxBatchSize;
  (void)nnXLen;
  (void)nnYLen;
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

/*
float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
}

float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* inputBuffers) {
  return inputBuffers->symmetriesBuffer;
}

int NeuralNet::getBatchEltSpatialLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputElts;
}

int NeuralNet::getBatchEltGlobalLen(const InputBuffers* inputBuffers) {
 return inputBuffers->singleInputGlobalElts;
}
*/

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  std::vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->model->modelDesc.version;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, true, inputBufs[nIdx]->symmetry);
  }

  clock_t start = clock();
  if(predict(
    batchSize,
    (int)inputBuffers->userInputBuffer,
    nnXLen * nnYLen,
    gpuHandle->model->modelDesc.numInputChannels,
    (int)inputBuffers->userInputGlobalBuffer,
    gpuHandle->model->modelDesc.numInputGlobalChannels,
    (int)inputBuffers->valueResults,
    (int)inputBuffers->scoreValueResults,
    (int)inputBuffers->ownershipResults,
    (int)inputBuffers->policyResults
  ) != 1) {
    cerr << "predict error " << endl;
  }
  cerr << "predict time(ms): " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC * 1000.0 << endl;
  assert(!isnan(inputBuffers->valueResults[0]));

  assert(outputs.size() == batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    const float* policySrcBuf = inputBuffers->policyResults + row * gpuHandle->policySize;
    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //policy probabilities and white game outcome probabilities
    //Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    policyProbs[gpuHandle->policySize-1] = policySrcBuf[gpuHandle->policySize-1];

    int numValueChannels = gpuHandle->model->modelDesc.numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->modelDesc.numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(version >= 9) {
      int numScoreValueChannels = gpuHandle->model->modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    }
    else if(version >= 8) {
      int numScoreValueChannels = gpuHandle->model->modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 4) {
      int numScoreValueChannels = gpuHandle->model->modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 3) {
      int numScoreValueChannels = gpuHandle->model->modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }

}



bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}

//Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

/*
bool NeuralNet::testEvaluateSymmetry(
  int batchSize,
  int numChannels,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const bool* symmetriesBuffer,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  (void)batchSize;
  (void)numChannels;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)symmetriesBuffer;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}
*/
#endif
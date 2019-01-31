// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int it
) {
    if (it == 0) {
        // --- Multiply Stimulus ---

        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

            _hiddenStimulus[hiddenIndex] = 0.0f;

            // For each visible layer
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                _hiddenStimulus[hiddenIndex] += vl._weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld._size.z);
            }

            _hiddenActivations[hiddenIndex] = _hiddenStimulus[hiddenIndex];
        }
    }
    else {
        // --- Multiply and Subtract Recon ---

        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

            _hiddenActivations[hiddenIndex] += _hiddenStimulus[hiddenIndex];

            // For each visible layer
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                _hiddenActivations[hiddenIndex] -= vl._weights.multiplyOHVs(vl._reconCs, hiddenIndex, vld._size.z);
            }
        }
    }

    // --- Find max ---

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

        if (_hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = _hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    _hiddenCs[address2C(pos, Int2(_hiddenSize.x, _hiddenSize.y))] = maxIndex;
}

void SparseCoder::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // --- Multiply ---
    
    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3C(Int3(pos.x, pos.y, vc), vld._size);

        vl._visibleActivations[visibleIndex] = vl._weights.multiplyOHVsT(_hiddenCs, visibleIndex, _hiddenSize.z);

        if (vl._visibleActivations[visibleIndex] > maxActivation) {
            maxActivation = vl._visibleActivations[visibleIndex];
            maxIndex = vc;
        }
    }

    vl._reconCs[address2C(pos, Int2(_hiddenSize.x, _hiddenSize.y))] = maxIndex;
}

void SparseCoder::learnWeights(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2C(pos, Int2(vld._size.x, vld._size.y));

    int targetC = (*inputCs[vli])[visibleColumnIndex];

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3C(Int3(pos.x, pos.y, vc), vld._size);

        float sum = vl._weights.multiplyOHVsT(_hiddenCs, visibleIndex, _hiddenSize.z);
    
        float delta = _alpha * ((vc == targetC ? 1.0f : 0.0f) - sigmoid(sum));

        vl._weights.deltaOHVsT(_hiddenCs, delta, visibleIndex, _hiddenSize.z);
    }
}

void SparseCoder::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = weightDist(cs._rng);

        // Generate transpose (needed for reconstruction)
        vl._weights.initT();

        // Visible activations buffer
        vl._visibleActivations = FloatBuffer(numVisible);

        // Reconstruction states
        vl._reconCs = IntBuffer(numVisibleColumns);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);

    // Hidden stimulus
    _hiddenStimulus = FloatBuffer(numHidden);

    // Hidden activations
    _hiddenActivations = FloatBuffer(numHidden);
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &visibleCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Sparse coding iterations: forward, reconstruct, repeat
    for (int it = 0; it < _explainIters; it++) {
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                forward(Int2(x, y), cs._rng, visibleCs, it);
#else
        runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, it), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

        if (it != _explainIters - 1) {
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < vld._size.x; x++)
                    for (int y = 0; y < vld._size.y; y++)
                        backward(Int2(x, y), cs._rng, visibleCs, vli);
#else
                runKernel2(cs, std::bind(SparseCoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
            }
        }
    }

    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    learnWeights(Int2(x, y), cs._rng, visibleCs, vli);
#else
            runKernel2(cs, std::bind(SparseCoder::learnWeightsKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }
}

void SparseCoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_explainIters), sizeof(int));

    writeBufferToStream(os, &_hiddenCs);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        writeSMToStream(os, vl._weights);
    }
}

void SparseCoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_explainIters), sizeof(int));

    readBufferFromStream(is, &_hiddenCs);

    _hiddenStimulus = FloatBuffer(numHidden);

    _hiddenActivations = FloatBuffer(numHidden);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        readSMFromStream(is, vl._weights);

        vl._visibleActivations = FloatBuffer(numVisible);

        vl._reconCs = IntBuffer(numVisibleColumns);
    }
}
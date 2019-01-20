// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

// Kernels
void SparseCoder::init(int pos, std::mt19937 &rng, int vli) {
    // Initialize weights into uniform range
	std::uniform_real_distribution<float> weightDist(0.99f, 1.0f);

    _visibleLayers[vli]._weights._nonZeroValues[pos] = weightDist(rng);
}

void SparseCoder::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, bool firstIter) {
    // --- Clear Activations ---

    if (firstIter) {
        for (int hc = 0; hc < _hiddenSize.z; hc++)
            _hiddenStimulus[address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y))] = 0.0f;

        // --- Multiply Stimulus ---

        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            int hiddenIndex = address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y));

            // For each visible layer
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                vl._weights.multiplyRangeOfRowOHVs(*inputCs[vli], _hiddenStimulus, hiddenIndex, 1, vld._size.z);
            }
        }

        // Set activation to stimulus
        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            int hiddenIndex = address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y));

            _hiddenActivations[hiddenIndex] = _hiddenStimulus[hiddenIndex];
        }
    }
    else {
        // Increment activations by stimulus
        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            int hiddenIndex = address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y));

            _hiddenActivations[hiddenIndex] += _hiddenStimulus[hiddenIndex];
        }

        // --- Multiply and Subtract Recon ---

        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            int hiddenIndex = address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y));

            // For each visible layer
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                vl._weights.multiplyRangeOfRowOHVs(vl._reconCs, _hiddenActivations, hiddenIndex, 1, vld._size.z, true);
            }
        }
    }

    // --- Find max ---

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3R(Int3(pos.x, pos.y, hc), Int2(_hiddenSize.x, _hiddenSize.y));

        if (_hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = _hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    _hiddenCs[address2R(pos, _hiddenSize.x)] = maxIndex;
}

void SparseCoder::backward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, int vli) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    // Clear activations
    for (int vc = 0; vc < vld._size.z; vc++)
        vl._visibleActivations[address3R(Int3(pos.x, pos.y, vc), Int2(vld._size.x, vld._size.y))] = 0.0f;

    // --- Multiply ---

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3R(Int3(pos.x, pos.y, vc), Int2(vld._size.x, vld._size.y));

        vl._weights.multiplyRangeOfRowOHVs(_hiddenCs, vl._visibleActivations, visibleIndex, 1, _hiddenSize.z);
    }

    // --- Find max ---

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3R(Int3(pos.x, pos.y, vc), Int2(vld._size.x, vld._size.y));

        if (vl._visibleActivations[visibleIndex] > maxActivation) {
            maxActivation = vl._visibleActivations[visibleIndex];
            maxIndex = vc;
        }
    }

    vl._reconCs[address2R(pos, _hiddenSize.x)] = maxIndex;
}

void SparseCoder::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, int vli) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2R(pos, _hiddenSize.x);

    // --- Delta Rule ---

    int positiveIndex = (*inputCs[vli])[visibleColumnIndex];
    int negativeIndex = vl._reconCs[visibleColumnIndex];

    vl._weights.deltaOHVRuleOHVs(_hiddenCs, visibleColumnIndex, _hiddenSize.z, vld._size.z, positiveIndex, negativeIndex, _alpha);
}

void SparseCoder::createRandom(ComputeSystem &cs,
    const Int3 &hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        createSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < vl._weights._nonZeroValues.size(); x++)
            init(x, cs._rng, vli);
#else
        runKernel1(cs, std::bind(SparseCoder::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), vl._weights._nonZeroValues.size(), cs._rng, cs._batchSize1);
#endif

        // Generate transpose (needed for reconstruction)
        vl._weights.createT();

        // Visible activations buffer
        vl._visibleActivations = FloatBuffer(numVisible);

        // Reconstruction states
        vl._reconCs = IntBuffer(numVisibleColumns);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Hidden stimulus
    _hiddenStimulus = FloatBuffer(numHidden);

    // Hidden activations
    _hiddenActivations = FloatBuffer(numHidden);
}

void SparseCoder::activate(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Sparse coding iterations: forward, reconstruct, repeat
    for (int it = 0; it < _explainIters; it++) {
        bool firstIter = it == 0;

#ifdef KERNEL_DEBUG
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                forward(Int2(x, y), cs._rng, visibleCs, firstIter);
#else
        runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, firstIter), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_DEBUG
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    backward(Int2(x, y), cs._rng, visibleCs, vli);
#else
            runKernel2(cs, std::bind(SparseCoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }
}

void SparseCoder::learn(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs) {
    // Final reconstruction + learning
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_DEBUG
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                learn(Int2(x, y), cs._rng, visibleCs, vli);
#else
        runKernel2(cs, std::bind(SparseCoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
    }
}

void SparseCoder::writeToStream(std::ostream &os) const {
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

void SparseCoder::readFromStream(std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_explainIters), sizeof(int));

    readBufferFromStream(is, &_hiddenCs);

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
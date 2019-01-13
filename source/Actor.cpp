// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

// Kernels
void Actor::init(int pos, std::mt19937 &rng, int vli) {
    // Randomly initialize weights in range
	std::uniform_real_distribution<float> weightDist(-0.0001f, 0.0001f);

    _visibleLayers[vli]._actionWeights[pos] = weightDist(rng);
}

void Actor::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs) {
    // Cache address calculations (taken from addressN functions)
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * _hiddenSize.z;

    int hiddenColumnIndex = address2(pos, _hiddenSize.x);

    // ------------------------------ Value ------------------------------

    int dPartialValue = pos.x + pos.y * _hiddenSize.x;

    float value = 0.0f;
    float count = 0.0f;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Center of projected position
        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        // Additional addressing dimensions
        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleC = (*inputCs[vli])[address2(visiblePosition, vld._size.x)];

                // Final component of address
                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                value += vl._valueWeights[dPartialValue + az * dxy]; // Used cached parts to compute weight address, equivalent to calling address4
            }

        count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    }

    value /= std::max(1.0f, count);

    _hiddenValues[hiddenColumnIndex] = value;

    // ------------------------------ Action ------------------------------

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    if (dist01(rng) < _epsilon) {
        std::uniform_int_distribution<int> columnDist(0, _hiddenSize.z - 1);

        _hiddenCs[hiddenColumnIndex] = columnDist(rng);
    }
    else {
        int maxIndex = 0;
        float maxActivation = -999999.0f;

        // For each hidden unit
        for (int hc = 0; hc < _hiddenSize.z; hc++) {
            Int3 hiddenPosition(pos.x, pos.y, hc);

            // Partially computed address of weight
            int dPartialAction = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

            float activation = 0.0f;
        
            // For each visible layer
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                // Center of projected position
                Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

                // Lower corner
                Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

                // Additional addressing dimensions
                int diam = vld._radius * 2 + 1;
                int diam2 = diam * diam;

                // Bounds of receptive field, clamped to input size
                Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
                Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

                for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
                    for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                        Int2 visiblePosition(x, y);

                        int visibleC = (*inputCs[vli])[address2(visiblePosition, vld._size.x)];

                        // Final component of address
                        int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                        activation += vl._actionWeights[dPartialAction + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                    }
            }

            activation /= std::max(1.0f, count);

            if (activation > maxActivation) {
                maxActivation = activation;

                maxIndex = hc;
            }
        }

        _hiddenCs[hiddenColumnIndex] = maxIndex;
    }
}

void Actor::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCsPrev, const IntBuffer* hiddenCsPrev, const FloatBuffer* hiddenValuesPrev, float q, float g) {
    // Cache address calculations
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * _hiddenSize.z;

    float valuePrev = 0.0f;
    float count = 0.0f;

    int hiddenColumnIndex = address2(pos, _hiddenSize.x);

    // Partially computed address, this time for action
    int dPartialValue = pos.x + pos.y * _hiddenSize.x;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Center of projected position
        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        // Additional addressing dimensions
        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleCPrev = (*inputCsPrev[vli])[address2(visiblePosition, vld._size.x)];

                // Final component of address
                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev * diam2;

                valuePrev += vl._valueWeights[dPartialValue + az * dxy]; // Used cached parts to compute weight address, equivalent to calling address4
            }

        // Count can be computed outside of loop, this is the value equavilent to count += 1.0f after each value increment
        count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    }

    valuePrev /= std::max(1.0f, count);

    // Temporal difference error
    float newValue = q + g * _hiddenValues[hiddenColumnIndex];

    float tdErrorValue = newValue - valuePrev;
    float tdErrorAction = newValue - (*hiddenValuesPrev)[hiddenColumnIndex];

    // Deltas for value and action
    float alphaTdErrorValue = _alpha * tdErrorValue;
    
    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Center of projected position
        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        // Additional addressing dimensions
        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleCPrev = (*inputCsPrev[vli])[address2(visiblePosition, vld._size.x)];

                // Final component of address
                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev * diam2;

                vl._valueWeights[dPartialValue + az * dxy] += alphaTdErrorValue;
            }
    }

    Int3 hiddenPosition(pos.x, pos.y, (*hiddenCsPrev)[hiddenColumnIndex]);

    int dPartialAction = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

    float betaTdErrorAction = _beta * tdErrorAction;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Center of projected position
        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        // Additional addressing dimensions
        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleCPrev = (*inputCsPrev[vli])[address2(visiblePosition, vld._size.x)];

                // Final component of address
                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev * diam2;

                vl._actionWeights[dPartialAction + az * dxyz] += betaTdErrorAction;
            }
    }
}

void Actor::createRandom(ComputeSystem &cs,
    const Int3 &hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs)
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

        // Projection constant
        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        int diam = vld._radius * 2 + 1;

        int numWeightsPerHidden = diam * diam * vld._size.z;

        int weightsSizeValue = numHiddenColumns * numWeightsPerHidden;
        int weightsSizeAction = numHidden * numWeightsPerHidden;

        // Create weight matrix for this visible layer and initialize randomly
        vl._valueWeights = FloatBuffer(weightsSizeValue);
        vl._actionWeights = FloatBuffer(weightsSizeAction);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < weightsSizeValue; x++)
            fillFloat(x, cs._rng, &vl._valueWeights, 0.0f);
#else
        runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &vl._valueWeights, 0.0f), weightsSizeValue, cs._rng, cs._batchSize1);
#endif

#ifdef KERNEL_DEBUG
        for (int x = 0; x < weightsSizeAction; x++)
            init(x, cs._rng, vli);
#else
        runKernel1(cs, std::bind(Actor::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), weightsSizeAction, cs._rng, cs._batchSize1);
#endif
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Hidden values
    _hiddenValues = FloatBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillFloat(x, cs._rng, &_hiddenValues, 0.0f);
#else
    runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &_hiddenValues, 0.0f), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Create (pre-allocated) history samples
    _historySize = 0;
    _historySamples.resize(historyCapacity);

    for (int i = 0; i < _historySamples.size(); i++) {
        _historySamples[i] = std::make_shared<HistorySample>();

        _historySamples[i]->_visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            _historySamples[i]->_visibleCs[vli] = IntBuffer(numVisibleColumns);
        }

        _historySamples[i]->_hiddenCs = IntBuffer(numHiddenColumns);

        _historySamples[i]->_hiddenValues = FloatBuffer(numHiddenColumns);
    }
}

const Actor &Actor::operator=(const Actor &other) {
    _hiddenSize = other._hiddenSize;

    _historySize = other._historySize;

    _hiddenCs = other._hiddenCs;

    _hiddenValues = other._hiddenValues;

    _visibleLayerDescs = other._visibleLayerDescs;
    _visibleLayers = other._visibleLayers;

    _alpha = other._alpha;
    _beta = other._beta;
    _gamma = other._gamma;

    _historySamples.resize(other._historySamples.size());

    for (int t = 0; t < _historySamples.size(); t++) {
        _historySamples[t] = std::make_shared<HistorySample>();

        HistorySample &s = *_historySamples[t];
        const HistorySample &otherS = *other._historySamples[t];

        s = otherS;
    }

    return *this;
}

void Actor::step(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs, float reward, bool learnEnabled) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, visibleCs);
#else
    runKernel2(cs, std::bind(Actor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    // Add sample
    if (_historySize == _historySamples.size()) {
        // Circular buffer swap
        std::shared_ptr<HistorySample> temp = _historySamples.front();

        for (int i = 0; i < _historySamples.size() - 1; i++)
            _historySamples[i] = _historySamples[i + 1];

        _historySamples.back() = temp;
    }

    // If not at cap, increment
    if (_historySize < _historySamples.size())
        _historySize++;
    
    // Add new sample
    {
        HistorySample &s = *_historySamples[_historySize - 1];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            // Copy visible Cs
#ifdef KERNEL_DEBUG
            for (int x = 0; x < numVisibleColumns; x++)
                copyInt(x, cs._rng, visibleCs[vli], s._visibleCs[vli].get());
#else
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, visibleCs[vli], &s._visibleCs[vli]), numVisibleColumns, cs._rng, cs._batchSize1);
#endif
        }

        // Copy hidden Cs
#ifdef KERNEL_DEBUG
        for (int x = 0; x < numHiddenColumns; x++)
            copyInt(x, cs._rng, &_hiddenCs, s._hiddenCs.get());
#else
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, &s._hiddenCs), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

        // Copy hidden values
#ifdef KERNEL_DEBUG
        for (int x = 0; x < numHiddenColumns; x++)
            copyFloat(x, cs._rng, &_hiddenValues, s._hiddenValues.get());
#else
        runKernel1(cs, std::bind(copyFloat, std::placeholders::_1, std::placeholders::_2, &_hiddenValues, &s._hiddenValues), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

        s._reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && _historySize > 2) {
        const HistorySample &sPrev = *_historySamples[0];

        // Compute (partial) Q value, rest is completed in the kernel
        float q = 0.0f;

        for (int t = _historySize - 1; t >= 1; t--)
            q += _historySamples[t]->_reward * std::pow(_gamma, t - 1);

        // Discount factor for remainder of Q value
        float g = std::pow(_gamma, _historySize - 1);

        // Learn kernel
#ifdef KERNEL_DEBUG
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                learn(Int2(x, y), cs._rng, constGet(sPrev._visibleCs), &sPrev._hiddenCs, &sPrev._hiddenValues, q, g);
#else
        runKernel2(cs, std::bind(Actor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, constGet(sPrev._visibleCs), &sPrev._hiddenCs, &sPrev._hiddenValues, q, g), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
    }
}

void Actor::writeToStream(std::ostream &os) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_epsilon), sizeof(float));

    os.write(reinterpret_cast<const char*>(&_historySize), sizeof(int));

    writeBufferToStream(os, &_hiddenCs);

    writeBufferToStream(os, &_hiddenValues);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        os.write(reinterpret_cast<const char*>(&vl._hiddenToVisible), sizeof(Float2));

        writeBufferToStream(os, &vl._valueWeights);
        writeBufferToStream(os, &vl._actionWeights);
    }

    int numHistorySamples = _historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    for (int t = 0; t < _historySamples.size(); t++) {
        const HistorySample &s = *_historySamples[t];

        for (int vli = 0; vli < _visibleLayers.size(); vli++)
            writeBufferToStream(os, &s._visibleCs[vli]);

        writeBufferToStream(os, &s._hiddenCs);
        writeBufferToStream(os, &s._hiddenValues);

        os.write(reinterpret_cast<const char*>(&s._reward), sizeof(float));
    }
}

void Actor::readFromStream(std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&_epsilon), sizeof(float));

    is.read(reinterpret_cast<char*>(&_historySize), sizeof(int));

    readBufferFromStream(is, &_hiddenCs);

    readBufferFromStream(is, &_hiddenValues);

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

        is.read(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(Float2));

        readBufferFromStream(is, &vl._valueWeights);
        readBufferFromStream(is, &vl._actionWeights);
    }

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    _historySamples.resize(numHistorySamples);

    for (int t = 0; t < _historySamples.size(); t++) {
        _historySamples[t] = std::make_shared<HistorySample>();

        HistorySample &s = *_historySamples[t];

        s._visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++)
            readBufferFromStream(is, &s._visibleCs[vli]);

        readBufferFromStream(is, &s._hiddenCs);
        readBufferFromStream(is, &s._hiddenValues);

        is.read(reinterpret_cast<char*>(&s._reward), sizeof(float));
    }
}
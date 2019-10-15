// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>

using namespace ogmaneo;

void Hierarchy::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    int vli,
    const IntBuffer* inputCs
) {
    if (l == 0) {
        int visibleColumnIndex = address2(pos, Int2(_inputSizes[vli].x, _inputSizes[vli].y));

        float maxValue = -999999.0f;

        if (inputCs == nullptr) {
            for (int vc = 0; vc < _inputSizes[vli].z; vc++) {
                int visibleIndex = address3(Int3(pos.x, pos.y, vc), _inputSizes[vli]);

                float sum;

                if (l == _scLayers.size() - 1)
                    sum = _rLayers[l]._weights[vli].multiplyOHVs(*hiddenCs, visibleIndex, _scLayers[l].getHiddenSize().z);
                else
                    sum = _rLayers[l]._weights[vli].multiplyOHVs(*hiddenCs, _rLayers[l + 1]._activations[0], visibleIndex, _scLayers[l].getHiddenSize().z);

                sum /= std::max(1, _rLayers[l]._visibleCounts[vli][visibleColumnIndex]);
                
                if (sum > maxValue) {
                    maxValue = sum;

                    _actions[vli][visibleColumnIndex] = vc;
                }
            }
        }
        else {
            int visibleIndex = address3(Int3(pos.x, pos.y, (*inputCs)[visibleColumnIndex]), _inputSizes[vli]);

            float sum;

            if (l == _scLayers.size() - 1)
                sum = _rLayers[l]._weights[vli].multiplyOHVs(*hiddenCs, visibleIndex, _scLayers[l].getHiddenSize().z);
            else
                sum = _rLayers[l]._weights[vli].multiplyOHVs(*hiddenCs, _rLayers[l + 1]._activations[0], visibleIndex, _scLayers[l].getHiddenSize().z);

            sum /= std::max(1, _rLayers[l]._visibleCounts[vli][visibleColumnIndex]);
            
            maxValue = sum;
        }

        _rLayers[l]._activations[vli][visibleColumnIndex] = maxValue;
    }
    else {
        int visibleColumnIndex = address2(pos, Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y));

        int inputC = (*inputCs)[visibleColumnIndex];

        int visibleIndex = address3(Int3(pos.x, pos.y, inputC), _scLayers[l - 1].getHiddenSize());

        float sum;

        if (l == _scLayers.size() - 1)
            sum = _rLayers[l]._weights[vli].multiplyOHVs(*hiddenCs, visibleIndex, _scLayers[l].getHiddenSize().z);
        else
            sum = _rLayers[l]._weights[vli].multiplyOHVs(*hiddenCs, _rLayers[l + 1]._activations[0], visibleIndex, _scLayers[l].getHiddenSize().z);

        sum /= std::max(1, _rLayers[l]._visibleCounts[vli][visibleColumnIndex]);
    
        _rLayers[l]._activations[vli][visibleColumnIndex] = sum;
    }
}

void Hierarchy::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, (*hiddenCs)[hiddenColumnIndex]), _scLayers[l].getHiddenSize());

    if (l == 0) {
        //float act = _rLayers[l + 1]._activations[0][hiddenColumnIndex];

        float error = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _rLayers[l]._weights.size(); vli++) {
            if (!_actions[vli].empty())
                error += _rLayers[l]._weights[vli].multiplyOHVsT(*inputCs[vli], _rLayers[l]._errors[vli], hiddenIndex, _inputSizes[vli].z);
        }

        error /= std::max(1, _rLayers[l]._hiddenCounts[hiddenColumnIndex]);

        _rLayers[l + 1]._errors[0][hiddenColumnIndex] = error;
    }
    else {
        //float act = _rLayers[l + 1]._activations[0][hiddenColumnIndex];

        float error = _rLayers[l]._weights[0].multiplyOHVsT(*inputCs[0], _rLayers[l]._errors[0], hiddenIndex, _scLayers[l - 1].getHiddenSize().z);

        error /= std::max(1, _rLayers[l]._hiddenCounts[hiddenColumnIndex]);
        
        _rLayers[l + 1]._errors[0][hiddenColumnIndex] = error;
    }
}

void Hierarchy::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    int vli,
    const IntBuffer* inputCs
) {
    if (l == 0) {
        int visibleColumnIndex = address2(pos, Int2(_inputSizes[vli].x, _inputSizes[vli].y));

        if (_rLayers[l]._errors[vli][visibleColumnIndex] != 0.0f) {
            int visibleIndex = address3(Int3(pos.x, pos.y, (*inputCs)[visibleColumnIndex]), _inputSizes[vli]);

            float delta = _alpha * std::min(1.0f, std::max(-1.0f, _rLayers[l]._errors[vli][visibleColumnIndex]));
            
            if (l == _scLayers.size() - 1)
                _rLayers[l]._weights[vli].deltaOHVs(*hiddenCs, delta, visibleIndex, _scLayers[l].getHiddenSize().z);
            else
                _rLayers[l]._weights[vli].deltaOHVs(*hiddenCs, _rLayers[l + 1]._activations[0], delta, visibleIndex, _scLayers[l].getHiddenSize().z);
        }
    }
    else {
        int visibleColumnIndex = address2(pos, Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y));

        if (_rLayers[l]._errors[vli][visibleColumnIndex] != 0.0f) {
            int inputC = (*inputCs)[visibleColumnIndex];

            int visibleIndex = address3(Int3(pos.x, pos.y, inputC), _scLayers[l - 1].getHiddenSize());

            float delta = _beta * std::min(1.0f, std::max(-1.0f, _rLayers[l]._errors[vli][visibleColumnIndex]));
            
            if (l == _scLayers.size() - 1)
                _rLayers[l]._weights[vli].deltaOHVs(*hiddenCs, delta, visibleIndex, _scLayers[l].getHiddenSize().z);
            else
                _rLayers[l]._weights[vli].deltaOHVs(*hiddenCs, _rLayers[l + 1]._activations[0], delta, visibleIndex, _scLayers[l].getHiddenSize().z);
        }
    }
}

void Hierarchy::initRandom(
    ComputeSystem &cs,
    const std::vector<Int3> &inputSizes,
    const std::vector<InputType> &inputTypes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    _scLayers.resize(layerDescs.size());
    _rLayers.resize(layerDescs.size());

    _ticks.assign(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    _historySizes.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    _updates.resize(layerDescs.size(), false);

    // Cache input sizes
    _inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        std::uniform_real_distribution<float> weightDistLow(-0.01f, 0.01f);
        std::uniform_real_distribution<float> weightDistHigh(0.99f, 1.01f);

        // Histories for all input layers or just the one sparse coder (if not the first layer)
        _histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);

        _historySizes[l].resize(_histories[l].size());
		
        // Create sparse coder visible layer descriptors
        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l]._temporalHorizon);

            _actions.resize(_inputSizes.size());

            _rLayers[l]._weights.resize(_inputSizes.size());
            _rLayers[l]._visibleCounts.resize(_inputSizes.size());

            _rLayers[l]._activations.resize(_inputSizes.size());
            _rLayers[l]._errors.resize(_inputSizes.size());

            _rLayers[l]._hiddenCounts = IntBuffer(layerDescs[l]._hiddenSize.x * layerDescs[l]._hiddenSize.y, 0);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = t + layerDescs[l]._temporalHorizon * i;

                    scVisibleLayerDescs[index]._size = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = layerDescs[l]._scRadius;
                }

                if (inputTypes[i] == InputType::_act) {
                    _actions[i] = IntBuffer(inputSizes[i].x * inputSizes[i].y, 0);
                    _rLayers[l]._activations[i] = FloatBuffer(inputSizes[i].x * inputSizes[i].y, 0.0f);
                    _rLayers[l]._errors[i] = FloatBuffer(inputSizes[i].x * inputSizes[i].y, 0.0f);

                    initSMLocalRF(layerDescs[l]._hiddenSize, inputSizes[i], layerDescs[l]._rRadius, _rLayers[l]._weights[i]);

                    _rLayers[l]._weights[i].initT();

                    // Init weights
                    for (int j = 0; j < _rLayers[l]._weights[i]._nonZeroValues.size(); j++)
                        _rLayers[l]._weights[i]._nonZeroValues[j] = weightDistLow(cs._rng);

                    for (int x = 0; x < layerDescs[l]._hiddenSize.x; x++)
                        for (int y = 0; y < layerDescs[l]._hiddenSize.y; y++) {
                            int hiddenColumnIndex = address2(Int2(x, y), Int2(layerDescs[l]._hiddenSize.x, layerDescs[l]._hiddenSize.y));
                            int hiddenIndex = address3(Int3(x, y, 0), layerDescs[l]._hiddenSize);

                            _rLayers[l]._hiddenCounts[hiddenColumnIndex] += _rLayers[l]._weights[i].countT(hiddenIndex) / inputSizes[i].z;
                        }

                    _rLayers[l]._visibleCounts[i] = IntBuffer(_actions[i].size());

                    for (int x = 0; x < inputSizes[i].x; x++)
                        for (int y = 0; y < inputSizes[i].y; y++) {
                            int visibleColumnIndex = address2(Int2(x, y), Int2(inputSizes[i].x, inputSizes[i].y));
                            int visibleIndex = address3(Int3(x, y, 0), inputSizes[i]);

                            _rLayers[l]._visibleCounts[i][visibleColumnIndex] = _rLayers[l]._weights[i].count(visibleIndex) / layerDescs[l]._hiddenSize.z;
                        }    
                }
            }
            
            // Initialize history buffers
			for (int v = 0; v < _histories[l].size(); v++) {
				int i = v / layerDescs[l]._temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				_histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                _historySizes[l][v] = inSize;
			}
        }
        else {
            scVisibleLayerDescs.resize(layerDescs[l]._temporalHorizon);

            _rLayers[l]._weights.resize(1);
            _rLayers[l]._visibleCounts.resize(1);

            _rLayers[l]._activations.resize(1);
            _rLayers[l]._errors.resize(1);

            _rLayers[l]._hiddenCounts = IntBuffer(layerDescs[l]._hiddenSize.x * layerDescs[l]._hiddenSize.y, 0);

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                scVisibleLayerDescs[t]._size = layerDescs[l - 1]._hiddenSize;
                scVisibleLayerDescs[t]._radius = layerDescs[l]._scRadius;
            }

            _rLayers[l]._activations[0] = FloatBuffer(layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y, 0.0f);
            _rLayers[l]._errors[0] = FloatBuffer(layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y, 0.0f);

            initSMLocalRF(layerDescs[l]._hiddenSize, layerDescs[l - 1]._hiddenSize, layerDescs[l]._rRadius, _rLayers[l]._weights[0]);

            _rLayers[l]._weights[0].initT();

            // Init weights
            for (int j = 0; j < _rLayers[l]._weights[0]._nonZeroValues.size(); j++)
                _rLayers[l]._weights[0]._nonZeroValues[j] = weightDistHigh(cs._rng);

            for (int x = 0; x < layerDescs[l]._hiddenSize.x; x++)
                for (int y = 0; y < layerDescs[l]._hiddenSize.y; y++) {
                    int hiddenColumnIndex = address2(Int2(x, y), Int2(layerDescs[l]._hiddenSize.x, layerDescs[l]._hiddenSize.y));
                    int hiddenIndex = address3(Int3(x, y, 0), layerDescs[l]._hiddenSize);

                    _rLayers[l]._hiddenCounts[hiddenColumnIndex] += _rLayers[l]._weights[0].countT(hiddenIndex) / layerDescs[l - 1]._hiddenSize.z;
                }

            _rLayers[l]._visibleCounts[0] = IntBuffer(layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y);

            for (int x = 0; x < layerDescs[l - 1]._hiddenSize.x; x++)
                for (int y = 0; y < layerDescs[l - 1]._hiddenSize.y; y++) {
                    int visibleColumnIndex = address2(Int2(x, y), Int2(layerDescs[l - 1]._hiddenSize.x, layerDescs[l - 1]._hiddenSize.y));
                    int visibleIndex = address3(Int3(x, y, 0), layerDescs[l - 1]._hiddenSize);

                    _rLayers[l]._visibleCounts[0][visibleColumnIndex] = _rLayers[l]._weights[0].count(visibleIndex) / layerDescs[l]._hiddenSize.z;
                }

            int inSize = layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y;

			for (int v = 0; v < _histories[l].size(); v++) {
                _histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                _historySizes[l][v] = inSize;
            }
        }

        // Create the sparse coding layer
        _scLayers[l].initRandom(cs, layerDescs[l]._hiddenSize, scVisibleLayerDescs);
    }
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    _scLayers = other._scLayers;
    _rLayers = other._rLayers;

    _qs = other._qs;
    _actions = other._actions;

    _historySizes = other._historySizes;
    _updates = other._updates;
    _ticks = other._ticks;
    _ticksPerUpdate = other._ticksPerUpdate;
    _inputSizes = other._inputSizes;

    _histories.resize(other._histories.size());

    for (int l = 0; l < _scLayers.size(); l++) {
        _histories[l].resize(other._histories[l].size());

        for (int v = 0; v < _histories[l].size(); v++) {
            _histories[l][v] = std::make_shared<IntBuffer>();
            
            (*_histories[l][v]) = (*other._histories[l][v]);
        }
    }

    _alpha = other._alpha;
    _beta = other._beta;
    _gamma = other._gamma;
    _maxHistorySamples = other._maxHistorySamples;
    _historyIters = other._historyIters;

    _historySamples = other._historySamples;

    return *this;
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    float reward,
    bool learnEnabled
) {
    assert(inputCs.size() == _inputSizes.size());

    // First tick is always 0
    _ticks[0] = 0;

    // Add input to first layer history   
    {
        int temporalHorizon = _histories.front().size() / _inputSizes.size();

        std::vector<std::shared_ptr<IntBuffer>> lasts(_inputSizes.size());
        
        for (int i = 0; i < _inputSizes.size(); i++)
            lasts[i] = _histories.front()[temporalHorizon - 1 + temporalHorizon * i];
  
        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int i = 0; i < _inputSizes.size(); i++) {
                // Shift
                _histories.front()[t + temporalHorizon * i] = _histories.front()[(t - 1) + temporalHorizon * i];
            }
        }

        for (int i = 0; i < _inputSizes.size(); i++) {
            assert(_inputSizes[i].x * _inputSizes[i].y == inputCs[i]->size());
            
            // Copy
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < inputCs[i]->size(); x++)
                copyInt(x, cs._rng, inputCs[i], lasts[i].get());
#else
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[i], lasts[i].get()), inputCs[i]->size(), cs._rng, cs._batchSize1);
#endif

            _histories.front()[0 + temporalHorizon * i] = lasts[i];
        }
    }

    // Set all updates to no update, will be set to true if an update occurred later
    _updates.clear();
    _updates.resize(_scLayers.size(), false);

    HistorySample ns;

    ns._states.resize(_scLayers.size());
    ns._reward = reward;

    // Forward
    for (int l = 0; l < _scLayers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            // Reset tick
            _ticks[l] = 0;

            // Updated
            _updates[l] = true;
            
            // Activate sparse coder
            _scLayers[l].step(cs, constGet(_histories[l]), learnEnabled);

            // Add to next layer's history
            if (l < _scLayers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = _histories[lNext].size();

                std::shared_ptr<IntBuffer> last = _histories[lNext].back();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    _histories[lNext][t] = _histories[lNext][t - 1];

                // Copy
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _scLayers[l].getHiddenCs().size(); x++)
                    copyInt(x, cs._rng, &_scLayers[l].getHiddenCs(), last.get());
#else
                runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_scLayers[l].getHiddenCs(), last.get()), _scLayers[l].getHiddenCs().size(), cs._rng, cs._batchSize1);
#endif

                _histories[lNext].front() = last;

                _ticks[lNext]++;
            }
        }

        ns._states[l] = _scLayers[l].getHiddenCs();
    }

    // Action into replay buffer
    ns._actionsPrev.resize(_actions.size());

    for (int vli = 0; vli < _actions.size(); vli++)
        if (!_actions[vli].empty())
            ns._actionsPrev[vli] = *inputCs[vli];

    // Add history sample
    _historySamples.insert(_historySamples.begin(), ns);

    if (_historySamples.size() > _maxHistorySamples)
        _historySamples.resize(_maxHistorySamples);

    // Learn
    if (learnEnabled && _historySamples.size() > 2) {
        // Find latest Q on-policy
        const HistorySample &s = _historySamples[1];
        const HistorySample &sNext = _historySamples[0];
            
        for (int l = _scLayers.size() - 1; l >= 0; l--) {
            if (l == 0) {
                for (int i = 0; i < _inputSizes.size(); i++) {
                    if (_actions[i].empty())
                        continue;

#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _inputSizes[i].x; x++)
                        for (int y = 0; y < _inputSizes[i].y; y++)
                            forward(Int2(x, y), cs._rng, &s._states[l], l, i, &sNext._actionsPrev[i]);
#else
                    runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, i, &sNext._actionsPrev[i]), Int2(_inputSizes[i].x, _inputSizes[i].y), cs._rng, cs._batchSize2);
#endif
                }
            }
            else {
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
                    for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                        forward(Int2(x, y), cs._rng, &s._states[l], l, 0, &s._states[l - 1]);
#else
                runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, 0, &s._states[l - 1]), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
            }
        }

        // Keep predicted Q values
        _qs = _rLayers.front()._activations;

        std::uniform_int_distribution<int> sampleDist(2, _historySamples.size() - 1);

        for (int it = 0; it < _historyIters; it++) {
            int t = sampleDist(cs._rng);

            const HistorySample &s = _historySamples[t];
            const HistorySample &sNext = _historySamples[t - 1];

            for (int l = _scLayers.size() - 1; l >= 0; l--) {
                if (l == 0) {
                    for (int i = 0; i < _inputSizes.size(); i++) {
                        if (_actions[i].empty())
                            continue;

#ifdef KERNEL_NOTHREAD
                        for (int x = 0; x < _inputSizes[i].x; x++)
                            for (int y = 0; y < _inputSizes[i].y; y++)
                                forward(Int2(x, y), cs._rng, &s._states[l], l, i, &sNext._actionsPrev[i]);
#else
                        runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, i, &sNext._actionsPrev[i]), Int2(_inputSizes[i].x, _inputSizes[i].y), cs._rng, cs._batchSize2);
#endif
                    }
                }
                else {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                            forward(Int2(x, y), cs._rng, &s._states[l], l, 0, &s._states[l - 1]);
#else
                    runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, 0, &s._states[l - 1]), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
            }

            // Determine target
            float baseQ = 0.0f;
            float g = 1.0f;
            
            for (int t2 = t - 1; t2 >= 1; t2--) {
                baseQ += g * _historySamples[t2]._reward;

                g *= _gamma;
            }

            // Errors
            for (int vli = 0; vli < _rLayers.front()._errors.size(); vli++)
                for (int i = 0; i < _rLayers.front()._errors[vli].size(); i++) {
                    float targetQ = baseQ + g * _qs[vli][i];

                    _rLayers.front()._errors[vli][i] = targetQ - _rLayers.front()._activations[vli][i];
                }

            // Backward
            for (int l = 0; l < _scLayers.size() - 1; l++) {
                if (l == 0) {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            backward(Int2(x, y), cs._rng, &s._states[l], l, constGet(sNext._actionsPrev));
#else
                    runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, constGet(sNext._actionsPrev)), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
                else {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            backward(Int2(x, y), cs._rng, &s._states[l], l, std::vector<const IntBuffer*>{ &s._states[l - 1] });
#else
                    runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, std::vector<const IntBuffer*>{ &s._states[l - 1] }), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
            }

            // Learn
            for (int l = _scLayers.size() - 1; l >= 0; l--) {
                if (l == 0) {
                    for (int i = 0; i < _inputSizes.size(); i++) {
                        if (_actions[i].empty())
                            continue;

#ifdef KERNEL_NOTHREAD
                        for (int x = 0; x < _inputSizes[i].x; x++)
                            for (int y = 0; y < _inputSizes[i].y; y++)
                                learn(Int2(x, y), cs._rng, &s._states[l], l, i, &sNext._actionsPrev[i]);
#else
                        runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, i, &sNext._actionsPrev[i]), Int2(_inputSizes[i].x, _inputSizes[i].y), cs._rng, cs._batchSize2);
#endif
                    }
                }
                else {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                            learn(Int2(x, y), cs._rng, &s._states[l], l, 0, &s._states[l - 1]);
#else
                    runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, 0, &s._states[l - 1]), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
            }
        }
    }

    for (int l = _scLayers.size() - 1; l >= 0; l--) {
        if (l == 0) {
            for (int i = 0; i < _inputSizes.size(); i++) {
                if (_actions[i].empty())
                    continue;
                    
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _inputSizes[i].x; x++)
                    for (int y = 0; y < _inputSizes[i].y; y++)
                        forward(Int2(x, y), cs._rng, &ns._states[l], l, i, nullptr);
#else
                runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns._states[l], l, i, nullptr), Int2(_inputSizes[i].x, _inputSizes[i].y), cs._rng, cs._batchSize2);
#endif
            }
        }
        else {
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
                for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                    forward(Int2(x, y), cs._rng, &ns._states[l], l, 0, &ns._states[l - 1]);
#else
            runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns._states[l], l, 0, &ns._states[l - 1]), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = _scLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = _inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    os.write(reinterpret_cast<const char*>(_updates.data()), _updates.size() * sizeof(char));
    os.write(reinterpret_cast<const char*>(_ticks.data()), _ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    for (int vli = 0; vli < _actions.size(); vli++)
        writeBufferToStream(os, &_actions[vli]);

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = _historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < _historySizes[l].size(); i++)
            writeBufferToStream(os, _histories[l][i].get());

        _scLayers[l].writeToStream(os);

        writeBufferToStream(os, &_rLayers[l]._hiddenCounts);

        for (int v = 0; v < _rLayers[l]._weights.size(); v++) {
            char exists = !_rLayers[l]._weights[v]._nonZeroValues.empty();

            os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

            if (exists) {
                writeSMToStream(os, _rLayers[l]._weights[v]);
                writeBufferToStream(os, &_rLayers[l]._visibleCounts[v]);
                        
                writeBufferToStream(os, &_rLayers[l]._activations[v]);
                writeBufferToStream(os, &_rLayers[l]._errors[v]);
            }
        }
    }

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_maxHistorySamples), sizeof(int));
    os.write(reinterpret_cast<const char*>(&_historyIters), sizeof(int));

    int numHistorySamples = _historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    // History samples
    for (int t = 0; t < numHistorySamples; t++) {
        const HistorySample &s = _historySamples[t];

        for (int l = 0; l < _scLayers.size(); l++)
            writeBufferToStream(os, &s._states[l]);

        for (int vli = 0; vli < _actions.size(); vli++)
            if (!_actions[vli].empty())
                writeBufferToStream(os, &s._actionsPrev[vli]);
        
        os.write(reinterpret_cast<const char*>(&s._reward), sizeof(float));
    }
}

void Hierarchy::readFromStream(
    std::istream &is
) {
    int numLayers;
    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    int numInputs;
    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));

    _inputSizes.resize(numInputs);

    is.read(reinterpret_cast<char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    _scLayers.resize(numLayers);
    _rLayers.resize(numLayers);

    _ticks.resize(numLayers);

    _histories.resize(numLayers);
    _historySizes.resize(numLayers);
    
    _ticksPerUpdate.resize(numLayers);

    _updates.resize(numLayers);

    is.read(reinterpret_cast<char*>(_updates.data()), _updates.size() * sizeof(char));
    is.read(reinterpret_cast<char*>(_ticks.data()), _ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    _actions.resize(_inputSizes.size());

    for (int vli = 0; vli < _actions.size(); vli++)
        readBufferFromStream(is, &_actions[vli]);

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;
        
        is.read(reinterpret_cast<char*>(&numHistorySizes), sizeof(int));
        _historySizes[l].resize(numHistorySizes);
        is.read(reinterpret_cast<char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        _histories[l].resize(numHistorySizes);

        for (int i = 0; i < _historySizes[l].size(); i++) {
            _histories[l][i] = std::make_shared<IntBuffer>();

            readBufferFromStream(is, _histories[l][i].get());
        }

        _scLayers[l].readFromStream(is);

        if (l == 0) {
            _rLayers[l]._weights.resize(_inputSizes.size());
            _rLayers[l]._visibleCounts.resize(_inputSizes.size());

            _rLayers[l]._activations.resize(_inputSizes.size());
            _rLayers[l]._errors.resize(_inputSizes.size());
        }
        else {
            _rLayers[l]._weights.resize(1);
            _rLayers[l]._visibleCounts.resize(1);

            _rLayers[l]._activations.resize(1);
            _rLayers[l]._errors.resize(1);
        }

        readBufferFromStream(is, &_rLayers[l]._hiddenCounts);

        for (int v = 0; v < _rLayers[l]._weights.size(); v++) {
            char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(char));

            if (exists) {
                readSMFromStream(is, _rLayers[l]._weights[v]);
                readBufferFromStream(is, &_rLayers[l]._visibleCounts[v]);

                readBufferFromStream(is, &_rLayers[l]._activations[v]);
                readBufferFromStream(is, &_rLayers[l]._errors[v]);
            }
        }
    }

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&_maxHistorySamples), sizeof(int));
    is.read(reinterpret_cast<char*>(&_historyIters), sizeof(int));

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    _historySamples.resize(numHistorySamples);

    // History samples
    for (int t = 0; t < numHistorySamples; t++) {
        HistorySample &s = _historySamples[t];

        s._states.resize(_scLayers.size());

        for (int l = 0; l < _scLayers.size(); l++)
            readBufferFromStream(is, &s._states[l]);

        s._actionsPrev.resize(_actions.size());

        for (int vli = 0; vli < _actions.size(); vli++)
            if (!_actions[vli].empty())
                readBufferFromStream(is, &s._actionsPrev[vli]);
        
        is.read(reinterpret_cast<char*>(&s._reward), sizeof(float));
    }
}
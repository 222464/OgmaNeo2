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
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2C(pos, Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y));

    int hiddenIndex = address3C(Int3(pos.x, pos.y, (*hiddenCs)[hiddenColumnIndex]), _scLayers[l].getHiddenSize());

    if (l == 0) {
        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _rLayers[l]._weights.size(); vli++) {
            if (!_rLayers[l]._weights[vli]._nonZeroValues.empty())
                sum += _rLayers[l]._weights[vli].multiplyOHVs(*inputCs[vli], hiddenIndex, _inputSizes[vli].z);
        }

        _rLayers[l]._activations[hiddenColumnIndex] = sum / std::max(1, _rLayers[l]._hiddenCounts[hiddenColumnIndex]);
    }
    else
        _rLayers[l]._activations[hiddenColumnIndex] = _rLayers[l]._weights[0].multiplyScalarOHVs(*inputCs[0], _rLayers[l - 1]._activations, hiddenIndex, _scLayers[l - 1].getHiddenSize().z) / std::max(1, _rLayers[l]._hiddenCounts[hiddenColumnIndex]);
}

void Hierarchy::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    int vli,
    const IntBuffer* inputCs
) {
    if (l == 0) {
        int visibleColumnIndex = address2C(pos, Int2(_inputSizes[vli].x, _inputSizes[vli].y));

        if (!_rLayers[l]._weights[vli]._nonZeroValues.empty()) {
            std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

            if (dist01(rng) < _epsilon) {
                std::uniform_int_distribution<int> actionDist(0, _inputSizes[vli].z - 1);

                _actions[vli][visibleColumnIndex] = actionDist(rng);
            }
            else {
                float maxValue = -999999.0f;

                for (int vc = 0; vc < _inputSizes[vli].z; vc++) {
                    int visibleIndex = address3C(Int3(pos.x, pos.y, vc), _inputSizes[vli]);

                    float sum = _rLayers[l]._weights[vli].multiplyScalarOHVsT(*hiddenCs, _rLayers[l]._errors, visibleIndex, _scLayers[l].getHiddenSize().z) / std::max(1, _rLayers[l]._visibleCounts[vli][visibleColumnIndex]);
                
                    if (sum > maxValue) {
                        maxValue = sum;

                        _actions[vli][visibleColumnIndex] = vc;
                    }
                }
            }
        }
    }
    else {
        int visibleColumnIndex = address2C(pos, Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y));

        int inputC = (*inputCs)[visibleColumnIndex];

        int visibleIndex = address3C(Int3(pos.x, pos.y, inputC), _scLayers[l - 1].getHiddenSize());

        _rLayers[l - 1]._errors[visibleColumnIndex] = _rLayers[l]._weights[vli].multiplyScalarOHVsT(*hiddenCs, _rLayers[l]._errors, visibleIndex, _scLayers[l].getHiddenSize().z) / std::max(1, _rLayers[l]._visibleCounts[vli][visibleColumnIndex]);
    }
}

void Hierarchy::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2C(pos, Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y));

    int hiddenIndex = address3C(Int3(pos.x, pos.y, (*hiddenCs)[hiddenColumnIndex]), _scLayers[l].getHiddenSize());

    float delta = _beta * _rLayers[l]._errors[hiddenColumnIndex];

    if (l == 0) {
        // For each visible layer
        for (int vli = 0; vli < _rLayers[l]._weights.size(); vli++) {
            if (!_rLayers[l]._weights[vli]._nonZeroValues.empty())
                _rLayers[l]._weights[vli].deltaOHVs(*inputCs[vli], delta, hiddenIndex, _inputSizes[vli].z);
        }
    }
    else
        _rLayers[l]._weights[0].deltaScalarOHVs(*inputCs[0], _rLayers[l - 1]._activations, delta, hiddenIndex, _scLayers[l - 1].getHiddenSize().z);
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

    std::uniform_real_distribution<float> noiseDist(-0.0001f, 0.0001f);

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
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

            _rLayers[l]._hiddenCounts = IntBuffer(layerDescs[l]._hiddenSize.x * layerDescs[l]._hiddenSize.y, 0);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = t + layerDescs[l]._temporalHorizon * i;

                    scVisibleLayerDescs[index]._size = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = layerDescs[l]._scRadius;
                }

                if (inputTypes[i] == InputType::_act) {
                    _actions[i] = IntBuffer(inputSizes[i].x * inputSizes[i].y, 0);

                    initSMLocalRF(inputSizes[i], layerDescs[l]._hiddenSize, layerDescs[l]._rRadius, _rLayers[l]._weights[i]);

                    _rLayers[l]._weights[i].initT();

                    // Init weights
                    for (int j = 0; j < _rLayers[l]._weights[i]._nonZeroValues.size(); j++)
                        _rLayers[l]._weights[i]._nonZeroValues[j] = 1.0f + noiseDist(cs._rng);

                    for (int j = 0; j < _rLayers[l]._hiddenCounts.size(); j++)
                        _rLayers[l]._hiddenCounts[j] += _rLayers[l]._weights[i].counts(j * layerDescs[l]._hiddenSize.z) / inputSizes[i].z;

                    _rLayers[l]._visibleCounts[i] = IntBuffer(_actions.size());

                    for (int j = 0; j < _rLayers[l]._visibleCounts[i].size(); j++)
                        _rLayers[l]._visibleCounts[i][j] = _rLayers[l]._weights[i].countsT(j * inputSizes[i].z) / layerDescs[l]._hiddenSize.z;
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

            _rLayers[l]._hiddenCounts = IntBuffer(layerDescs[l]._hiddenSize.x * layerDescs[l]._hiddenSize.y, 0);

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                scVisibleLayerDescs[t]._size = layerDescs[l - 1]._hiddenSize;
                scVisibleLayerDescs[t]._radius = layerDescs[l]._scRadius;
            }

            initSMLocalRF(layerDescs[l - 1]._hiddenSize, layerDescs[l]._hiddenSize, layerDescs[l]._rRadius, _rLayers[l]._weights[0]);

            _rLayers[l]._weights[0].initT();

            // Init weights
            for (int j = 0; j < _rLayers[l]._weights[0]._nonZeroValues.size(); j++)
                _rLayers[l]._weights[0]._nonZeroValues[j] = 1.0f + noiseDist(cs._rng);

            for (int j = 0; j < _rLayers[l]._hiddenCounts.size(); j++)
                _rLayers[l]._hiddenCounts[j] += _rLayers[l]._weights[0].counts(j * layerDescs[l]._hiddenSize.z) / layerDescs[l - 1]._hiddenSize.z;

            _rLayers[l]._visibleCounts[0] = IntBuffer(layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y);

            for (int j = 0; j < _rLayers[l]._visibleCounts[0].size(); j++)
                _rLayers[l]._visibleCounts[0][j] = _rLayers[l]._weights[0].countsT(j * layerDescs[l - 1]._hiddenSize.z) / layerDescs[l]._hiddenSize.z;

            int inSize = layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y;

			for (int v = 0; v < _histories[l].size(); v++) {
                _histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                _historySizes[l][v] = inSize;
            }
        }

        _rLayers[l]._activations = FloatBuffer(layerDescs[l]._hiddenSize.x * layerDescs[l]._hiddenSize.y, 1.0f);
        _rLayers[l]._errors = FloatBuffer(_rLayers[l]._activations.size(), 0.0f);
		
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

    _beta = other._beta;
    _gamma = other._gamma;
    _epsilon = other._epsilon;
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

    // Determine actions by setting errors to 1 and propagating
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _rLayers.back()._errors.size(); x++)
        fillFloat(x, cs._rng, &_rLayers.back()._errors, 1.0f);
#else
    runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &_rLayers.back()._errors, 1.0f), _rLayers.back()._errors.size(), cs._rng, cs._batchSize1);
#endif

    for (int l = _scLayers.size() - 1; l >= 0; l--) {
        if (l == 0) {
            for (int i = 0; i < _inputSizes.size(); i++) {
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _inputSizes[i].x; x++)
                    for (int y = 0; y < _inputSizes[i].y; y++)
                        backward(Int2(x, y), cs._rng, &ns._states[l], l, i, nullptr);
#else
                runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns._states[l], l, i, nullptr), Int2(_inputSizes[i].x, _inputSizes[i].y), cs._rng, cs._batchSize2);
#endif
            }
        }
        else {
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
                for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                    backward(Int2(x, y), cs._rng, &ns._states[l], l, 0, &ns._states[l - 1]);
#else
            runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns._states[l], l, 0, &ns._states[l - 1]), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
        }
    }

    // Action into replay buffer
    ns._actions = _actions;

    // Add history sample
    _historySamples.insert(_historySamples.begin(), ns);

    if (_historySamples.size() > _maxHistorySamples)
        _historySamples.resize(_maxHistorySamples);

    // Determine Q
    for (int l = 0; l < _scLayers.size(); l++) {
        if (l == 0) {
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                    forward(Int2(x, y), cs._rng, &ns._states[l], l, constGet(ns._actions));
#else
            runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns._states[l], l, constGet(ns._actions)), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
        }
        else {
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                    forward(Int2(x, y), cs._rng, &ns._states[l], l, std::vector<const IntBuffer*>{ &ns._states[l - 1] });
#else
            runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns._states[l], l, std::vector<const IntBuffer*>{ &ns._states[l - 1] }), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
        }
    }

    // Keep predicted Q values
    _q = _rLayers.back()._activations;

    // Learn
    if (learnEnabled && _historySamples.size() > 1) {
        std::uniform_int_distribution<int> sampleDist(0, _historySamples.size() - 1);

        for (int it = 0; it < _historyIters; it++) {
            int t = sampleDist(cs._rng);

            const HistorySample &s = _historySamples[t];

            // Forward
            for (int l = 0; l < _scLayers.size(); l++) {
                if (l == 0) {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            forward(Int2(x, y), cs._rng, &s._states[l], l, constGet(s._actions));
#else
                    runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, constGet(s._actions)), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
                else {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            forward(Int2(x, y), cs._rng, &s._states[l], l, std::vector<const IntBuffer*>{ &s._states[l - 1] });
#else
                    runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, std::vector<const IntBuffer*>{ &s._states[l - 1] }), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
            }

            // Errors
            for (int i = 0; i < _rLayers.back()._errors.size(); i++) {
                // Determine target
                float targetQ = 0.0f;
                float g = 1.0f;
                
                for (int t2 = t - 1; t2 >= 0; t2--) {
                    targetQ += g * _historySamples[t2]._reward;

                    g *= _gamma;
                }

                targetQ += g * _q[i];

                _rLayers.back()._errors[i] = targetQ - _rLayers.back()._activations[i];
            }

            // Backward
            for (int l = _scLayers.size() - 1; l >= 1; l--) {
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
                    for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                        backward(Int2(x, y), cs._rng, &s._states[l], l, 0, &s._states[l - 1]);
#else
                runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, 0, &s._states[l - 1]), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
            }

            // learn
            for (int l = 0; l < _scLayers.size(); l++) {
                if (l == 0) {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            learn(Int2(x, y), cs._rng, &s._states[l], l, constGet(s._actions));
#else
                    runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, constGet(s._actions)), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
                else {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            learn(Int2(x, y), cs._rng, &s._states[l], l, std::vector<const IntBuffer*>{ &s._states[l - 1] });
#else
                    runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, &s._states[l], l, std::vector<const IntBuffer*>{ &s._states[l - 1] }), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
            }
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

    writeBufferToStream(os, &_actions);

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = _historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < _historySizes[l].size(); i++)
            writeBufferToStream(os, _histories[l][i].get());

        _scLayers[l].writeToStream(os);

        writeBufferToStream(os, &_rLayers[l]._activations);
        writeBufferToStream(os, &_rLayers[l]._errors);
        writeBufferToStream(os, &_rLayers[l]._hiddenCounts);

        for (int v = 0; v < _rLayers[l]._weights.size(); v++) {
            char exists = !_rLayers[l]._weights[v]._nonZeroValues.empty();

            os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

            if (exists) {
                writeSMToStream(os, _rLayers[l]._weights[v]);
                writeBufferToStream(os, &_rLayers[l]._visibleCounts[v]);
            }
        }
    }

    os.write(reinterpret_cast<const char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_epsilon), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_maxHistorySamples), sizeof(int));
    os.write(reinterpret_cast<const char*>(&_historyIters), sizeof(int));

    int numHistorySamples = _historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    // History samples
    for (int t = 0; t < numHistorySamples; t++) {
        const HistorySample &s = _historySamples[t];

        for (int l = 0; l < _scLayers.size(); l++)
            writeBufferToStream(os, &s._states[l]);
        
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

    readBufferFromStream(is, &_actions);

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

        readBufferFromStream(is, &_rLayers[l]._activations);
        readBufferFromStream(is, &_rLayers[l]._errors);
        readBufferFromStream(is, &_rLayers[l]._hiddenCounts);

        for (int v = 0; v < _rLayers[l]._weights.size(); v++) {
            char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(char));

            if (exists) {
                readSMFromStream(is, _rLayers[l]._weights[v]);
                readBufferFromStream(is, &_rLayers[l]._visibleCounts);
            }
        }
    }

    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&_epsilon), sizeof(float));
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
        
        is.read(reinterpret_cast<char*>(&s._reward), sizeof(float));
    }
}
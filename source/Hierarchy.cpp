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
    const std::vector<IntBuffer> &hiddenStates,
    std::vector<IntBuffer> &actions,
    int l,
    int a,
    bool determineActions
) {
    if (l == _rLayers.size()) { // Last layers
        int maxIndex = 0;
        float maxActivation = -999999.0f;

        int hiddenColumnIndex = address2C(pos, Int2(_actionSizes[a].x, _actionSizes[a].y));

        if (determineActions) {
            for (int hc = 0; hc < _actionSizes[a].z; hc++) {
                int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _actionSizes[a]);

                RouteLayer &r = _actionLayers[a];
        
                float sum = (l == 0 ? r._weights.multiplyOHVs(hiddenStates[l], hiddenIndex, _scLayers[l].getHiddenSize().z) :
                    r._weights.multiplyScalarOHVs(hiddenStates[l], _rLayers[l - 1]._activations, hiddenIndex, _scLayers[l].getHiddenSize().z));

                sum /= std::max(1, r._hiddenCounts[hiddenColumnIndex]);

                if (sum > maxActivation) {
                    maxActivation = sum;

                    maxIndex = hc;
                }
            }

            _actions[a][hiddenColumnIndex] = maxIndex;
        }
        else { // Partial recompute
            int hiddenIndex = address3C(Int3(pos.x, pos.y, actions[a][hiddenColumnIndex]), _actionSizes[a]);

            RouteLayer &r = _actionLayers[a];
    
            float sum = (l == 0 ? r._weights.multiplyOHVs(hiddenStates[l], hiddenIndex, _scLayers[l].getHiddenSize().z) :
                r._weights.multiplyScalarOHVs(hiddenStates[l], _rLayers[l - 1]._activations, hiddenIndex, _scLayers[l].getHiddenSize().z));

            r._activations[hiddenColumnIndex] = sum / std::max(1, r._hiddenCounts[hiddenColumnIndex]);
        }
    }
    else { // Hidden layer
        int lNext = l + 1;

        int hiddenColumnIndex = address2C(pos, Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y));
    
        RouteLayer &r = _rLayers[l];

        int hiddenIndex = address3C(Int3(pos.x, pos.y, hiddenStates[lNext][hiddenColumnIndex]), _scLayers[lNext].getHiddenSize());
    
        float sum = (l == 0 ? r._weights.multiplyOHVs(hiddenStates[l], hiddenIndex, _scLayers[l].getHiddenSize().z) :
            r._weights.multiplyScalarOHVs(hiddenStates[l], _rLayers[l - 1]._activations, hiddenIndex, _scLayers[l].getHiddenSize().z));

        r._activations[hiddenColumnIndex] = sum / std::max(1, r._hiddenCounts[hiddenColumnIndex]);
    }
}

void Hierarchy::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<IntBuffer> &hiddenStates,
    const std::vector<IntBuffer> &actions,
    int l
) {
    int lNext = l + 1;

    if (l == _rLayers.size() - 1) {
        RouteLayer &r = _rLayers[l];

        int visibleColumnIndex = address2C(pos, Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y));

        int visibleIndex = address3C(Int3(pos.x, pos.y, hiddenStates[l][visibleColumnIndex]), _scLayers[l].getHiddenSize());

        r._errors[visibleColumnIndex] = 0.0f;

        for (int a = 0; a < _actionLayers.size(); a++)
            r._errors[visibleColumnIndex] += _rLayers[l]._weights.multiplyScalarOHVsT(actions[a], _rLayers[lNext]._errors, visibleIndex, _actionSizes[a].z);

        r._errors[visibleColumnIndex] /= std::max(1, _rLayers[lNext]._visibleCounts[visibleColumnIndex]) * _actionLayers.size();
    }
    else {
        RouteLayer &r = _rLayers[l];

        int visibleColumnIndex = address2C(pos, Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y));

        int visibleIndex = address3C(Int3(pos.x, pos.y, hiddenStates[l][visibleColumnIndex]), _scLayers[l].getHiddenSize());

        float sum = _rLayers[l]._weights.multiplyScalarOHVsT(hiddenStates[lNext], _rLayers[lNext]._errors, visibleIndex, _scLayers[l].getHiddenSize().z);

        r._errors[visibleColumnIndex] = sum / std::max(1, _rLayers[lNext]._visibleCounts[visibleColumnIndex]);
    }
}

void Hierarchy::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<IntBuffer> &hiddenStates,
    const std::vector<IntBuffer> &actions,
    int l,
    int a
) {
    if (l == _rLayers.size()) { // Last layers
        int maxIndex = 0;
        float maxActivation = -999999.0f;

        int hiddenColumnIndex = address2C(pos, Int2(_actionSizes[a].x, _actionSizes[a].y));

        int hiddenIndex = address3C(Int3(pos.x, pos.y, actions[a][hiddenColumnIndex]), _actionSizes[a]);

        RouteLayer &r = _actionLayers[a];

        float delta = _beta * r._errors[hiddenColumnIndex];

        if (l == 0)
            r._weights.deltaOHVs(hiddenStates[l], delta, hiddenIndex, _scLayers[l].getHiddenSize().z);
        else
            r._weights.deltaScalarOHVs(hiddenStates[l], _rLayers[l - 1]._activations, delta, hiddenIndex, _scLayers[l].getHiddenSize().z);
    }
    else { // Hidden layer
        int lNext = l + 1;

        int hiddenColumnIndex = address2C(pos, Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y));
    
        RouteLayer &r = _rLayers[l];

        int hiddenIndex = address3C(Int3(pos.x, pos.y, hiddenStates[lNext][hiddenColumnIndex]), _scLayers[lNext].getHiddenSize());
    
        float delta = _beta * r._errors[hiddenColumnIndex];

        if (l == 0)
            r._weights.deltaOHVs(hiddenStates[l], delta, hiddenIndex, _scLayers[l].getHiddenSize().z);
        else
            r._weights.deltaScalarOHVs(hiddenStates[l], _rLayers[l - 1]._activations, delta, hiddenIndex, _scLayers[l].getHiddenSize().z);
    }
}

void Hierarchy::initRandom(
    ComputeSystem &cs,
    const std::vector<Int3> &inputSizes,
    const std::vector<Int3> &actionSizes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    _scLayers.resize(layerDescs.size());
    _rLayers.resize(layerDescs.size() - 1);

    _ticks.assign(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    _historySizes.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    _updates.resize(layerDescs.size(), false);

    // Cache input sizes
    _inputSizes = inputSizes;
    _actionSizes = actionSizes;

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

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = t + layerDescs[l]._temporalHorizon * i;

                    scVisibleLayerDescs[index]._size = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = layerDescs[l]._scRadius;
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

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                scVisibleLayerDescs[t]._size = layerDescs[l - 1]._hiddenSize;
                scVisibleLayerDescs[t]._radius = layerDescs[l]._scRadius;
            }

            int inSize = layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y;

			for (int v = 0; v < _histories[l].size(); v++) {
                _histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                _historySizes[l][v] = inSize;
            }
        }

        if (l < _rLayers.size() - 1) { // Not topmost
            int lNext = l + 1;

            _rLayers[l]._hiddenCounts = IntBuffer(layerDescs[lNext]._hiddenSize.x * layerDescs[lNext]._hiddenSize.y, 0);

            initSMLocalRF(layerDescs[l]._hiddenSize, layerDescs[lNext]._hiddenSize, layerDescs[l]._rRadius, _rLayers[l]._weights);

            _rLayers[l]._weights.initT();

            // Init weights
            for (int j = 0; j < _rLayers[l]._weights._nonZeroValues.size(); j++)
                _rLayers[l]._weights._nonZeroValues[j] = 1.0f + noiseDist(cs._rng);

            for (int j = 0; j < _rLayers[l]._hiddenCounts.size(); j++)
                _rLayers[l]._hiddenCounts[j] += _rLayers[l]._weights.counts(j * layerDescs[lNext]._hiddenSize.z) / layerDescs[l]._hiddenSize.z;

            _rLayers[l]._visibleCounts = IntBuffer(layerDescs[l]._hiddenSize.x * layerDescs[l]._hiddenSize.y);

            for (int j = 0; j < _rLayers[l]._visibleCounts.size(); j++)
                _rLayers[l]._visibleCounts[j] = _rLayers[l]._weights.countsT(j * layerDescs[l]._hiddenSize.z) / layerDescs[lNext]._hiddenSize.z;

            _rLayers[l]._activations = FloatBuffer(layerDescs[l]._hiddenSize.x * layerDescs[l]._hiddenSize.y, 1.0f);
            _rLayers[l]._errors = FloatBuffer(_rLayers[l]._activations.size(), 0.0f);
        }
		
        // Create the sparse coding layer
        _scLayers[l].initRandom(cs, layerDescs[l]._hiddenSize, scVisibleLayerDescs);
    }

    _actionLayers.resize(_actionSizes.size());
    _actions.resize(_actionSizes.size());

    for (int a = 0; a < _actionSizes.size(); a++) {
        _actionLayers[a]._hiddenCounts = IntBuffer(_actionSizes[a].x * _actionSizes[a].y, 0);

        initSMLocalRF(layerDescs.back()._hiddenSize, _actionSizes[a], layerDescs.back()._rRadius, _actionLayers[a]._weights);

        _actionLayers[a]._weights.initT();

        // Init weights
        for (int j = 0; j < _actionLayers[a]._weights._nonZeroValues.size(); j++)
            _actionLayers[a]._weights._nonZeroValues[j] = 1.0f + noiseDist(cs._rng);

        for (int j = 0; j < _actionLayers[a]._hiddenCounts.size(); j++)
            _actionLayers[a]._hiddenCounts[j] += _actionLayers[a]._weights.counts(j * _actionSizes[a].z) / layerDescs.back()._hiddenSize.z;

        _actionLayers[a]._visibleCounts = IntBuffer(layerDescs.back()._hiddenSize.x * layerDescs.back()._hiddenSize.y);

        for (int j = 0; j < _actionLayers[a]._visibleCounts.size(); j++)
            _actionLayers[a]._visibleCounts[j] = _actionLayers[a]._weights.countsT(j * layerDescs.back()._hiddenSize.z) / _actionSizes[a].z;

        _actionLayers[a]._activations = FloatBuffer(_actionSizes[a].x * _actionSizes[a].y, 1.0f);
        _actionLayers[a]._errors = FloatBuffer(_actionLayers[a]._activations.size(), 0.0f);

        _actions[a] = IntBuffer(_actionSizes[a].x * _actionSizes[a].y, 0);
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

    // Find Q values
    for (int l = 0; l < _scLayers.size(); l++) {
        if (l == _scLayers.size() - 1) {
            for (int a = 0; a < _actionLayers.size(); a++) {
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                    for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                        forward(Int2(x, y), cs._rng, ns._states, ns._actions, l, a, true);
#else
                runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, ns._states, ns._actions, l, a, true), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
            }
        }
        else {
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                    forward(Int2(x, y), cs._rng, ns._states, ns._actions, l, 0, true);
#else
            runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, ns._states, ns._actions, l, 0, true), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
        }
    }

    // Keep predicted Q values
    _q = _rLayers.back()._activations;

    // Action into replay buffer
    ns._actions = _actions;

    // Determine actions by setting errors to 1 and propagating
    for (int a = 0; a < _actionLayers.size(); a++) {
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _actionLayers[a]._errors.size(); x++)
            fillFloat(x, cs._rng, &_actionLayers[a]._errors, 1.0f);
#else
        runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &_actionLayers[a]._errors, 1.0f), _actionLayers[a]._errors.size(), cs._rng, cs._batchSize1);
#endif
    }

    for (int l = _scLayers.size() - 1; l >= 0; l--) {
#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
            for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                backward(Int2(x, y), cs._rng, ns._states, ns._actions, l);
#else
        runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, ns._states, ns._actions, l), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
    }

    // Add history sample
    _historySamples.insert(_historySamples.begin(), ns);

    if (_historySamples.size() > _maxHistorySamples)
        _historySamples.resize(_maxHistorySamples);

    // Learn
    if (learnEnabled && _historySamples.size() > 1) {
        std::uniform_int_distribution<int> sampleDist(1, _historySamples.size() - 1);

        for (int it = 0; it < _historyIters; it++) {
            int t = sampleDist(cs._rng);

            const HistorySample &s = _historySamples[t];
            const HistorySample &sNext = _historySamples[t - 1];

            // Find Q values
            for (int l = 0; l < _scLayers.size(); l++) {
                if (l == _scLayers.size() - 1) {
                    for (int a = 0; a < _actionLayers.size(); a++) {
#ifdef KERNEL_NOTHREAD
                        for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                            for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                                forward(Int2(x, y), cs._rng, s._states, s._actions, l, a);
#else
                        runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, s._states, s._actions, l, a, false), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                    }
                }
                else {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            forward(Int2(x, y), cs._rng, s._states, s._actions, l, 0);
#else
                    runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, s._states, s._actions, l, 0, false), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                }
            }

            // Errors
            for (int a = 0; a < _actionLayers.size(); a++) {
                for (int i = 0; i < _actionLayers[a]._errors.size(); i++) {
                    // Determine target
                    float targetQ = 0.0f;
                    float g = 1.0f;
                    
                    for (int t2 = t - 1; t2 >= 0; t2--) {
                        targetQ += g * _historySamples[t2]._reward;

                        g *= _gamma;
                    }

                    targetQ += g * _q[i];

                    _actionLayers[a]._errors[i] = (targetQ - _actionLayers[a]._activations[i]) * g;
                }
            }

            for (int l = _scLayers.size() - 1; l >= 0; l--) {
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _scLayers[l - 1].getHiddenSize().x; x++)
                    for (int y = 0; y < _scLayers[l - 1].getHiddenSize().y; y++)
                        backward(Int2(x, y), cs._rng, s._states, s._actions, l);
#else
                runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, s._states, s._actions, l), Int2(_scLayers[l - 1].getHiddenSize().x, _scLayers[l - 1].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
            }

            // learn
            for (int l = 0; l < _scLayers.size(); l++) {
                if (l == _scLayers.size() - 1) {
                    for (int a = 0; a < _actionLayers.size(); a++) {
#ifdef KERNEL_NOTHREAD
                        for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                            for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                                learn(Int2(x, y), cs._rng, ns._states, ns._actions, l, a);
#else
                        runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, ns._states, ns._actions, l, a), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
#endif
                    }
                }
                else {
#ifdef KERNEL_NOTHREAD
                    for (int x = 0; x < _scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < _scLayers[l].getHiddenSize().y; y++)
                            learn(Int2(x, y), cs._rng, ns._states, ns._actions, l, 0);
#else
                    runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, ns._states, ns._actions, l, 0), Int2(_scLayers[l].getHiddenSize().x, _scLayers[l].getHiddenSize().y), cs._rng, cs._batchSize2);
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
    int numActions = _actionSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));
    os.write(reinterpret_cast<const char*>(&numActions), sizeof(int));

    os.write(reinterpret_cast<const char*>(_inputSizes.data()), numInputs * sizeof(Int3));
    os.write(reinterpret_cast<const char*>(_actionSizes.data()), numActions * sizeof(Int3));
    
    os.write(reinterpret_cast<const char*>(_updates.data()), _updates.size() * sizeof(char));
    os.write(reinterpret_cast<const char*>(_ticks.data()), _ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = _historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < _historySizes[l].size(); i++)
            writeBufferToStream(os, _histories[l][i].get());

        _scLayers[l].writeToStream(os);
    }

    for (int l = 0; l < _rLayers.size(); l++) {
        writeBufferToStream(os, &_rLayers[l]._activations);
        writeBufferToStream(os, &_rLayers[l]._errors);
        writeBufferToStream(os, &_rLayers[l]._hiddenCounts);
        writeSMToStream(os, _rLayers[l]._weights);
        writeBufferToStream(os, &_rLayers[l]._visibleCounts);
    }

    for (int a = 0; a < _actionLayers.size(); a++) {
        writeBufferToStream(os, &_actionLayers[a]._activations);
        writeBufferToStream(os, &_actionLayers[a]._errors);
        writeBufferToStream(os, &_actionLayers[a]._hiddenCounts);
        writeSMToStream(os, _actionLayers[a]._weights);
        writeBufferToStream(os, &_actionLayers[a]._visibleCounts);
    }

    writeBufferToStream(os, &_actions);

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
        
        os.write(reinterpret_cast<const char*>(&s._reward), sizeof(float));
    }
}

void Hierarchy::readFromStream(
    std::istream &is
) {
    int numLayers;
    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    int numInputs;
    int numActions;

    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));
    is.read(reinterpret_cast<char*>(&numActions), sizeof(int));

    _inputSizes.resize(numInputs);
    _actionSizes.resize(numActions);

    is.read(reinterpret_cast<char*>(_inputSizes.data()), numInputs * sizeof(Int3));
    is.read(reinterpret_cast<char*>(_actionSizes.data()), numActions * sizeof(Int3));

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
    }

    for (int l = 0; l < _rLayers.size(); l++) {
        readBufferFromStream(is, &_rLayers[l]._activations);
        readBufferFromStream(is, &_rLayers[l]._errors);
        readBufferFromStream(is, &_rLayers[l]._hiddenCounts);
        readSMFromStream(is, _rLayers[l]._weights);
        readBufferFromStream(is, &_rLayers[l]._visibleCounts);
    }

    for (int a = 0; a < _actionLayers.size(); a++) {
        readBufferFromStream(is, &_actionLayers[a]._activations);
        readBufferFromStream(is, &_actionLayers[a]._errors);
        readBufferFromStream(is, &_actionLayers[a]._hiddenCounts);
        readSMFromStream(is, _actionLayers[a]._weights);
        readBufferFromStream(is, &_actionLayers[a]._visibleCounts);
    }

    readBufferFromStream(is, &_actions);

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
        
        is.read(reinterpret_cast<char*>(&s._reward), sizeof(float));
    }
}
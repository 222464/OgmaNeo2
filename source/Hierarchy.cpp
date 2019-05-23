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

void Hierarchy::initRandom(
    ComputeSystem &cs,
    const std::vector<Int3> &inputSizes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    _pLayers.resize(layerDescs.size());

    _ticks.assign(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    _historySizes.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    _updates.resize(layerDescs.size(), false);

    _rewards.resize(layerDescs.size(), 0.0f);
    _rewardCounts.resize(layerDescs.size(), 0.0f);

    // Cache input sizes
    _inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    _inputTemporalHorizon = layerDescs.front()._temporalHorizon;

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Histories for all input layers or just the one sparse coder (if not the first layer)
        _histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);

        _historySizes[l].resize(_histories[l].size());
		
        // Create sparse coder visible layer descriptors
        std::vector<Pather::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l]._temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = t + layerDescs[l]._temporalHorizon * i;

                    scVisibleLayerDescs[index]._size = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = layerDescs[l]._radius;
                }
            }
            
            // Initialize history buffers
			for (int v = 0; v < _histories[l].size(); v++) {
				int i = v / layerDescs[l]._temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				_histories[l][v] = std::make_shared<IntBuffer>(inSize);

#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < inSize; x++)
                    fillInt(x, cs._rng, _histories[l][v].get(), 0);
#else
                runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, _histories[l][v].get(), 0), inSize, cs._rng, cs._batchSize1);
#endif

                _historySizes[l][v] = inSize;
			}
        }
        else {
            scVisibleLayerDescs.resize(layerDescs[l]._temporalHorizon);

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                scVisibleLayerDescs[t]._size = layerDescs[l - 1]._hiddenSize;
                scVisibleLayerDescs[t]._radius = layerDescs[l]._radius;
            }

            int inSize = layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y;

			for (int v = 0; v < _histories[l].size(); v++) {
                _histories[l][v] = std::make_shared<IntBuffer>(inSize);

#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < inSize; x++)
                    fillInt(x, cs._rng, _histories[l][v].get(), 0);
#else
                runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, _histories[l][v].get(), 0), inSize, cs._rng, cs._batchSize1);
#endif

                _historySizes[l][v] = inSize;
            }
        }
		
        // Create the sparse coding layer
        _pLayers[l].initRandom(cs, layerDescs[l]._hiddenSize, scVisibleLayerDescs, l == 0);
    }

    _topRewards = FloatBuffer(_pLayers.back().getHiddenSize().x * _pLayers.back().getHiddenSize().y * _pLayers.back().getHiddenSize().z, 0.0f);
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    _pLayers = other._pLayers;

    _historySizes = other._historySizes;
    _updates = other._updates;
    _ticks = other._ticks;
    _ticksPerUpdate = other._ticksPerUpdate;
    _rewards = other._rewards;
    _rewardCounts = other._rewardCounts;
    _inputSizes = other._inputSizes;

    _topRewards = other._topRewards;

    _inputTemporalHorizon = other._inputTemporalHorizon;

    _histories.resize(other._histories.size());

    for (int l = 0; l < _pLayers.size(); l++) {
        _histories[l].resize(other._histories[l].size());

        for (int v = 0; v < _histories[l].size(); v++) {
            _histories[l][v] = std::make_shared<IntBuffer>();
            
            (*_histories[l][v]) = (*other._histories[l][v]);
        }
    }

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
    _updates.resize(_pLayers.size(), false);

    // Forward
    for (int l = 0; l < _pLayers.size(); l++) {
        _rewards[l] += reward;
        _rewardCounts[l] += 1.0f;

        // If is time for layer to tick
        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            // Reset tick
            _ticks[l] = 0;

            // Updated
            _updates[l] = true;
            
            // Activate sparse coder
            _pLayers[l].stepUp(cs, constGet(_histories[l]), learnEnabled);

            // Add to next layer's history
            if (l < _pLayers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = _histories[lNext].size();

                std::shared_ptr<IntBuffer> last = _histories[lNext].back();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    _histories[lNext][t] = _histories[lNext][t - 1];

                // Copy
#ifdef KERNEL_NOTHREAD
                for (int x = 0; x < _pLayers[l].getHiddenCs().size(); x++)
                    copyInt(x, cs._rng, &_pLayers[l].getHiddenCs(), last.get());
#else
                runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_pLayers[l].getHiddenCs(), last.get()), _pLayers[l].getHiddenCs().size(), cs._rng, cs._batchSize1);
#endif

                _histories[lNext].front() = last;

                _ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = _pLayers.size() - 1; l >= 0; l--) {
        if (_updates[l]) {
            // Feed back is current layer state and next higher layer prediction
            const FloatBuffer* feedBackRewards;

            if (l < _pLayers.size() - 1)
                feedBackRewards = &_pLayers[l + 1].getVisibleLayer(_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1])._visibleRewards;
            else
                feedBackRewards = &_topRewards;

            float r;
            
            if (l > 0)
                r = _rewards[l - 1] / std::max(1.0f, _rewardCounts[l - 1]);
            else
                r = 0.0f;

            if (l == _pLayers.size() - 1) {
                for (int x = 0; x < _pLayers[l].getHiddenSize().x; x++)
                    for (int y = 0; y < _pLayers[l].getHiddenSize().y; y++) {
                        _topRewards[
                            address3(Int3(x, y, _pLayers[l].getHiddenCs()[
                                address2(Int2(x, y), Int2(_pLayers[l].getHiddenSize().x, _pLayers[l].getHiddenSize().y))
                                ]), _pLayers[l].getHiddenSize())
                                ] = _rewards[l] / std::max(1.0f, _rewardCounts[l]);
                    }
            }

            _pLayers[l].stepDown(cs, constGet(_histories[l]), feedBackRewards, l == 0, r, learnEnabled);
        }
    }

    // Clear reward data
    for (int l = _pLayers.size() - 1; l >= 0; l--) {
        if (_updates[l]) {
            _rewards[l] = 0.0f;
            _rewardCounts[l] = 0.0f;
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = _pLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = _inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    os.write(reinterpret_cast<const char*>(_updates.data()), _updates.size() * sizeof(char));
    os.write(reinterpret_cast<const char*>(_ticks.data()), _ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    os.write(reinterpret_cast<const char*>(_rewards.data()), _rewards.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(_rewardCounts.data()), _rewardCounts.size() * sizeof(float));

    os.write(reinterpret_cast<const char*>(&_inputTemporalHorizon), sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = _historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < _historySizes[l].size(); i++)
            writeBufferToStream(os, _histories[l][i].get());

        _pLayers[l].writeToStream(os);
    }

    os.write(reinterpret_cast<const char*>(_topRewards.data()), _topRewards.size() * sizeof(float));
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

    _pLayers.resize(numLayers);

    _ticks.resize(numLayers);

    _histories.resize(numLayers);
    _historySizes.resize(numLayers);
    
    _ticksPerUpdate.resize(numLayers);

    _updates.resize(numLayers);

    _rewards.resize(numLayers);
    _rewardCounts.resize(numLayers);

    is.read(reinterpret_cast<char*>(_updates.data()), _updates.size() * sizeof(char));
    is.read(reinterpret_cast<char*>(_ticks.data()), _ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    is.read(reinterpret_cast<char*>(_rewards.data()), _rewards.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(_rewardCounts.data()), _rewardCounts.size() * sizeof(float));

    is.read(reinterpret_cast<char*>(&_inputTemporalHorizon), sizeof(int));

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

        _pLayers[l].readFromStream(is);
    }

    _topRewards.resize(_pLayers.back().getHiddenSize().x * _pLayers.back().getHiddenSize().y * _pLayers.back().getHiddenSize().z);

    is.read(reinterpret_cast<char*>(_topRewards.data()), _topRewards.size() * sizeof(float));
}
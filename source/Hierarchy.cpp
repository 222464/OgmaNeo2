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
    const std::vector<InputType> &inputTypes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    _rLayers.resize(layerDescs.size());
    _pLayers.resize(layerDescs.size());

    // Cache input sizes
    _inputSizes = inputSizes;

    _differencesInfer.resize(layerDescs.size());

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        std::vector<Reservior::VisibleLayerDesc> rVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            rVisibleLayerDescs.resize(inputSizes.size());

            for (int i = 0; i < inputSizes.size(); i++) {
                rVisibleLayerDescs[i]._size = inputSizes[i];
                rVisibleLayerDescs[i]._radius = layerDescs[l]._rfRadius;
                rVisibleLayerDescs[i]._scale = layerDescs[l]._rfScale;
                rVisibleLayerDescs[i]._dropRatio = layerDescs[l]._rfDropRatio;
            }

            // Predictors
            _pLayers[l].resize(inputSizes.size());

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
            pVisibleLayerDescs[0]._radius = layerDescs[l]._pRadius;
            pVisibleLayerDescs[0]._dropRatio = layerDescs[l]._pDropRatio;

            // Create actors
            for (int p = 0; p < _pLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::_predict) {
                    _pLayers[l][p] = std::make_unique<Predictor>();

                    _pLayers[l][p]->initRandom(cs, inputSizes[p], pVisibleLayerDescs);
                }
            }
        }
        else {
            rVisibleLayerDescs.resize(1);

            rVisibleLayerDescs[0]._size = layerDescs[l - 1]._hiddenSize;
            rVisibleLayerDescs[0]._radius = layerDescs[l]._rfRadius;
            rVisibleLayerDescs[0]._scale = layerDescs[l]._rfScale;
            rVisibleLayerDescs[0]._dropRatio = layerDescs[l]._rfDropRatio;

            _pLayers[l].resize(1);

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
            pVisibleLayerDescs[0]._radius = layerDescs[l]._pRadius;
            pVisibleLayerDescs[0]._dropRatio = layerDescs[l]._pDropRatio;

            // Create actors
            for (int p = 0; p < _pLayers[l].size(); p++) {
                _pLayers[l][p] = std::make_unique<Predictor>();

                _pLayers[l][p]->initRandom(cs, layerDescs[l - 1]._hiddenSize, pVisibleLayerDescs);
            }
        }

        // Recurrent
        if (layerDescs[l]._rrRadius != -1) {
            Reservior::VisibleLayerDesc vld;

            vld._size = layerDescs[l]._hiddenSize;
            vld._radius = layerDescs[l]._rrRadius;
            vld._scale = layerDescs[l]._rrScale;
            vld._dropRatio = layerDescs[l]._rrDropRatio;

            rVisibleLayerDescs.push_back(vld);
        }
		
        // Create the sparse coding layer
        _rLayers[l].initRandom(cs, layerDescs[l]._hiddenSize, rVisibleLayerDescs);

        _differencesInfer[l] = FloatBuffer(_rLayers[l].getHiddenStates().size(), 0.0f);
    }

    _differencesLearn = _differencesInfer;
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    _rLayers = other._rLayers;

    _inputSizes = other._inputSizes;

    _pLayers.resize(other._pLayers.size());

    for (int l = 0; l < _rLayers.size(); l++) {
        _pLayers[l].resize(other._pLayers[l].size());

        for (int v = 0; v < _pLayers[l].size(); v++) {
            if (other._pLayers[l][v] != nullptr) {
                _pLayers[l][v] = std::make_unique<Predictor>();

                (*_pLayers[l][v]) = (*other._pLayers[l][v]);
            }
            else
                _pLayers[l][v] = nullptr;
        }
    }

    _differencesInfer = other._differencesInfer;
    _differencesLearn = other._differencesLearn;

    return *this;
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputStates,
    const FloatBuffer* goalStates,
    bool learnEnabled
) {
    assert(inputStates.size() == _inputSizes.size());

    // Forward
    for (int l = 0; l < _rLayers.size(); l++) {
        std::vector<const FloatBuffer*> fullLayerInputs = l == 0 ? inputStates : std::vector<const FloatBuffer*>{ &_rLayers[l - 1].getHiddenStates() };

        // Add recurrent if needed
        if (fullLayerInputs.size() < _rLayers[l].getNumVisibleLayers())
            fullLayerInputs.push_back(&_rLayers[l].getHiddenStates());

        _rLayers[l].step(cs, fullLayerInputs);
    }

    // Backward
    for (int l = _rLayers.size() - 1; l >= 0; l--) {
        std::vector<const FloatBuffer*> statesInfer(2);
        std::vector<const FloatBuffer*> statesLearn(2);
        
        const FloatBuffer* feedBackStates = l < _rLayers.size() - 1 ? &_pLayers[l + 1][0]->getHiddenStates() : goalStates;

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _differencesInfer[l].size(); x++)
            diffFloat(x, cs._rng, feedBackStates, &_rLayers[l].getHiddenStates(), &_differencesInfer[l]);
#else
        runKernel1(cs, std::bind(diffFloat, std::placeholders::_1, std::placeholders::_2, feedBackStates, &_rLayers[l].getHiddenStates(), &_differencesInfer[l]), _differencesInfer[l].size(), cs._rng, cs._batchSize1);
#endif

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _differencesLearn[l].size(); x++)
            diffFloat(x, cs._rng, &_rLayers[l].getHiddenStates(), &_rLayers[l].getHiddenStatesPrev(), &_differencesLearn[l]);
#else
        runKernel1(cs, std::bind(diffFloat, std::placeholders::_1, std::placeholders::_2, &_rLayers[l].getHiddenStates(), &_rLayers[l].getHiddenStatesPrev(), &_differencesLearn[l]), _differencesLearn[l].size(), cs._rng, cs._batchSize1);
#endif

        // Step actor layers
        for (int p = 0; p < _pLayers[l].size(); p++) {
            if (_pLayers[l][p] != nullptr) {
                if (learnEnabled) 
                    _pLayers[l][p]->learn(cs, l == 0 ? inputStates[p] : &_rLayers[l - 1].getHiddenStates(), { &_differencesLearn[l] });

                _pLayers[l][p]->activate(cs, { &_differencesInfer[l] });
            }
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = _rLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = _inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    for (int l = 0; l < numLayers; l++) {
        _rLayers[l].writeToStream(os);

        for (int v = 0; v < _pLayers[l].size(); v++) {
            char exists = _pLayers[l][v] != nullptr;

            os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

            if (exists)
                _pLayers[l][v]->writeToStream(os);
        }
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

    _rLayers.resize(numLayers);
    _pLayers.resize(numLayers);

    _differencesInfer.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        _rLayers[l].readFromStream(is);

        _pLayers[l].resize(l == 0 ? _inputSizes.size() : 1);

        for (int v = 0; v < _pLayers[l].size(); v++) {
            char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(char));

            if (exists) {
                _pLayers[l][v] = std::make_unique<Predictor>();
                _pLayers[l][v]->readFromStream(is);
            }
            else
                _pLayers[l][v] = nullptr;
        }

        _differencesInfer[l] = FloatBuffer(_rLayers[l].getHiddenStates().size(), 0.0f);
    }

    _differencesLearn = _differencesInfer;
}
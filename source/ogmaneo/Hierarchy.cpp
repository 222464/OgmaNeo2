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
    _rLayers.resize(layerDescs.size());
    _pLayers.resize(layerDescs.size());

    // Cache input sizes
    _inputSizes = inputSizes;

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        std::vector<Reservoir::VisibleLayerDesc> rVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            rVisibleLayerDescs.resize(inputSizes.size());

            for (int i = 0; i < inputSizes.size(); i++) {
                rVisibleLayerDescs[i]._size = inputSizes[i];
                rVisibleLayerDescs[i]._radius = layerDescs[l]._rfRadius;
                rVisibleLayerDescs[i]._scale = layerDescs[l]._rfScale;
                rVisibleLayerDescs[i]._dropRatio = layerDescs[l]._rfDropRatio;
                rVisibleLayerDescs[i]._noDiagonal = false;
            }

            // Predictors
            _pLayers[l].resize(inputSizes.size());
            _pErrors[l].resize(inputSizes.size());

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc._size = layerDescs[l]._hiddenSize;
            pVisibleLayerDesc._radius = layerDescs[l]._pRadius;
            pVisibleLayerDesc._scale = layerDescs[l]._pScale;
            pVisibleLayerDesc._dropRatio = layerDescs[l]._pDropRatio;

            // Create actors
            for (int p = 0; p < _pLayers[l].size(); p++) {
                _pLayers[l][p].initRandom(cs, inputSizes[p], pVisibleLayerDesc);

                _pErrors[l][p] = FloatBuffer(inputSizes[p].x * inputSizes[p].y * inputSizes[p].z, 0.0f);
            }
        }
        else {
            rVisibleLayerDescs.resize(1);

            rVisibleLayerDescs[0]._size = layerDescs[l - 1]._hiddenSize;
            rVisibleLayerDescs[0]._radius = layerDescs[l]._rfRadius;
            rVisibleLayerDescs[0]._scale = layerDescs[l]._rfScale;
            rVisibleLayerDescs[0]._dropRatio = layerDescs[l]._rfDropRatio;
            rVisibleLayerDescs[0]._noDiagonal = false;

            _pLayers[l].resize(1);
            _pErrors[l].resize(1);

            // Predictor visible layer descriptors
            Predictor::VisibleLayerDesc pVisibleLayerDesc;

            pVisibleLayerDesc._size = layerDescs[l]._hiddenSize;
            pVisibleLayerDesc._radius = layerDescs[l]._pRadius;
            pVisibleLayerDesc._scale = layerDescs[l]._pScale;
            pVisibleLayerDesc._dropRatio = layerDescs[l]._pDropRatio;

            // Create actors
            for (int p = 0; p < _pLayers[l].size(); p++) {
                _pLayers[l][p].initRandom(cs, layerDescs[l - 1]._hiddenSize, pVisibleLayerDesc);

                _pErrors[l][p] = FloatBuffer(layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y * layerDescs[l - 1]._hiddenSize.z, 0.0f);
            }
        }

        // Recurrent
        if (layerDescs[l]._rrRadius != -1) {
            Reservoir::VisibleLayerDesc vld;

            vld._size = layerDescs[l]._hiddenSize;
            vld._radius = layerDescs[l]._rrRadius;
            vld._scale = layerDescs[l]._rrScale;
            vld._dropRatio = layerDescs[l]._rrDropRatio;
            vld._noDiagonal = true;

            rVisibleLayerDescs.push_back(vld);
        }
		
        // Create the sparse coding layer
        _rLayers[l].initRandom(cs, layerDescs[l]._hiddenSize, rVisibleLayerDescs, layerDescs[l]._rbScale);
    }
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputStates,
    bool learnEnabled
) {
    assert(inputStates.size() == _inputSizes.size());

    // Forward
    for (int l = 0; l < _rLayers.size(); l++) {
        std::vector<const FloatBuffer*> fullLayerInputs(inputStates.size());

        // Find differences
        for (int p = 0; p < _pErrors[l].size(); p++) {
            // Difference
#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < _pErrors[l][p].size(); x++)
                diffFloat(x, cs._rng, &_pLayers[l][p].getHiddenStates(), &_pErrors[l][p]);
#else
            runKernel1(cs, std::bind(diffFloat, std::placeholders::_1, std::placeholders::_2, inputStates[p], &_pLayers[l][p].getHiddenStates(), &_pErrors[l][p]), _pErrors[l][p].size(), cs._rng, cs._batchSize1);
#endif

            fullLayerInputs[p] = &_pErrors[l][p];
        }

        // Add recurrent if needed
        if (fullLayerInputs.size() < _rLayers[l].getNumVisibleLayers())
            fullLayerInputs.push_back(&_rLayers[l].getHiddenStatesPrev());

        _rLayers[l].step(cs, fullLayerInputs);
    }

    // Backward
    for (int l = _rLayers.size() - 1; l >= 0; l--) {
        const FloatBuffer* feedBackStates = l < _rLayers.size() - 1 ? &_pLayers[l + 1][0].getHiddenStates() : nullptr;

        // Step actor layers
        for (int p = 0; p < _pLayers[l].size(); p++) {
            if (learnEnabled) 
                _pLayers[l][p].learn(cs, feedBackStates, &_rLayers[l].getHiddenStates(), l == 0 ? inputStates[p] : &_rLayers[l - 1].getHiddenStates());

            _pLayers[l][p].activate(cs, feedBackStates, &_rLayers[l].getHiddenStates());
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

        for (int v = 0; v < _pLayers[l].size(); v++)
            _pLayers[l][v].writeToStream(os);
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

    for (int l = 0; l < numLayers; l++) {
        _rLayers[l].readFromStream(is);

        _pLayers[l].resize(l == 0 ? _inputSizes.size() : 1);

        for (int v = 0; v < _pLayers[l].size(); v++)
            _pLayers[l][v].readFromStream(is);
    }
}
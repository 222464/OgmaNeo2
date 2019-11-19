// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

namespace ogmaneo {
// Sparse coder
class Reservoir {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 _size; // Size of input

        int _radius; // Radius onto input

        float _scale; // Scale of weights
        float _dropRatio; // Ratio of weights to drop

        bool _noDiagonal;

        // Defaults
        VisibleLayerDesc()
        :
        _size(4, 4, 16),
        _radius(2),
        _scale(1.0f),
        _dropRatio(0.0f),
        _noDiagonal(false)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weights; // Weight matrix
    };

private:
    Int3 _hiddenSize; // Size of hidden/output layer

    FloatBuffer _hiddenStates; // Hidden states
    FloatBuffer _hiddenStatesPrev; // Previous tick hidden states

    FloatBuffer _hiddenBiases; // Bias weights

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputStates,
        bool learnEnabled
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Reservoir* sc,
        const std::vector<const FloatBuffer*> &inputStates,
        bool learnEnabled
    ) {
        sc->forward(pos, rng, inputStates, learnEnabled);
    }

public:
    float _alpha; // Self-organization learning rate

    Reservoir()
    :
    _alpha(0.001f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs, // Descriptors for visible layers
        float biasScale // Scale for bias weights
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputStates,
        bool learnEnabled
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get the number of visible layers
    int getNumVisibleLayers() const {
        return _visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return _visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return _visibleLayerDescs[i];
    }

    // Get the hidden states
    const FloatBuffer &getHiddenStates() const {
        return _hiddenStates;
    }

    // Get the hidden states
    const FloatBuffer &getHiddenStatesPrev() const {
        return _hiddenStatesPrev;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }

    // Get the weights for a visible layer
    const SparseMatrix &getWeights(
        int i // Index of visible layer
    ) const {
        return _visibleLayers[i]._weights;
    }
};
} // namespace ogmaneo

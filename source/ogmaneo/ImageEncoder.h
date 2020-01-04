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
class ImageEncoder {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 _size; // Size of input

        int _encRadius; // Radius onto input
        int _decRadius; // Radius onto hidden

        // Defaults
        VisibleLayerDesc()
        :
        _size(4, 4, 16),
        _encRadius(2),
        _decRadius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _encWeights;
        SparseMatrix _decWeights;

        FloatBuffer _reconActs;
    };

private:
    Int3 _hiddenSize; // Size of hidden/output layer

    IntBuffer _hiddenCs; // Hidden states

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputActs
    );

    void backward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int vli
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const FloatBuffer* inputActs,
        int vli
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* enc,
        const std::vector<const FloatBuffer*> &inputActs
    ) {
        enc->forward(pos, rng, inputActs);
    }

    static void backwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* enc,
        const IntBuffer* hiddenCs,
        int vli
    ) {
        enc->backward(pos, rng, hiddenCs, vli);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* enc,
        const FloatBuffer* inputActs,
        int vli
    ) {
        enc->learn(pos, rng, inputActs, vli);
    }

public:
    float _alpha; // Resource depletion rate
    float _epsilon; // Error tolerance

    // Defaults
    ImageEncoder()
    :
    _alpha(0.5f),
    _epsilon(0.001f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void activate(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputActs // Input states
    );

    void learn(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &targetActs // Target states
    );

    void reconstruct(
        ComputeSystem &cs, // Compute system
        const IntBuffer* hiddenCs
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
    const IntBuffer &getHiddenCs() const {
        return _hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }
};
} // namespace ogmaneo

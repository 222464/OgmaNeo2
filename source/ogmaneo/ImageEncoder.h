// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

namespace ogmaneo {
// Encodes images (dense -> CSDR)
class ImageEncoder {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 _size; // Size of input

        int _radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        : 
        _size(4, 4, 16),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weights; // Weight matrix
    };

private:
    Int3 _hiddenSize; // Hidden layer size

    IntBuffer _hiddenCs; // Hidden state

    std::vector<VisibleLayer> _visibleLayers; // Layers
    std::vector<VisibleLayerDesc> _visibleLayerDescs; // Descs

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputActivations
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc,
        const std::vector<const FloatBuffer*> &inputActivations
    ) {
        sc->forward(pos, rng, inputActivations);
    }

public:
    // Create a randomly initialized image encoder
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Size of the hidden layer
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descs
    );

    // Step the image encoder
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputActivations // Input state (activations)
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );
    
    // Get the number of visible (input) layers
    int getNumVisibleLayers() const {
        return _visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of the visible layer
    ) const {
        return _visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of the visible layer
    ) const {
        return _visibleLayerDescs[i];
    }

    // Get the hidden state
    const IntBuffer &getHiddenCs() const {
        return _hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }
};
} // namespace ogmaneo

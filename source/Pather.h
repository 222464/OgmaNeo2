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
// Pathfinding
int findNextIndex(
    int startIndex,
    int endIndex,
    int size,
    int transitionsStart,
    const FloatBuffer &transitions,
    float gamma
);

// Sparse coder
class Pather {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 _size; // Size of input

        int _radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        _size({ 4, 4, 16 }),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weights; // Weight matrix

        IntBuffer _visibleCounts; // Number touching

        IntBuffer _recons; // Reconstruction
    };

private:
    Int3 _hiddenSize; // Size of hidden/output layer

    IntBuffer _hiddenCs; // Hidden states
    IntBuffer _hiddenCsPrev; // Hidden states from previous tick

    IntBuffer _predictedCs; // Predicted (pathed) states

    FloatBuffer _transitionWeights; // Transitioning (state to next state) probability weights

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learnWeights(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs,
        int vli
    );

    void transition(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* feedBackCs,
        bool learnEnabled
    );

    void reconstruct(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int vli
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        sc->forward(pos, rng, inputCs);
    }

    static void learnWeightsKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const std::vector<const IntBuffer*> &inputCs,
        int vli
    ) {
        sc->learnWeights(pos, rng, inputCs, vli);
    }

    static void transitionKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const IntBuffer* feedBackCs,
        bool learnEnabled
    ) {
        sc->transition(pos, rng, feedBackCs, learnEnabled);
    }

    static void reconstructKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const IntBuffer* hiddenCs,
        int vli
    ) {
        sc->reconstruct(pos, rng, hiddenCs, vli);
    }

public:
    float _alpha; // Weight learning rate
    float _beta; // Transition learning rate
    float _gamma; // Distance penalty

    // Defaults
    Pather()
    :
    _alpha(0.01f),
    _beta(0.1f),
    _gamma(0.9f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void stepUp(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
    );

    void stepDown(
        ComputeSystem &cs, // Compute system
        const IntBuffer* feedBackCs, // States to reconstruct
        bool learnEnabled // Whether to learn
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

    // Get the weights for a visible layer
    const SparseMatrix &getWeights(
        int i // Index of visible layer
    ) const {
        return _visibleLayers[i]._weights;
    }
};
} // namespace ogmaneo

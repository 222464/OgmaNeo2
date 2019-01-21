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
// A prediction layer (predicts x_(t+1))
class Predictor {
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

        IntBuffer _inputCsPrev; // Previous timestep (prev) input states
    };

private:
    Int3 _hiddenSize; // Size of the output/hidden/prediction

    IntBuffer _hiddenCs; // Hidden state

    FloatBuffer _hiddenActivations; // Hidden activations, used for interal computation
    FloatBuffer _hiddenDeltas; // Hidden targets, used for interal computation

    // Visible layers and descs
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;

    // --- Kernels ---

    void init(
        int pos,
        std::mt19937 &rng,
        int vli
    );

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenTargetCs
    );

    static void initKernel(
        int pos,
        std::mt19937 &rng,
        Predictor* p,
        int vli
    ) {
        p->init(pos, rng, vli);
    }

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        p->forward(pos, rng, inputCs);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        const IntBuffer* hiddenTargetCs
    ) {
        p->learn(pos, rng, hiddenTargetCs);
    }

public:
    float _alpha; // Learning rate

    // Defaults
    Predictor()
    :
    _alpha(0.1f)
    {}

    // Create with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // First visible layer must be from current hidden state, second must be feed back state, rest can be whatever
    ); 

    // Activate the predictor (predict values)
    void activate(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &visibleCs // Hidden/output/prediction size
    );

    // Learning predictions (update weights)
    void learn(
        ComputeSystem &cs,
        const IntBuffer* hiddenTargetCs
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get number of visible layers
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

    // Get the hidden activations (predictions)
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
    ) {
        return _visibleLayers[i]._weights;
    }
};
} // Namespace ogmaneo

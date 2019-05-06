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
class MSOM {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int2 _size; // Size of input

        int _radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        _size(16, 16),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weights; // Weight matrix

        FloatBuffer _recons; // Reconstructions
    };

private:
    Int2 _hiddenSize; // Size of hidden/output layer

    int _crossRadius;

    FloatBuffer _hiddenActivations;
    FloatBuffer _hiddenStates;
    FloatBuffer _hiddenBlurs;

    FloatBuffer _hiddenPathsPing;
    FloatBuffer _hiddenPathsPong;

    FloatBuffer _hiddenPredictions;

    FloatBuffer _hiddenStatesPrev;

    // Traces
    SparseMatrix _crossTraces;

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputs
    );

    void inhibit(
        const Int2 &pos,
        std::mt19937 &rng,
        FloatBuffer* hiddenStates
    );

    void blur(
        const Int2 &pos,
        std::mt19937 &rng
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputs
    );

    void backward(
        const Int2 &pos,
        std::mt19937 &rng,
        const FloatBuffer* hiddenStates,
        int vli
    );

    void plan(
        const Int2 &pos,
        std::mt19937 &rng,
        const FloatBuffer* feedBackStates,
        int t
    );

    void predict(
        const Int2 &pos,
        std::mt19937 &rng,
        const FloatBuffer* feedBackStates
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        MSOM* p,
        const std::vector<const FloatBuffer*> &inputs
    ) {
        p->forward(pos, rng, inputs);
    }

    static void inhibitKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        MSOM* p,
        FloatBuffer* hiddenStates
    ) {
        p->inhibit(pos, rng, hiddenStates);
    }

    static void blurKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        MSOM* p
    ) {
        p->blur(pos, rng);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        MSOM* p,
        const std::vector<const FloatBuffer*> &inputs
    ) {
        p->learn(pos, rng, inputs);
    }

    static void backwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        MSOM* p,
        const FloatBuffer* hiddenStates,
        int vli
    ) {
        p->backward(pos, rng, hiddenStates, vli);
    }

    static void planKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        MSOM* p,
        const FloatBuffer* feedBackStates,
        int t
    ) {
        p->plan(pos, rng, feedBackStates, t);
    }

    static void predictKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        MSOM* p,
        const FloatBuffer* feedBackStates
    ) {
        p->predict(pos, rng, feedBackStates);
    }

public:
    float _alpha; // Feed learning rate
    float _gamma; // Trace decay
    float _minTrace; // Minimum trace threshold for a prediction to occur
    int _inhibitRadius; // Max activation radius
    int _blurRadius; // Radius of learning
    int _planRadius; // Radius of planning
    int _planIters; // Pathing iterations

    // Defaults
    MSOM()
    :
    _alpha(0.05f),
    _gamma(0.98f),
    _minTrace(0.01f),
    _inhibitRadius(2),
    _blurRadius(1),
    _planRadius(3),
    _planIters(16) // Must be even (implementation detail)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int2 &hiddenSize, // Hidden/output size
        int crossRadius, // Trace radius
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void activate(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputs // Input states
    );

    void learn(
        ComputeSystem &cs,
        const std::vector<const FloatBuffer*> &inputs // Input states
    );

    void reconstruct(
        ComputeSystem &cs,
        const FloatBuffer* hiddenStates
    );

    void predict(
        ComputeSystem &cs,
        const FloatBuffer* feedBackStates
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

    // Get the hidden blurs
    const FloatBuffer &getHiddenBlurs() const {
        return _hiddenBlurs;
    }

    // Get the hidden predictions
    const FloatBuffer &getHiddenPredictions() const {
        return _hiddenPredictions;
    }

    // Get the hidden size
    const Int2 &getHiddenSize() const {
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

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

        float _scale;
        float _dropRatio;

        // Defaults
        VisibleLayerDesc()
        :
        _size(4, 4, 16),
        _radius(2),
        _scale(1.0f),
        _dropRatio(0.0f)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weightsFeedBack;
        SparseMatrix _weightsInput;
    };

    struct HistorySample {
        FloatBuffer _inputStates;
        FloatBuffer _hiddenTargetStates;
    };

private:
    Int3 _hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer _hiddenStates; // Hidden states

    // Visible layers and descs
    VisibleLayer _visibleLayer;
    VisibleLayerDesc _visibleLayerDesc;

    std::vector<std::shared_ptr<HistorySample>> _historySamples;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const FloatBuffer* feedBackStates,
        const FloatBuffer* inputStates
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        int index
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        const FloatBuffer* feedBackStates,
        const FloatBuffer* inputStates
    ) {
        p->forward(pos, rng, feedBackStates, inputStates);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        int index
    ) {
        p->learn(pos, rng, index);
    }

public:
    float _alpha; // Prediction learning rate
    float _beta; // Value learning rate
    float _gamma; // Discount factor

    int _maxHistorySamples;
    int _historyIters;

    // Defaults
    Predictor()
    :
    _alpha(0.01f),
    _gamma(0.99f),
    _maxHistorySamples(256),
    _historyIters(16)
    {}

    // Create with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const VisibleLayerDesc &visibleLayerDesc
    ); 

    // Activate the predictor (predict values)
    void activate(
        ComputeSystem &cs, // Compute system
        const FloatBuffer* feedBackStates,
        const FloatBuffer* inputStates
    );

    // Learning predictions (update weights)
    void learn(
        ComputeSystem &cs,
        const FloatBuffer* hiddenTargetStates,
        const FloatBuffer* inputStates
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get a visible layer
    const VisibleLayer &getVisibleLayer() const {
        return _visibleLayer;
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc() const {
        return _visibleLayerDesc;
    }

    // Get the hidden activations (predictions)
    const FloatBuffer &getHiddenStates() const {
        return _hiddenStates;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }
};
} // Namespace ogmaneo

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
        SparseMatrix _weights;

        FloatBuffer _difference;
    };

    struct HistorySample {
        std::vector<FloatBuffer> _inputStates;
        FloatBuffer _hiddenTargetStates;
    };

private:
    Int3 _hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer _hiddenStates; // Hidden states

    // Visible layers and descs
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;

    std::vector<std::shared_ptr<HistorySample>> _historySamples;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        int index
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p
    ) {
        p->forward(pos, rng);
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
    _historyIters(32)
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
        const std::vector<const FloatBuffer*> &feedBackStates,
        const std::vector<const FloatBuffer*> &inputStates
    );

    // Learning predictions (update weights)
    void learn(
        ComputeSystem &cs,
        const FloatBuffer* hiddenTargetStates,
        const std::vector<const FloatBuffer*> &inputStates
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
    const FloatBuffer &getHiddenStates() const {
        return _hiddenStates;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }
};
} // Namespace ogmaneo

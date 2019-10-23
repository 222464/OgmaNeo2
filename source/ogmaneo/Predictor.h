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
        _size(4, 4, 16),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weights; // Weight matrix
    };

    struct HistorySample {
        IntBuffer _inputCs;

        IntBuffer _hiddenTargetCs;
    };

private:
    Int3 _hiddenSize; // Size of the output/hidden/prediction

    IntBuffer _hiddenCs; // Hidden state

    std::vector<std::shared_ptr<HistorySample>> _historySamples;

    // Visible layers and descs
    VisibleLayer _visibleLayer;
    VisibleLayerDesc _visibleLayerDesc;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* feedBackCs,
        const IntBuffer* inputCs
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* feedBackCs,
        int index
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        const IntBuffer* feedBackCs,
        const IntBuffer* inputCs
    ) {
        p->forward(pos, rng, feedBackCs, inputCs);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Predictor* p,
        const IntBuffer* feedBackCs,
        int index
    ) {
        p->learn(pos, rng, feedBackCs, index);
    }

public:
    float _alpha; // Learning rate
    float _gamma; // Discount
    int _maxHistorySize;

    // Defaults
    Predictor()
    :
    _alpha(0.1f),
    _gamma(0.9f),
    _maxHistorySize(7)
    {}

    // Copy
    Predictor(
        const Predictor &other // Predictor to copy from
    ) {
        *this = other;
    }

    // Assignment
    const Predictor &operator=(
        const Predictor &other // Predictor to assign from
    );

    // Create with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const VisibleLayerDesc &visibleLayerDesc
    ); 

    // Activate the predictor (predict values)
    void activate(
        ComputeSystem &cs, // Compute system
        const IntBuffer* feedBackCs,
        const IntBuffer* inputCs
    );

    // Learning predictions (update weights)
    void learn(
        ComputeSystem &cs,
        const IntBuffer* hiddenTargetCs,
        const IntBuffer* inputCs
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
    const IntBuffer &getHiddenCs() const {
        return _hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }
};
} // Namespace ogmaneo

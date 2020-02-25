// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
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
        Int3 size; // Size of input

        int radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    struct HistorySample {
        IntBuffer inputCs;
        IntBuffer hiddenTargetCs;
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    int historySize;

    IntBuffer hiddenCs; // Hidden state

    SparseMatrix weights; // Weight matrix
    VisibleLayerDesc visibleLayerDesc;

    std::vector<HistorySample> historySamples;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenTargetCs,
        const std::vector<const IntBuffer*> &inputCs
    );

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
        const IntBuffer* hiddenTargetCs,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        p->learn(pos, rng, hiddenTargetCs, inputCs);
    }

public:
    float alpha; // Learning rate
    int historyIters;

    // Defaults
    Predictor()
    :
    alpha(0.1f),
    historyIters(4)
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
        const IntBuffer* goalCs,
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

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc() const {
        return visibleLayerDesc;
    }

    // Get the hidden activations (predictions)
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace ogmaneo

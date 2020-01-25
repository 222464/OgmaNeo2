// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEOLICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

namespace ogmaneo {
// A prediction layer (predicts x(t+1))
class Predictor {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Size of input

        int radius; // Radius onto input

        float scale;
        float dropRatio;

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2),
        scale(1.0f),
        dropRatio(0.0f)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix feedBackWeights;
        SparseMatrix inputWeights;
    };

    struct HistorySample {
        FloatBuffer feedBackStates;
        FloatBuffer inputStates;
        FloatBuffer hiddenTargetStates;
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer hiddenStates; // Hidden states

    // Visible layers and descs
    VisibleLayer visibleLayer;
    VisibleLayerDesc visibleLayerDesc;

    std::vector<std::shared_ptr<HistorySample>> historySamples;

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
    float alpha; // Prediction learning rate

    int maxHistorySamples;
    int historyIters;

    // Defaults
    Predictor()
    :
    alpha(0.001f),
    maxHistorySamples(512),
    historyIters(16)
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
        const FloatBuffer* feedBackStates,
        const FloatBuffer* inputStates,
        const FloatBuffer* hiddenTargetStates
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
        return visibleLayer;
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc() const {
        return visibleLayerDesc;
    }

    // Get the hidden activations (predictions)
    const FloatBuffer &getHiddenStates() const {
        return hiddenStates;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace ogmaneo

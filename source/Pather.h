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
// Planning
void iterate(
    int numStates,
    int transitionsStart,
    int rewardsStart,
    const FloatBuffer &transitions,
    const FloatBuffer &rewards,
    FloatBuffer &qs,
    float gamma
);

int getPolicy(
    int startIndex,
    int numStates,
    int transitionsStart,
    const FloatBuffer &qs
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
        _size(4, 4, 16),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _ffWeights; // Feed forward weight matrix
        SparseMatrix _fbWeights; // Feed back weight matrix

        FloatBuffer _visibleRewards; // Reward predictions by feed back matrix
        IntBuffer _visibleActions; // Actions predictions by transposed feed forward matrix
    };

private:
    Int3 _hiddenSize; // Size of hidden/output layer

    IntBuffer _hiddenCs; // Hidden states
    IntBuffer _hiddenCsPrev; // Hidden states from previous tick

    IntBuffer _predictedCs; // Predicted (pathed) states

    FloatBuffer _transitionWeights; // Transitioning (state to next state) probability weights

    FloatBuffer _qs; // Q values

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learnFF(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs,
        int vli
    );

    void learnFB(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs,
        int vli,
        float reward
    );

    void transition(
        const Int2 &pos,
        std::mt19937 &rng,
        const FloatBuffer* feedBackRewards,
        bool learnEnabled
    );

    void backwardActions(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int vli
    );

    void backwardRewards(
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

    static void learnFFKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const std::vector<const IntBuffer*> &inputCs,
        int vli
    ) {
        sc->learnFF(pos, rng, inputCs, vli);
    }

    static void learnFBKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const std::vector<const IntBuffer*> &inputCs,
        int vli,
        float reward
    ) {
        sc->learnFB(pos, rng, inputCs, vli, reward);
    }

    static void transitionKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const FloatBuffer* feedBackRewards,
        bool learnEnabled
    ) {
        sc->transition(pos, rng, feedBackRewards, learnEnabled);
    }

    static void backwardActionsKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const IntBuffer* hiddenCs,
        int vli
    ) {
        sc->backwardActions(pos, rng, hiddenCs, vli);
    }

    static void backwardRewardsKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Pather* sc,
        const IntBuffer* hiddenCs,
        int vli
    ) {
        sc->backwardRewards(pos, rng, hiddenCs, vli);
    }

public:
    float _ffLearnRate; // FF weight learning rate
    float _fbLearnRate; // FB weight learning rate
    float _tLearnRate; // Transition learning rate
    float _gamma; // Distance penalty
    int _iterations; // Iterations for pathing solution

    // Defaults
    Pather()
    :
    _ffLearnRate(0.01f),
    _fbLearnRate(0.01f),
    _tLearnRate(0.1f),
    _gamma(0.98f),
    _iterations(16)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs, // Descriptors for visible layers
        bool isFirstLayer // Whether this is the first (bottom-most) layer
    );

    // Activate the sparse coder (perform sparse coding)
    void stepUp(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
    );

    void stepDown(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Input states
        const FloatBuffer* feedBackRewards, // Rewards from layer above or given by user
        bool isFirstLayer,
        float reward,
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
};
} // namespace ogmaneo

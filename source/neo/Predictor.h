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
    /*!
    \brief A actor layer (swarm intelligence of actor columns)
    Maps input CSDRs to actions using a swarm of actor-critic units with Boltzmann exploration
    */
    class Predictor {
    private:
        /*!
        \brief Size of the hidden layer (output/action size)
        */
        Int3 _hiddenSize;
        Int3 _visibleSize;

        int _radius;

        FloatBuffer _weights;

        Float2 _hiddenToVisible;

        /*!
        \brief Buffers for state
        */
        IntBuffer _hiddenCs;

        //!@{
        /*!
        \brief Kernels
        */
        void init(int pos, std::mt19937 &rng);
        void forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs);
        void learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCsPrev, const IntBuffer* hiddenTargetCs);

        static void initKernel(int pos, std::mt19937 &rng, Predictor* a) {
            a->init(pos, rng);
        }

        static void forwardKernel(const Int2 &pos, std::mt19937 &rng, Predictor* a, const std::vector<const IntBuffer*> &inputCs) {
            a->forward(pos, rng, inputCs);
        }

        static void learnKernel(const Int2 &pos, std::mt19937 &rng, Predictor* a, const std::vector<const IntBuffer*> &inputCsPrev, const IntBuffer* hiddenTargetCs) {
            a->learn(pos, rng, inputCsPrev, hiddenTargetCs);
        }
        //!@}

    public:
        /*!
        \brief Value learning rate
        */
        float _alpha;

        /*!
        \brief Initialize defaults
        */
        Predictor()
        : _alpha(1.0f)
        {}

        /*!
        \brief Create an actor layer with random initialization
        \param cs is the ComputeSystem
        \param hiddenSize size of the actions (output)
        \param historyCapacity maximum number of history samples (fixed)
        \param visibleLayerDescs are descriptors for visible layers
        */
        void createRandom(ComputeSystem &cs,
            const Int3 &hiddenSize, const Int3 &visibleSize, int radius);

        /*!
        \brief Activate the predictor (predict values)
        \param cs is the ComputeSystem
        \param visibleCs the visible (input) layer states
        \param hiddenPredictionCsPrev the previously taken actions
        \param learn whether to learn
        */
        void activate(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs);

        /*!
        \brief Learn the predictor (update weights)
        \param cs is the ComputeSystem
        \param visibleCsPrev the previous visible (input) layer states
        \param hiddenTargetCs the target states that should be predicted
        */
        void learn(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCsPrev, const IntBuffer* hiddenTargetCs);

        /*!
        \brief Get the hidden activations (predictions)
        */
        const IntBuffer &getHiddenCs() const {
            return _hiddenCs;
        }

        /*!
        \brief Get the hidden size
        */
        const Int3 &getHiddenSize() const {
            return _hiddenSize;
        }

        /*!
        \brief Get the weights for a visible layer
        */
        const FloatBuffer &getWeights() {
            return _weights;
        }
    };
}

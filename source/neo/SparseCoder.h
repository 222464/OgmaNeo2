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
    \brief Pathfinding
    */
    int findNextIndex(int startIndex, int endIndex, int size, int weightsStart, const std::vector<float> &weights);

    /*!
    \brief Sparse Coder
    A 2D sparse coding layer, using Columnar Binary Sparse Coding (computes CSDR -> compressed CSDR)
    */
    class SparseCoder {
    public:
        /*!
        \brief Visible layer descriptor
        */
        struct VisibleLayerDesc {
            /*!
            \brief Visible layer size
            */
            Int3 _size;

            /*!
            \brief Radius onto hidden layer
            */
            int _radius;

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 4, 4, 32 }),
                _radius(2)
            {}
        };

        /*!
        \brief Visible layer
        */
        struct VisibleLayer {
            //!@{
            /*!
            \brief Visible layer values and buffers
            */
            FloatBuffer _weights;

            FloatBuffer _visibleActivations;

            IntBuffer _visibleReconCs;

            Float2 _visibleToHidden; // For projection
            Float2 _hiddenToVisible; // For projection

            Int2 _reverseRadii; // Pre-computed reverse radii
            //!@}
        };

    private:
        /*!
        \brief Size of the hidden layer
        */
        Int3 _hiddenSize;

        //!@{
        /*!
        \brief Buffers for hidden state
        */
        IntBuffer _hiddenCs;
        IntBuffer _hiddenCsPrev;

        FloatBuffer _hiddenActivations;

        FloatBuffer _hiddenTransitionWeights;

        IntBuffer _hiddenReconCs;
        IntBuffer _hiddenReconCsPrev;
        //!@}

        //!@{
        /*!
        \brief Visible layers and associated descriptors
        */
        std::vector<VisibleLayer> _visibleLayers;
        std::vector<VisibleLayerDesc> _visibleLayerDescs;
        //!@}

        //!@{
        /*!
        \brief Kernels
        */
        void init(int pos, std::mt19937 &rng, int vli);
        void forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputs, bool firstIter);
        void backward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputs, int vli);
        void pathfind(const Int2 &pos, std::mt19937 &rng, const IntBuffer* goalCs);
        void reconstruct(const Int2 &pos, std::mt19937 &rng, const IntBuffer* hiddenReconCs, int vli);
        void learnFeed(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputs, int vli);
        void learnTransition(const Int2 &pos, std::mt19937 &rng);

        static void initKernel(int pos, std::mt19937 &rng, SparseCoder* sc, int vli) {
            sc->init(pos, rng, vli);
        }

        static void forwardKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const std::vector<const IntBuffer*> &inputs, bool firstIter) {
            sc->forward(pos, rng, inputs, firstIter);
        }

        static void backwardKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const std::vector<const IntBuffer*> &inputs, int vli) {
            sc->backward(pos, rng, inputs, vli);
        }

        static void pathfindKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const IntBuffer* goalCs) {
            sc->pathfind(pos, rng, goalCs);
        }

        static void reconstructKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const IntBuffer* hiddenReconCs, int vli) {
            sc->reconstruct(pos, rng, hiddenReconCs, vli);
        }

        static void learnFeedKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const std::vector<const IntBuffer*> &inputs, int vli) {
            sc->learnFeed(pos, rng, inputs, vli);
        }

        static void learnTransitionKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc) {
            sc->learnTransition(pos, rng);
        }
        //!@}

    public:
        /*!
        \brief Learning rate for feed weights
        */
        float _alpha;

        /*!
        \brief Learning rate for transitions
        */
        float _beta;

        /*!
        \brief Explaining-away iterations (part of iterative sparse coding)
        */
        int _explainIters;

        /*!
        \brief Initialize defaults
        */
        SparseCoder()
        : _alpha(0.1f), _beta(0.1f), _explainIters(4)
        {}

        /*!
        \brief Create a sparse coding layer with random initialization
        \param cs is the ComputeSystem
        \param hiddenSize size of the hidden layer
        \param visibleLayerDescs the descriptors for the visible layers
        */
        void createRandom(ComputeSystem &cs,
            const Int3 &hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs);

        /*!
        \brief Activate the sparse coder (perform sparse coding)
        \param cs is the ComputeSystem
        \param visibleCs the visible (input) layer states
        */
        void activate(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs);

        /*!
        \brief Inference (determine actions)
        \param cs is the ComputeSystem
        \param goalCs target goal states
        */
        void infer(ComputeSystem &cs, const IntBuffer* goalCs);

        /*!
        \brief Learn the sparse code
        \param cs is the ComputeSystem.
        \param visibleCs the visible (input) layer states
        */
        void learn(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs);

        /*!
        \brief Get the number of visible layers
        */
        int getNumVisibleLayers() const {
            return _visibleLayers.size();
        }

        /*!
        \brief Get a visible layer
        */
        const VisibleLayer &getVisibleLayer(int index) const {
            return _visibleLayers[index];
        }

        /*!
        \brief Get a visible layer descriptor
        */
        const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        /*!
        \brief Get the hidden activations (state)
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
        \brief Get a visible layer's feed weights
        */
        const FloatBuffer &getWeights(int v) const {
            return _visibleLayers[v]._weights;
        }
    };
}

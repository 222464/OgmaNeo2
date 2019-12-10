// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseMatrix.h"

namespace ogmaneo {
class SparseCoder {
public:
    struct VisibleLayerDesc {
        Int3 _size;

        cl_int _radius;

        VisibleLayerDesc()
        :
        _size(8, 8, 16),
        _radius(2)
        {}
    };

    struct VisibleLayer {
        SparseMatrix _weights;
    };

private:
    Int3 _hiddenSize;

    SparseMatrix _laterals;

    cl::Buffer _hiddenCounts;
    
    DoubleBuffer _hiddenCs;

    cl::Buffer _hiddenStimulus;
    cl::Buffer _hiddenActivations;
    cl::Buffer _hiddenUsages;

    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;

    cl::Kernel _forwardKernel;
    cl::Kernel _inhibitKernel;
    cl::Kernel _learnKernel;
    cl::Kernel _usageKernel;

public:
    int _explainIters;

    SparseCoder()
    :
    _explainIters(3)
    {}

    void init(
        ComputeSystem &cs,
        ComputeProgram &prog,
        Int3 hiddenSize,
        int lateralRadius,
        const std::vector<VisibleLayerDesc> &visibleLayerDescs,
        std::mt19937 &rng
    );

    void step(
        ComputeSystem &cs,
        const std::vector<cl::Buffer> &inputCs,
        bool learnEnabled
    );

    void writeToStream(
        ComputeSystem &cs,
        std::ostream &os
    );

    void readFromStream(
        ComputeSystem &cs,
        ComputeProgram &prog,
        std::istream &is
    ); 

    int getNumVisibleLayers() const {
        return _visibleLayers.size();
    }

    const VisibleLayer &getVisibleLayer(
        int index
    ) const {
        return _visibleLayers[index];
    }

    const VisibleLayerDesc &getVisibleLayerDesc(
        int index
    ) const {
        return _visibleLayerDescs[index];
    }

    const cl::Buffer &getHiddenCs() const {
        return _hiddenCs[_front];
    }

    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }
};
} // namespace ogmaneo

// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::init(
    ComputeSystem &cs,
    ComputeProgram &prog,
    Int3 hiddenSize,
    int lateralRadius,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Counts
    _hiddenCounts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCounts, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));

    cl::Kernel countKernel = cl::Kernel(prog.getProgram(), "scCount");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        vl._weights.initLocalRF(cs, vld._size, _hiddenSize, vld._radius, 0.99f, 1.0f, rng);

        int argIndex = 0;

        countKernel.setArg(argIndex++, vl._weights._rowRanges);
        countKernel.setArg(argIndex++, _hiddenCounts);
        countKernel.setArg(argIndex++, vld._size);
        countKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(countKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    _laterals.initLocalRF(cs, _hiddenSize, _hiddenSize, lateralRadius, 0.0f, 0.01f, rng);

    // Hidden Cs
    _hiddenCs = createDoubleBuffer(cs, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs[_front], static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));

    _hiddenUsages = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenUsages, static_cast<cl_int>(0), 0, numHidden * sizeof(cl_int));

    // Hidden activations
    _hiddenStimulus = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "scInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "scLearn");
    _usageKernel = cl::Kernel(prog.getProgram(), "scUsage");
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<cl::Buffer> &visibleCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Initialize to 0
    cs.getQueue().enqueueFillBuffer(_hiddenStimulus, static_cast<cl_float>(0), 0, numHidden * sizeof(cl_float));
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0), 0, numHidden * sizeof(cl_float));

    // Forward
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _forwardKernel.setArg(argIndex++, visibleCs[vli]);
        _forwardKernel.setArg(argIndex++, _hiddenStimulus);
        _forwardKernel.setArg(argIndex++, vl._weights._nonZeroValues);
        _forwardKernel.setArg(argIndex++, vl._weights._rowRanges);
        _forwardKernel.setArg(argIndex++, vl._weights._columnIndices);
        _forwardKernel.setArg(argIndex++, vld._size);
        _forwardKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
    }

    // Inhibit
    for (int it = 0; it < _explainIters; it++) {
        cl_uchar enableLateralInhibition;

        if (it != 0) {
            enableLateralInhibition = 1;

            std::swap(_hiddenCs[_front], _hiddenCs[_back]);
        }
        else
            enableLateralInhibition = 0;
            
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenStimulus);
        _inhibitKernel.setArg(argIndex++, _hiddenActivations);
        _inhibitKernel.setArg(argIndex++, _hiddenCs[_back]);
        _inhibitKernel.setArg(argIndex++, _hiddenCs[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenCounts);
        _inhibitKernel.setArg(argIndex++, _laterals._nonZeroValues);
        _inhibitKernel.setArg(argIndex++, _laterals._rowRanges);
        _inhibitKernel.setArg(argIndex++, _laterals._columnIndices);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);
        _inhibitKernel.setArg(argIndex++, enableLateralInhibition);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    if (learnEnabled) {
        // Learn forward
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnKernel.setArg(argIndex++, visibleCs[vli]);
            _learnKernel.setArg(argIndex++, _hiddenCs[_front]);
            _learnKernel.setArg(argIndex++, _hiddenUsages);
            _learnKernel.setArg(argIndex++, vl._weights._nonZeroValues);
            _learnKernel.setArg(argIndex++, vl._weights._rowRanges);
            _learnKernel.setArg(argIndex++, vl._weights._columnIndices);
            _learnKernel.setArg(argIndex++, vld._size);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, _alpha);

            cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Learn lateral
        {
            int argIndex = 0;

            _learnKernel.setArg(argIndex++, _hiddenCs[_front]);
            _learnKernel.setArg(argIndex++, _hiddenCs[_front]);
            _learnKernel.setArg(argIndex++, _hiddenUsages);
            _learnKernel.setArg(argIndex++, _laterals._nonZeroValues);
            _learnKernel.setArg(argIndex++, _laterals._rowRanges);
            _learnKernel.setArg(argIndex++, _laterals._columnIndices);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, _alpha);

            cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }
    }

    // Usage update
    {
        int argIndex = 0;

        _usageKernel.setArg(argIndex++, _hiddenCs[_front]);
        _usageKernel.setArg(argIndex++, _hiddenUsages);
        _usageKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_usageKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void SparseCoder::writeToStream(
    ComputeSystem &cs,
    std::ostream &os
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_explainIters), sizeof(cl_int));
    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(cl_float));

    writeBufferToStream(cs, os, _hiddenCs[_front], numHiddenColumns * sizeof(cl_int));
    writeBufferToStream(cs, os, _hiddenUsages, numHidden * sizeof(cl_int));

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<const char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        vl._weights.writeToStream(cs, os);
    }

    _laterals.writeToStream(cs, os);
}

void SparseCoder::readFromStream(
    ComputeSystem &cs,
    ComputeProgram &prog,
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_explainIters), sizeof(cl_int));
    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));

    readBufferFromStream(cs, is, _hiddenCs[_front], numHiddenColumns * sizeof(cl_int));
    _hiddenCs[_back] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    readBufferFromStream(cs, is, _hiddenUsages, numHidden * sizeof(cl_int));

    _hiddenStimulus = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        vl._weights.readFromStream(cs, is);
    }

    _laterals.readFromStream(cs, is);

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "scInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "scLearn");
    _usageKernel = cl::Kernel(prog.getProgram(), "scUsage");

    // Counts
    _hiddenCounts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCounts, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));

    cl::Kernel countKernel = cl::Kernel(prog.getProgram(), "scCount");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        countKernel.setArg(argIndex++, vl._weights._rowRanges);
        countKernel.setArg(argIndex++, _hiddenCounts);
        countKernel.setArg(argIndex++, vld._size);
        countKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(countKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}
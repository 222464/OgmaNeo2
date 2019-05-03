// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

void Actor::init(
    ComputeSystem &cs,
    ComputeProgram &prog,
    Int3 hiddenSize,
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

    cl::Kernel countKernel = cl::Kernel(prog.getProgram(), "aCount");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        vl._weights.initLocalRF(cs, vld._size, Int3(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z), vld._radius, -0.001f, 0.001f, rng);
   
        int argIndex = 0;

        countKernel.setArg(argIndex++, vl._weights._rowRanges);
        countKernel.setArg(argIndex++, _hiddenCounts);
        countKernel.setArg(argIndex++, vld._size);
        countKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(countKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        vl._visibleCsPrev = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));

        cs.getQueue().enqueueFillBuffer(vl._visibleCsPrev, static_cast<cl_int>(0), 0, numVisibleColumns * sizeof(cl_int));
    }

    // Hidden Cs
    _hiddenCs = createDoubleBuffer(cs, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs[_front], static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Stimulus
    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));

    cs.getQueue().enqueueFillBuffer(_hiddenActivations[_front], static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}

void Actor::step(
    ComputeSystem &cs,
    const std::vector<cl::Buffer> &visibleCs,
    const cl::Buffer &hiddenCs,
    std::mt19937 &rng,
    float reward,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    std::swap(_hiddenCs[_front], _hiddenCs[_back]);
    std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations[_front], static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));
    // Compute feed stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _forwardKernel.setArg(argIndex++, visibleCs[vli]);
        _forwardKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _forwardKernel.setArg(argIndex++, vl._weights._nonZeroValues);
        _forwardKernel.setArg(argIndex++, vl._weights._rowRanges);
        _forwardKernel.setArg(argIndex++, vl._weights._columnIndices);
        _forwardKernel.setArg(argIndex++, vld._size);
        _forwardKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
    }

    // Activate
    {
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenCs[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Learn
    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnKernel.setArg(argIndex++, vl._visibleCsPrev);
            _learnKernel.setArg(argIndex++, _hiddenActivations[_front]);
            _learnKernel.setArg(argIndex++, _hiddenActivations[_back]);
            _learnKernel.setArg(argIndex++, hiddenCs);
            _learnKernel.setArg(argIndex++, _hiddenCounts);
            _learnKernel.setArg(argIndex++, vl._weights._nonZeroValues);
            _learnKernel.setArg(argIndex++, vl._weights._rowRanges);
            _learnKernel.setArg(argIndex++, vl._weights._columnIndices);
            _learnKernel.setArg(argIndex++, vld._size);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, _alpha);
            _learnKernel.setArg(argIndex++, _gamma);
            _learnKernel.setArg(argIndex++, reward);

            cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }
    }

    // Copy visibleCs
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        cs.getQueue().enqueueCopyBuffer(visibleCs[vli], vl._visibleCsPrev, 0, 0, numVisibleColumns * sizeof(cl_int));
    }
}

void Actor::writeToStream(ComputeSystem &cs, std::ostream &os) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(cl_float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(cl_float));

    std::vector<cl_int> hiddenCounts(numHiddenColumns);
    cs.getQueue().enqueueReadBuffer(_hiddenCounts, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCounts.data());
    os.write(reinterpret_cast<const char*>(hiddenCounts.data()), numHiddenColumns * sizeof(cl_int));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    cs.getQueue().enqueueReadBuffer(_hiddenCs[_front], CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());
    os.write(reinterpret_cast<const char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));

    std::vector<cl_float> hiddenActivations(numHidden);
    cs.getQueue().enqueueReadBuffer(_hiddenActivations[_front], CL_TRUE, 0, numHidden * sizeof(cl_float), hiddenActivations.data());
    os.write(reinterpret_cast<const char*>(hiddenActivations.data()), numHidden * sizeof(cl_float));

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<const char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        vl._weights.writeToStream(cs, os);

        std::vector<cl_int> visibleCsPrev(numVisibleColumns);
        cs.getQueue().enqueueReadBuffer(vl._visibleCsPrev, CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCsPrev.data());
        os.write(reinterpret_cast<const char*>(visibleCsPrev.data()), numVisibleColumns * sizeof(cl_int));
    }
}

void Actor::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(cl_float));

    std::vector<cl_int> hiddenCounts(numHiddenColumns);
    is.read(reinterpret_cast<char*>(hiddenCounts.data()), numHiddenColumns * sizeof(cl_int));
    _hiddenCounts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    cs.getQueue().enqueueWriteBuffer(_hiddenCounts, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCounts.data());

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    is.read(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));
    _hiddenCs = createDoubleBuffer(cs, numHiddenColumns * sizeof(cl_int));
    cs.getQueue().enqueueWriteBuffer(_hiddenCs[_front], CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());

    std::vector<cl_float> hiddenActivations(numHidden);
    is.read(reinterpret_cast<char*>(hiddenActivations.data()), numHidden * sizeof(cl_float));
    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));
    cs.getQueue().enqueueWriteBuffer(_hiddenActivations[_front], CL_TRUE, 0, numHidden * sizeof(cl_float), hiddenActivations.data());

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld._size.x * vld._size.y;

        vl._weights.readFromStream(cs, is);

        std::vector<cl_int> visibleCsPrev(numVisibleColumns);
        is.read(reinterpret_cast<char*>(visibleCsPrev.data()), numVisibleColumns * sizeof(cl_int));
        vl._visibleCsPrev = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));
        cs.getQueue().enqueueWriteBuffer(vl._visibleCsPrev, CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCsPrev.data());
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}
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
    int historyCapacity,
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

        vl._weights.initLocalRF(cs, vld._size, _hiddenSize, vld._radius, -0.01f, 0.01f, rng);

        int argIndex = 0;

        countKernel.setArg(argIndex++, vl._weights._rowRanges);
        countKernel.setArg(argIndex++, _hiddenCounts);
        countKernel.setArg(argIndex++, vld._size);
        countKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(countKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Stimulus
    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));

    // History samples
    _historySize = 0;
    _historySamples.resize(historyCapacity);

    for (int i = 0; i < _historySamples.size(); i++) {
        _historySamples[i]._visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            _historySamples[i]._visibleCs[vli] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));
        }

        _historySamples[i]._hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    }

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
        std::uniform_int_distribution<int> seedDist(0, 99999999);

        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenCs);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Add sample
    if (_historySize == _historySamples.size()) {
        // Circular buffer swap
        HistorySample temp = _historySamples.front();

        for (int i = 0; i < _historySamples.size() - 1; i++) {
            _historySamples[i] = _historySamples[i + 1];
        }

        _historySamples.back() = temp;
    }

    if (_historySize < _historySamples.size())
        _historySize++;
    
    {
        HistorySample &s = _historySamples[_historySize - 1];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            // Copy visible Cs
            cs.getQueue().enqueueCopyBuffer(visibleCs[vli], s._visibleCs[vli],
                0, 0, numVisibleColumns * sizeof(cl_int));
        }

        cs.getQueue().enqueueCopyBuffer(hiddenCs, s._hiddenCs, 0, 0, numHiddenColumns * sizeof(cl_int));

        s._reward = reward;
    }

    // Learn
    if (learnEnabled && _historySize > 1) {
        std::uniform_int_distribution<int> historyDist(0, _historySize - 2);

        for (int it = 0; it < _historyIters; it++) {
            int t = historyDist(rng);

            const HistorySample &s = _historySamples[t];
            const HistorySample &sNext = _historySamples[t + 1];

            // Initialize stimulus to 0
            cs.getQueue().enqueueFillBuffer(_hiddenActivations[_back], static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));
            cs.getQueue().enqueueFillBuffer(_hiddenActivations[_front], static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

            // Compute feed stimulus
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                // s
                {
                    int argIndex = 0;

                    _forwardKernel.setArg(argIndex++, s._visibleCs[vli]);
                    _forwardKernel.setArg(argIndex++, _hiddenActivations[_back]);
                    _forwardKernel.setArg(argIndex++, vl._weights._nonZeroValues);
                    _forwardKernel.setArg(argIndex++, vl._weights._rowRanges);
                    _forwardKernel.setArg(argIndex++, vl._weights._columnIndices);
                    _forwardKernel.setArg(argIndex++, vld._size);
                    _forwardKernel.setArg(argIndex++, _hiddenSize);

                    cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
                }

                // sNext
                {
                    int argIndex = 0;

                    _forwardKernel.setArg(argIndex++, sNext._visibleCs[vli]);
                    _forwardKernel.setArg(argIndex++, _hiddenActivations[_front]);
                    _forwardKernel.setArg(argIndex++, vl._weights._nonZeroValues);
                    _forwardKernel.setArg(argIndex++, vl._weights._rowRanges);
                    _forwardKernel.setArg(argIndex++, vl._weights._columnIndices);
                    _forwardKernel.setArg(argIndex++, vld._size);
                    _forwardKernel.setArg(argIndex++, _hiddenSize);

                    cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
                }
            }

            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                int argIndex = 0;

                _learnKernel.setArg(argIndex++, s._visibleCs[vli]);
                _learnKernel.setArg(argIndex++, _hiddenActivations[_front]);
                _learnKernel.setArg(argIndex++, _hiddenActivations[_back]);
                _learnKernel.setArg(argIndex++, sNext._hiddenCs);
                _learnKernel.setArg(argIndex++, _hiddenCounts);
                _learnKernel.setArg(argIndex++, vl._weights._nonZeroValues);
                _learnKernel.setArg(argIndex++, vl._weights._rowRanges);
                _learnKernel.setArg(argIndex++, vl._weights._columnIndices);
                _learnKernel.setArg(argIndex++, vld._size);
                _learnKernel.setArg(argIndex++, _hiddenSize);
                _learnKernel.setArg(argIndex++, _alpha);
                _learnKernel.setArg(argIndex++, _gamma);
                _learnKernel.setArg(argIndex++, _tau);
                _learnKernel.setArg(argIndex++, sNext._reward);

                cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
            }
        }
    }
}

void Actor::writeToStream(ComputeSystem &cs, std::ostream &os) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(cl_float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(cl_float));
    os.write(reinterpret_cast<const char*>(&_tau), sizeof(cl_float));
    os.write(reinterpret_cast<const char*>(&_historyIters), sizeof(int));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    cs.getQueue().enqueueReadBuffer(_hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());
    os.write(reinterpret_cast<const char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));

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

    int historyCapacity = _historySamples.size();

    os.write(reinterpret_cast<const char*>(&historyCapacity), sizeof(int));
    os.write(reinterpret_cast<const char*>(&_historySize), sizeof(int));

    for (int i = 0; i < _historySamples.size(); i++) {
        HistorySample &s = _historySamples[i];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;
            int numVisible = numVisibleColumns * vld._size.z;

            std::vector<cl_int> visibleCs(numVisibleColumns);
            cs.getQueue().enqueueReadBuffer(s._visibleCs[vli], CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
            os.write(reinterpret_cast<const char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
        }

        std::vector<cl_int> hiddenCs(numHiddenColumns);
        cs.getQueue().enqueueReadBuffer(s._hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());
        os.write(reinterpret_cast<const char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));

        os.write(reinterpret_cast<const char*>(&s._reward), sizeof(cl_float));
    }
}

void Actor::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_tau), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_historyIters), sizeof(int));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    is.read(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    cs.getQueue().enqueueWriteBuffer(_hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());

    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));

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

    int historyCapacity;

    is.read(reinterpret_cast<char*>(&historyCapacity), sizeof(int));
    is.read(reinterpret_cast<char*>(&_historySize), sizeof(int));

    _historySamples.resize(historyCapacity);

    for (int i = 0; i < _historySamples.size(); i++) {
        _historySamples[i]._visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            std::vector<cl_int> visibleCs(numVisibleColumns);
            is.read(reinterpret_cast<char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
            _historySamples[i]._visibleCs[vli] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));
            cs.getQueue().enqueueWriteBuffer(_historySamples[i]._visibleCs[vli], CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
        }

        std::vector<cl_int> hiddenCs(numHiddenColumns);
        is.read(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));
        _historySamples[i]._hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
        cs.getQueue().enqueueWriteBuffer(_historySamples[i]._hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());

        is.read(reinterpret_cast<char*>(&_historySamples[i]._reward), sizeof(cl_float));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}
// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace ogmaneo;

void ImageEncoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    std::vector<float> sum0s(_hiddenSize.z);
    std::vector<float> activations(_hiddenSize.z);
    
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);
        
        if (_hiddenStatuses[hiddenIndex] == 0)
            continue;

        float sum0 = 0.0f;
        float sum1 = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum0 += vl._weights.addMins(*inputActs[vli], hiddenIndex);
            sum1 += vl._weights.total(hiddenIndex);
        }

        sum0s[hc] = sum0;
        activations[hc] = sum0 / (_alpha + sum1);

        if (activations[hc] > maxActivation) {
            maxActivation = activations[hc];
            maxIndex = hc;
        }
    }

    bool search = true;
    bool commit = false;

    if (maxIndex == -1) {
        maxIndex = 0;
        search = false;
        commit = true;
    }

    bool found = true;

    if (search) {
        int originalMaxIndex = maxIndex;
        
        bool reset;

        int iters = 0;

        while (true) {
            reset = false;

            iters++;

            int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), _hiddenSize);

            // For each visible layer
            float sum2 = 0.0f;

            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                sum2 += vl._weights.counts(*inputActs[vli], hiddenIndexMax);
            }

            // Check vigilance
            float match = sum0s[maxIndex] / std::max(0.0001f, sum2);
            
            if (match < _minVigilance) {
                // Reset
                reset = true;

                // Deactivate unit
                activations[maxIndex] = -1.0f;

                maxIndex = 0;
                maxActivation = -999999.0f;

                int numAvailable = 0;

                for (int hc = 0; hc < _hiddenSize.z; hc++) {
                    int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

                    if (_hiddenStatuses[hiddenIndex] == 0 || activations[hc] < 0.0f)
                        continue;

                    numAvailable++;

                    if (activations[hc] > maxActivation) {
                        maxActivation = activations[hc];
                        maxIndex = hc;
                    }
                }
                
                if (numAvailable == 0)
                    break;
            }
            else
                break;
        }

        // If ended in reset
        if (reset) {
            // If uncommitted nodes present
            int uncommittedIndex = -1;

            for (int hc = 0; hc < _hiddenSize.z; hc++) {
                int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

                if (_hiddenStatuses[hiddenIndex] == 0) {
                    uncommittedIndex = hc;

                    break;
                }
            }

            // If no uncommitted
            if (uncommittedIndex == -1) {
                maxIndex = originalMaxIndex;
                found = false;
            }
            else { // Found uncommitted
                maxIndex = uncommittedIndex;
                commit = true;
            }
        }
    }

    int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), _hiddenSize);

    _hiddenCs[hiddenColumnIndex] = maxIndex;

    _hiddenStatuses[hiddenIndexMax] = 1;

    if (learnEnabled) {
        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            if (found)
                vl._weights.hebbDecreasing(*inputActs[vli], hiddenIndexMax, commit ? 1.0f : _beta);
            //else
            //    vl._weights.hebb(*inputActs[vli], hiddenIndexMax, _beta);
        }
    }
}

void ImageEncoder::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights, 0.0f, cs._rng);

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = 1.0f;
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);

    _hiddenStatuses = IntBuffer(numHidden, 0); // Uncommitted status
}

void ImageEncoder::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &visibleCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, visibleCs, learnEnabled);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void ImageEncoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_minVigilance), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);

    writeBufferToStream(os, &_hiddenStatuses);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        writeSMToStream(os, vl._weights);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_minVigilance), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);

    readBufferFromStream(is, &_hiddenStatuses);

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

        readSMFromStream(is, vl._weights);
    }
}
// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace ogmaneo;

// Kernels
void Predictor::init(int pos, std::mt19937 &rng) {
    // Randomly initialize weights in range
	std::uniform_real_distribution<float> weightDist(-0.01f, 0.01f);

    _weights[pos] = weightDist(rng);
}

void Predictor::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs) {
    // Cache address calculations (taken from addressN functions)
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * _hiddenSize.z;

    // ------------------------------ Action ------------------------------

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        Int3 hiddenPosition(pos.x, pos.y, hc);

        // Partially computed address of weight
        int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

        float sum = 0.0f;
    
        // Center of projected position
        Int2 visiblePositionCenter = project(pos, _hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - _radius, visiblePositionCenter.y - _radius);

        // Additional addressing dimensions
        int diam = _radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(_visibleSize.x - 1, visiblePositionCenter.x + _radius), std::min(_visibleSize.y - 1, visiblePositionCenter.y + _radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleIndex = address2(visiblePosition, _visibleSize.x);
                
                {
                    int visibleC0 = (*inputCs[0])[visibleIndex];
                
                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC0 * diam2;

                    sum -= _weights[dPartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                }

                {
                    int visibleC1 = (*inputCs[1])[visibleIndex];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC1 * diam2;

                    sum += _weights[dPartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                }
            }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCs[address2(pos, _hiddenSize.x)] = maxIndex;
}

void Predictor::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCsPrev, const IntBuffer* hiddenTargetCs) {
    // Cache address calculations
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * _hiddenSize.z;

    int hiddenIndex = address2(pos, _hiddenSize.x);

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        Int3 hiddenPosition(pos.x, pos.y, hc);

        // Partially computed address of weight
        int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

        float sum = 0.0f;
        
        // Center of projected position
        Int2 visiblePositionCenter = project(pos, _hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - _radius, visiblePositionCenter.y - _radius);

        // Additional addressing dimensions
        int diam = _radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(_visibleSize.x - 1, visiblePositionCenter.x + _radius), std::min(_visibleSize.y - 1, visiblePositionCenter.y + _radius));

        float count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    
        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                {
                    int visibleCPrev0 = (*inputCsPrev[0])[address2(visiblePosition, _visibleSize.x)];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev0 * diam2;

                    sum -= _weights[dPartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                }

                {
                    int visibleCPrev1 = (*inputCsPrev[1])[address2(visiblePosition, _visibleSize.x)];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev1 * diam2;

                    sum += _weights[dPartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                }
            }

        float delta = _alpha * ((hc == (*hiddenTargetCs)[hiddenIndex] ? 1.0f : 0.0f) - sigmoid(sum / std::max(1.0f, count)));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                {
                    int visibleCPrev0 = (*inputCsPrev[0])[address2(visiblePosition, _visibleSize.x)];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev0 * diam2;

                    _weights[dPartial + az * dxyz] -= delta; // Used cached parts to compute weight address, equivalent to calling address4
                }

                {
                    int visibleCPrev1 = (*inputCsPrev[1])[address2(visiblePosition, _visibleSize.x)];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev1 * diam2;

                    _weights[dPartial + az * dxyz] += delta; // Used cached parts to compute weight address, equivalent to calling address4
                }
            }
    }
}

void Predictor::createRandom(ComputeSystem &cs,
    const Int3 &hiddenSize, const Int3 &visibleSize, int radius)
{
    _hiddenSize = hiddenSize;
    _visibleSize = visibleSize;
    _radius = radius;

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    _hiddenToVisible = Float2(static_cast<float>(_visibleSize.x) / static_cast<float>(_hiddenSize.x),
        static_cast<float>(_visibleSize.y) / static_cast<float>(_hiddenSize.y));

    int diam = _radius * 2 + 1;

    int numWeightsPerHidden = diam * diam * _visibleSize.z;

    int weightsSize = numHidden * numWeightsPerHidden;

    // Create weight matrix for this visible layer and initialize randomly
    _weights = FloatBuffer(weightsSize);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < weightsSize; x++)
        init(x, cs._rng);
#else
    runKernel1(cs, std::bind(Predictor::initKernel, std::placeholders::_1, std::placeholders::_2, this), weightsSize, cs._rng, cs._batchSize1);
#endif

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif
}

void Predictor::activate(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs) {
    assert(visibleCs.size() == 2);

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, visibleCs);
#else
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Predictor::learn(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCsPrev, const IntBuffer* hiddenTargetCs) {
    assert(visibleCsPrev.size() == 2);
    
    // Learn kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            learn(Int2(x, y), cs._rng, visibleCsPrev, hiddenTargetCs);
#else
    runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCsPrev, hiddenTargetCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}
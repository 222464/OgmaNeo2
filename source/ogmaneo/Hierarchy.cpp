// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEOLICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>
#include <iostream>

using namespace ogmaneo;

void Hierarchy::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    int vli,
    const IntBuffer* inputCs
) {
    if (l == 0) {
        int visibleColumnIndex = address2(pos, Int2(inputSizes[vli].x, inputSizes[vli].y));

        float maxValue = -999999.0f;

        if (inputCs == nullptr) {
            for (int vc = 0; vc < inputSizes[vli].z; vc++) {
                int visibleIndex = address3(Int3(pos.x, pos.y, vc), inputSizes[vli]);

                float sum;

                if (l == scLayers.size() - 1)
                    sum = rLayers[l].weights[vli].multiplyOHVs(*hiddenCs, visibleIndex, scLayers[l].getHiddenSize().z);
                else
                    sum = rLayers[l].weights[vli].multiplyOHVs(*hiddenCs, rLayers[l + 1].activations[0], visibleIndex, scLayers[l].getHiddenSize().z);

                sum /= std::max(1, rLayers[l].visibleCounts[vli][visibleColumnIndex]);
            
                if (sum > maxValue) {
                    maxValue = sum;

                    actions[vli][visibleColumnIndex] = vc;
                }
            }
        }
        else {
            int visibleIndex = address3(Int3(pos.x, pos.y, (*inputCs)[visibleColumnIndex]), inputSizes[vli]);

            float sum;

            if (l == scLayers.size() - 1)
                sum = rLayers[l].weights[vli].multiplyOHVs(*hiddenCs, visibleIndex, scLayers[l].getHiddenSize().z);
            else
                sum = rLayers[l].weights[vli].multiplyOHVs(*hiddenCs, rLayers[l + 1].activations[0], visibleIndex, scLayers[l].getHiddenSize().z);

            sum /= std::max(1, rLayers[l].visibleCounts[vli][visibleColumnIndex]);
            
            maxValue = sum;
        }

        rLayers[l].activations[vli][visibleColumnIndex] = maxValue;
    }
    else {
        int visibleColumnIndex = address2(pos, Int2(scLayers[l - 1].getHiddenSize().x, scLayers[l - 1].getHiddenSize().y));

        int visibleIndex = address3(Int3(pos.x, pos.y, (*inputCs)[visibleColumnIndex]), scLayers[l - 1].getHiddenSize());

        float sum;

        if (l == scLayers.size() - 1)
            sum = rLayers[l].weights[vli].multiplyOHVs(*hiddenCs, visibleIndex, scLayers[l].getHiddenSize().z);
        else
            sum = rLayers[l].weights[vli].multiplyOHVs(*hiddenCs, rLayers[l + 1].activations[0], visibleIndex, scLayers[l].getHiddenSize().z);

        sum /= std::max(1, rLayers[l].visibleCounts[vli][visibleColumnIndex]);
        
        rLayers[l].activations[vli][visibleColumnIndex] = sum;
    }
}

void Hierarchy::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(scLayers[l].getHiddenSize().x, scLayers[l].getHiddenSize().y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, (*hiddenCs)[hiddenColumnIndex]), scLayers[l].getHiddenSize());

    if (l == 0) {
        //float act = rLayers[l + 1].activations[0][hiddenColumnIndex];

        float error = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < rLayers[l].weights.size(); vli++) {
            if (!actions[vli].empty())
                error += rLayers[l].weights[vli].multiplyOHVsT(*inputCs[vli], rLayers[l].errors[vli], hiddenIndex, inputSizes[vli].z);
        }

        error /= std::max(1, rLayers[l].hiddenCounts[hiddenColumnIndex]);

        rLayers[l + 1].errors[0][hiddenColumnIndex] = error;// * act * (1.0f - act);
    }
    else {
        //float act = rLayers[l + 1].activations[0][hiddenColumnIndex];

        float error = rLayers[l].weights[0].multiplyOHVsT(*inputCs[0], rLayers[l].errors[0], hiddenIndex, scLayers[l - 1].getHiddenSize().z);

        error /= std::max(1, rLayers[l].hiddenCounts[hiddenColumnIndex]);
        
        rLayers[l + 1].errors[0][hiddenColumnIndex] = error;// * act * (1.0f - act);
    }
}

void Hierarchy::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int l,
    int vli,
    const IntBuffer* inputCs
) {
    if (l == 0) {
        int visibleColumnIndex = address2(pos, Int2(inputSizes[vli].x, inputSizes[vli].y));

        int visibleIndex = address3(Int3(pos.x, pos.y, (*inputCs)[visibleColumnIndex]), inputSizes[vli]);

        float delta = alpha * std::tanh(rLayers[l].errors[vli][visibleColumnIndex]);
        
        if (l == scLayers.size() - 1)
            rLayers[l].weights[vli].deltaOHVs(*hiddenCs, delta, visibleIndex, scLayers[l].getHiddenSize().z);
        else
            rLayers[l].weights[vli].deltaOHVs(*hiddenCs, rLayers[l + 1].activations[0], delta, visibleIndex, scLayers[l].getHiddenSize().z);
    }
    else {
        int visibleColumnIndex = address2(pos, Int2(scLayers[l - 1].getHiddenSize().x, scLayers[l - 1].getHiddenSize().y));

        int visibleIndex = address3(Int3(pos.x, pos.y, (*inputCs)[visibleColumnIndex]), scLayers[l - 1].getHiddenSize());

        float delta = beta * std::tanh(rLayers[l].errors[vli][visibleColumnIndex]);
        
        if (l == scLayers.size() - 1)
            rLayers[l].weights[vli].deltaOHVs(*hiddenCs, delta, visibleIndex, scLayers[l].getHiddenSize().z);
        else
            rLayers[l].weights[vli].deltaOHVs(*hiddenCs, rLayers[l + 1].activations[0], delta, visibleIndex, scLayers[l].getHiddenSize().z);
    }
}

void Hierarchy::initRandom(
    ComputeSystem &cs,
    const std::vector<Int3> &inputSizes,
    const std::vector<InputType> &inputTypes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    scLayers.resize(layerDescs.size());
    rLayers.resize(layerDescs.size());

    ticks.assign(layerDescs.size(), 0);

    histories.resize(layerDescs.size());
    historySizes.resize(layerDescs.size());
    
    ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    updates.resize(layerDescs.size(), false);

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l].ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        std::normal_distribution<float> weightDistLow(0.0f, 0.01f);
        std::uniform_real_distribution<float> weightDistHigh(0.99f, 1.0f);

        // Histories for all input layers or just the one sparse coder (if not the first layer)
        histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l].temporalHorizon : layerDescs[l].temporalHorizon);

        historySizes[l].resize(histories[l].size());
		
        // Create sparse coder visible layer descriptors
        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon);

            actions.resize(inputSizes.size());

            rLayers[l].weights.resize(inputSizes.size());
            rLayers[l].visibleCounts.resize(inputSizes.size());

            rLayers[l].activations.resize(inputSizes.size());
            rLayers[l].errors.resize(inputSizes.size());

            rLayers[l].hiddenCounts = IntBuffer(layerDescs[l].hiddenSize.x * layerDescs[l].hiddenSize.y, 0);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    scVisibleLayerDescs[index].size = inputSizes[i];
                    scVisibleLayerDescs[index].radius = layerDescs[l].ffRadius;
                }

                if (inputTypes[i] == InputType::action) {
                    actions[i] = IntBuffer(inputSizes[i].x * inputSizes[i].y, 0);
                    rLayers[l].activations[i] = FloatBuffer(inputSizes[i].x * inputSizes[i].y, 0.0f);
                    rLayers[l].errors[i] = FloatBuffer(inputSizes[i].x * inputSizes[i].y, 0.0f);

                    initSMLocalRF(layerDescs[l].hiddenSize, inputSizes[i], layerDescs[l].rRadius, rLayers[l].weights[i]);

                    rLayers[l].weights[i].initT();

                    // Init weights
                    for (int j = 0; j < rLayers[l].weights[i].nonZeroValues.size(); j++)
                        rLayers[l].weights[i].nonZeroValues[j] = weightDistLow(cs.rng);

                    for (int x = 0; x < layerDescs[l].hiddenSize.x; x++)
                        for (int y = 0; y < layerDescs[l].hiddenSize.y; y++) {
                            int hiddenColumnIndex = address2(Int2(x, y), Int2(layerDescs[l].hiddenSize.x, layerDescs[l].hiddenSize.y));
                            int hiddenIndex = address3(Int3(x, y, 0), layerDescs[l].hiddenSize);

                            rLayers[l].hiddenCounts[hiddenColumnIndex] += rLayers[l].weights[i].countT(hiddenIndex) / inputSizes[i].z;
                        }

                    rLayers[l].visibleCounts[i] = IntBuffer(actions[i].size());

                    for (int x = 0; x < inputSizes[i].x; x++)
                        for (int y = 0; y < inputSizes[i].y; y++) {
                            int visibleColumnIndex = address2(Int2(x, y), Int2(inputSizes[i].x, inputSizes[i].y));
                            int visibleIndex = address3(Int3(x, y, 0), inputSizes[i]);

                            rLayers[l].visibleCounts[i][visibleColumnIndex] = rLayers[l].weights[i].count(visibleIndex) / layerDescs[l].hiddenSize.z;
                        }    
                }
            }
            
            // Initialize history buffers
			for (int v = 0; v < histories[l].size(); v++) {
				int i = v / layerDescs[l].temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                historySizes[l][v] = inSize;
			}
        }
        else {
            scVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            rLayers[l].weights.resize(1);
            rLayers[l].visibleCounts.resize(1);

            rLayers[l].activations.resize(1);
            rLayers[l].errors.resize(1);

            rLayers[l].hiddenCounts = IntBuffer(layerDescs[l].hiddenSize.x * layerDescs[l].hiddenSize.y, 0);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                scVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                scVisibleLayerDescs[t].radius = layerDescs[l].ffRadius;
            }

            rLayers[l].activations[0] = FloatBuffer(layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y, 0.0f);
            rLayers[l].errors[0] = FloatBuffer(layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y, 0.0f);

            initSMLocalRF(layerDescs[l].hiddenSize, layerDescs[l - 1].hiddenSize, layerDescs[l].rRadius, rLayers[l].weights[0]);

            rLayers[l].weights[0].initT();

            // Init weights
            for (int j = 0; j < rLayers[l].weights[0].nonZeroValues.size(); j++)
                rLayers[l].weights[0].nonZeroValues[j] = weightDistHigh(cs.rng);

            for (int x = 0; x < layerDescs[l].hiddenSize.x; x++)
                for (int y = 0; y < layerDescs[l].hiddenSize.y; y++) {
                    int hiddenColumnIndex = address2(Int2(x, y), Int2(layerDescs[l].hiddenSize.x, layerDescs[l].hiddenSize.y));
                    int hiddenIndex = address3(Int3(x, y, 0), layerDescs[l].hiddenSize);

                    rLayers[l].hiddenCounts[hiddenColumnIndex] += rLayers[l].weights[0].countT(hiddenIndex) / layerDescs[l - 1].hiddenSize.z;
                }

            rLayers[l].visibleCounts[0] = IntBuffer(layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y);

            for (int x = 0; x < layerDescs[l - 1].hiddenSize.x; x++)
                for (int y = 0; y < layerDescs[l - 1].hiddenSize.y; y++) {
                    int visibleColumnIndex = address2(Int2(x, y), Int2(layerDescs[l - 1].hiddenSize.x, layerDescs[l - 1].hiddenSize.y));
                    int visibleIndex = address3(Int3(x, y, 0), layerDescs[l - 1].hiddenSize);

                    rLayers[l].visibleCounts[0][visibleColumnIndex] = rLayers[l].weights[0].count(visibleIndex) / layerDescs[l].hiddenSize.z;
                }

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

			for (int v = 0; v < histories[l].size(); v++) {
                histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                historySizes[l][v] = inSize;
            }
        }

        // Create the sparse coding layer
        scLayers[l].initRandom(cs, layerDescs[l].hiddenSize, scVisibleLayerDescs);
    }
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    scLayers = other.scLayers;
    rLayers = other.rLayers;

    qs = other.qs;
    actions = other.actions;

    historySizes = other.historySizes;
    updates = other.updates;
    ticks = other.ticks;
    ticksPerUpdate = other.ticksPerUpdate;
    inputSizes = other.inputSizes;

    histories.resize(other.histories.size());

    for (int l = 0; l < scLayers.size(); l++) {
        histories[l].resize(other.histories[l].size());

        for (int v = 0; v < histories[l].size(); v++) {
            histories[l][v] = std::make_shared<IntBuffer>();
            
            (*histories[l][v]) = (*other.histories[l][v]);
        }
    }

    alpha = other.alpha;
    beta = other.beta;
    gamma = other.gamma;
    maxHistorySamples = other.maxHistorySamples;
    historyIters = other.historyIters;

    historySamples.resize(other.historySamples.size());

    for (int t = 0; t < historySamples.size(); t++) {
        historySamples[t] = std::make_shared<HistorySample>();

        (*historySamples[t]) = (*other.historySamples[t]);
    }

    return *this;
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    float reward,
    bool learnEnabled
) {
    assert(inputCs.size() == inputSizes.size());

    // First tick is always 0
    ticks[0] = 0;

    // Add input to first layer history   
    {
        int temporalHorizon = histories.front().size() / inputSizes.size();

        std::vector<std::shared_ptr<IntBuffer>> lasts(inputSizes.size());
        
        for (int i = 0; i < inputSizes.size(); i++)
            lasts[i] = histories.front()[temporalHorizon - 1 + temporalHorizon * i];
  
        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int i = 0; i < inputSizes.size(); i++) {
                // Shift
                histories.front()[t + temporalHorizon * i] = histories.front()[(t - 1) + temporalHorizon * i];
            }
        }

        for (int i = 0; i < inputSizes.size(); i++) {
            assert(inputSizes[i].x * inputSizes[i].y == inputCs[i]->size());
            
            // Copy
#ifdef KERNEL_NO_THREAD
            for (int x = 0; x < inputCs[i]->size(); x++)
                copyInt(x, cs.rng, inputCs[i], lasts[i].get());
#else
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[i], lasts[i].get()), inputCs[i]->size(), cs.rng, cs.batchSize1);
#endif

            histories.front()[0 + temporalHorizon * i] = lasts[i];
        }
    }

    // Set all updates to no update, will be set to true if an update occurred later
    updates.clear();
    updates.resize(scLayers.size(), false);

    std::shared_ptr<HistorySample> ns = std::make_shared<HistorySample>();

    ns->states.resize(scLayers.size());
    ns->reward = reward;

    // Forward
    for (int l = 0; l < scLayers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;
            
            // Activate sparse coder
            scLayers[l].step(cs, constGet(histories[l]), learnEnabled);

            // Add to next layer's history
            if (l < scLayers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = histories[lNext].size();

                std::shared_ptr<IntBuffer> last = histories[lNext].back();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    histories[lNext][t] = histories[lNext][t - 1];

                // Copy
#ifdef KERNEL_NO_THREAD
                for (int x = 0; x < scLayers[l].getHiddenCs().size(); x++)
                    copyInt(x, cs.rng, &scLayers[l].getHiddenCs(), last.get());
#else
                runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &scLayers[l].getHiddenCs(), last.get()), scLayers[l].getHiddenCs().size(), cs.rng, cs.batchSize1);
#endif

                histories[lNext].front() = last;

                ticks[lNext]++;
            }
        }

        ns->states[l] = scLayers[l].getHiddenCs();
    }

    // Action into replay buffer
    ns->actionsPrev.resize(actions.size());

    for (int vli = 0; vli < actions.size(); vli++)
        if (!actions[vli].empty())
            ns->actionsPrev[vli] = *inputCs[vli];

    // Add history sample
    historySamples.insert(historySamples.begin(), ns);

    if (historySamples.size() > maxHistorySamples)
        historySamples.resize(maxHistorySamples);

    // Learn
    if (learnEnabled && historySamples.size() > 2) {
        // Find latest Q on-policy
        const HistorySample &s = *historySamples[1];
        const HistorySample &sNext = *historySamples[0];
            
        for (int l = scLayers.size() - 1; l >= 0; l--) {
            if (l == 0) {
                for (int i = 0; i < inputSizes.size(); i++) {
                    if (actions[i].empty())
                        continue;

#ifdef KERNEL_NO_THREAD
                    for (int x = 0; x < inputSizes[i].x; x++)
                        for (int y = 0; y < inputSizes[i].y; y++)
                            forward(Int2(x, y), cs.rng, &s.states[l], l, i, &sNext.actionsPrev[i]);
#else
                    runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, i, &sNext.actionsPrev[i]), Int2(inputSizes[i].x, inputSizes[i].y), cs.rng, cs.batchSize2);
#endif
                }
            }
            else {
#ifdef KERNEL_NO_THREAD
                for (int x = 0; x < scLayers[l - 1].getHiddenSize().x; x++)
                    for (int y = 0; y < scLayers[l - 1].getHiddenSize().y; y++)
                        forward(Int2(x, y), cs.rng, &s.states[l], l, 0, &s.states[l - 1]);
#else
                runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, 0, &s.states[l - 1]), Int2(scLayers[l - 1].getHiddenSize().x, scLayers[l - 1].getHiddenSize().y), cs.rng, cs.batchSize2);
#endif
            }
        }

        // Keep predicted Q values
        qs = rLayers.front().activations;

        std::uniform_int_distribution<int> sampleDist(2, historySamples.size() - 1);

        for (int it = 0; it < historyIters; it++) {
            int t = sampleDist(cs.rng);

            const HistorySample &s = *historySamples[t];
            const HistorySample &sNext = *historySamples[t - 1];

            for (int l = scLayers.size() - 1; l >= 0; l--) {
                if (l == 0) {
                    for (int i = 0; i < inputSizes.size(); i++) {
                        if (actions[i].empty())
                            continue;

#ifdef KERNEL_NO_THREAD
                        for (int x = 0; x < inputSizes[i].x; x++)
                            for (int y = 0; y < inputSizes[i].y; y++)
                                forward(Int2(x, y), cs.rng, &s.states[l], l, i, &sNext.actionsPrev[i]);
#else
                        runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, i, &sNext.actionsPrev[i]), Int2(inputSizes[i].x, inputSizes[i].y), cs.rng, cs.batchSize2);
#endif
                    }
                }
                else {
#ifdef KERNEL_NO_THREAD
                    for (int x = 0; x < scLayers[l - 1].getHiddenSize().x; x++)
                        for (int y = 0; y < scLayers[l - 1].getHiddenSize().y; y++)
                            forward(Int2(x, y), cs.rng, &s.states[l], l, 0, &s.states[l - 1]);
#else
                    runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, 0, &s.states[l - 1]), Int2(scLayers[l - 1].getHiddenSize().x, scLayers[l - 1].getHiddenSize().y), cs.rng, cs.batchSize2);
#endif
                }
            }

            // Determine target
            float baseQ = 0.0f;
            float g = 1.0f;
            
            for (int t2 = t - 1; t2 >= 1; t2--) {
                baseQ += g * historySamples[t2]->reward;

                g *= gamma;
            }

            // Errors
            for (int vli = 0; vli < rLayers.front().errors.size(); vli++)
                for (int i = 0; i < rLayers.front().errors[vli].size(); i++) {
                    float targetQ = baseQ + g * qs[vli][i];

                    rLayers.front().errors[vli][i] = targetQ - rLayers.front().activations[vli][i];
                }

            // Backward
            for (int l = 0; l < scLayers.size() - 1; l++) {
                if (l == 0) {
#ifdef KERNEL_NO_THREAD
                    for (int x = 0; x < scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < scLayers[l].getHiddenSize().y; y++)
                            backward(Int2(x, y), cs.rng, &s.states[l], l, constGet(sNext.actionsPrev));
#else
                    runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, constGet(sNext.actionsPrev)), Int2(scLayers[l].getHiddenSize().x, scLayers[l].getHiddenSize().y), cs.rng, cs.batchSize2);
#endif
                }
                else {
#ifdef KERNEL_NO_THREAD
                    for (int x = 0; x < scLayers[l].getHiddenSize().x; x++)
                        for (int y = 0; y < scLayers[l].getHiddenSize().y; y++)
                            backward(Int2(x, y), cs.rng, &s.states[l], l, std::vector<const IntBuffer*>{ &s.states[l - 1] });
#else
                    runKernel2(cs, std::bind(Hierarchy::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, std::vector<const IntBuffer*>{ &s.states[l - 1] }), Int2(scLayers[l].getHiddenSize().x, scLayers[l].getHiddenSize().y), cs.rng, cs.batchSize2);
#endif
                }
            }

            // Learn
            for (int l = scLayers.size() - 1; l >= 0; l--) {
                if (l == 0) {
                    for (int i = 0; i < inputSizes.size(); i++) {
                        if (actions[i].empty())
                            continue;

#ifdef KERNEL_NO_THREAD
                        for (int x = 0; x < inputSizes[i].x; x++)
                            for (int y = 0; y < inputSizes[i].y; y++)
                                learn(Int2(x, y), cs.rng, &s.states[l], l, i, &sNext.actionsPrev[i]);
#else
                        runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, i, &sNext.actionsPrev[i]), Int2(inputSizes[i].x, inputSizes[i].y), cs.rng, cs.batchSize2);
#endif
                    }
                }
                else {
#ifdef KERNEL_NO_THREAD
                    for (int x = 0; x < scLayers[l - 1].getHiddenSize().x; x++)
                        for (int y = 0; y < scLayers[l - 1].getHiddenSize().y; y++)
                            learn(Int2(x, y), cs.rng, &s.states[l], l, 0, &s.states[l - 1]);
#else
                    runKernel2(cs, std::bind(Hierarchy::learnKernel, std::placeholders::_1, std::placeholders::_2, this, &s.states[l], l, 0, &s.states[l - 1]), Int2(scLayers[l - 1].getHiddenSize().x, scLayers[l - 1].getHiddenSize().y), cs.rng, cs.batchSize2);
#endif
                }
            }
        }
    }

    for (int l = scLayers.size() - 1; l >= 0; l--) {
        if (l == 0) {
            for (int i = 0; i < inputSizes.size(); i++) {
                if (actions[i].empty())
                    continue;
                    
#ifdef KERNEL_NO_THREAD
                for (int x = 0; x < inputSizes[i].x; x++)
                    for (int y = 0; y < inputSizes[i].y; y++)
                        forward(Int2(x, y), cs.rng, &ns->states[l], l, i, nullptr);
#else
                runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns->states[l], l, i, nullptr), Int2(inputSizes[i].x, inputSizes[i].y), cs.rng, cs.batchSize2);
#endif
            }
        }
        else {
#ifdef KERNEL_NO_THREAD
            for (int x = 0; x < scLayers[l - 1].getHiddenSize().x; x++)
                for (int y = 0; y < scLayers[l - 1].getHiddenSize().y; y++)
                    forward(Int2(x, y), cs.rng, &ns->states[l], l, 0, &ns->states[l - 1]);
#else
            runKernel2(cs, std::bind(Hierarchy::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, &ns->states[l], l, 0, &ns->states[l - 1]), Int2(scLayers[l - 1].getHiddenSize().x, scLayers[l - 1].getHiddenSize().y), cs.rng, cs.batchSize2);
#endif
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = scLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(inputSizes.data()), numInputs * sizeof(Int3));

    os.write(reinterpret_cast<const char*>(updates.data()), updates.size() * sizeof(char));
    os.write(reinterpret_cast<const char*>(ticks.data()), ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(ticksPerUpdate.data()), ticksPerUpdate.size() * sizeof(int));

    for (int vli = 0; vli < actions.size(); vli++)
        writeBufferToStream(os, &actions[vli]);

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < historySizes[l].size(); i++)
            writeBufferToStream(os, histories[l][i].get());

        scLayers[l].writeToStream(os);

        writeBufferToStream(os, &rLayers[l].hiddenCounts);

        for (int v = 0; v < rLayers[l].weights.size(); v++) {
            char exists = !rLayers[l].weights[v].nonZeroValues.empty();

            os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

            if (exists) {
                writeSMToStream(os, rLayers[l].weights[v]);
                writeBufferToStream(os, &rLayers[l].visibleCounts[v]);
                        
                writeBufferToStream(os, &rLayers[l].activations[v]);
                writeBufferToStream(os, &rLayers[l].errors[v]);
            }
        }
    }

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&maxHistorySamples), sizeof(int));
    os.write(reinterpret_cast<const char*>(&historyIters), sizeof(int));

    int numHistorySamples = historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    // History samples
    for (int t = 0; t < numHistorySamples; t++) {
        const HistorySample &s = *historySamples[t];

        for (int l = 0; l < scLayers.size(); l++)
            writeBufferToStream(os, &s.states[l]);

        for (int vli = 0; vli < actions.size(); vli++)
            if (!actions[vli].empty())
                writeBufferToStream(os, &s.actionsPrev[vli]);
        
        os.write(reinterpret_cast<const char*>(&s.reward), sizeof(float));
    }
}

void Hierarchy::readFromStream(
    std::istream &is
) {
    int numLayers;
    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    int numInputs;
    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));

    inputSizes.resize(numInputs);

    is.read(reinterpret_cast<char*>(inputSizes.data()), numInputs * sizeof(Int3));

    scLayers.resize(numLayers);
    rLayers.resize(numLayers);

    ticks.resize(numLayers);

    histories.resize(numLayers);
    historySizes.resize(numLayers);
    
    ticksPerUpdate.resize(numLayers);

    updates.resize(numLayers);

    is.read(reinterpret_cast<char*>(updates.data()), updates.size() * sizeof(char));
    is.read(reinterpret_cast<char*>(ticks.data()), ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(ticksPerUpdate.data()), ticksPerUpdate.size() * sizeof(int));

    actions.resize(inputSizes.size());

    for (int vli = 0; vli < actions.size(); vli++)
        readBufferFromStream(is, &actions[vli]);

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;
        
        is.read(reinterpret_cast<char*>(&numHistorySizes), sizeof(int));
        historySizes[l].resize(numHistorySizes);
        is.read(reinterpret_cast<char*>(historySizes[l].data()), numHistorySizes * sizeof(int));

        histories[l].resize(numHistorySizes);

        for (int i = 0; i < historySizes[l].size(); i++) {
            histories[l][i] = std::make_shared<IntBuffer>();

            readBufferFromStream(is, histories[l][i].get());
        }

        scLayers[l].readFromStream(is);

        if (l == 0) {
            rLayers[l].weights.resize(inputSizes.size());
            rLayers[l].visibleCounts.resize(inputSizes.size());

            rLayers[l].activations.resize(inputSizes.size());
            rLayers[l].errors.resize(inputSizes.size());
        }
        else {
            rLayers[l].weights.resize(1);
            rLayers[l].visibleCounts.resize(1);

            rLayers[l].activations.resize(1);
            rLayers[l].errors.resize(1);
        }

        readBufferFromStream(is, &rLayers[l].hiddenCounts);

        for (int v = 0; v < rLayers[l].weights.size(); v++) {
            char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(char));

            if (exists) {
                readSMFromStream(is, rLayers[l].weights[v]);
                readBufferFromStream(is, &rLayers[l].visibleCounts[v]);

                readBufferFromStream(is, &rLayers[l].activations[v]);
                readBufferFromStream(is, &rLayers[l].errors[v]);
            }
        }
    }

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&maxHistorySamples), sizeof(int));
    is.read(reinterpret_cast<char*>(&historyIters), sizeof(int));

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    historySamples.resize(numHistorySamples);

    // History samples
    for (int t = 0; t < numHistorySamples; t++) {
        historySamples[t] = std::make_shared<HistorySample>();

        HistorySample &s = *historySamples[t];

        s.states.resize(scLayers.size());

        for (int l = 0; l < scLayers.size(); l++)
            readBufferFromStream(is, &s.states[l]);

        s.actionsPrev.resize(actions.size());

        for (int vli = 0; vli < actions.size(); vli++)
            if (!actions[vli].empty())
                readBufferFromStream(is, &s.actionsPrev[vli]);
        
        is.read(reinterpret_cast<char*>(&s.reward), sizeof(float));
    }
}
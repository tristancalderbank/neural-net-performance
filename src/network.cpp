#include "network.h"

#include <cmath>
#include <random>

float layer1Weights[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];
float layer1WeightsGradTotal[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];
float layer1Bias[HIDDEN_LAYER_SIZE];
float layer1BiasGradTotal[HIDDEN_LAYER_SIZE];
float layer1Activation[HIDDEN_LAYER_SIZE];

float layer2Weights[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float layer2WeightsGradTotal[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float layer2Bias[HIDDEN_LAYER_SIZE];
float layer2BiasGrad[HIDDEN_LAYER_SIZE];
float layer2BiasGradTotal[HIDDEN_LAYER_SIZE];
float layer2Activation[HIDDEN_LAYER_SIZE];

float outputLayerWeights[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerWeightsGradTotal[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerBias[OUTPUT_LAYER_SIZE];
float outputLayerBiasGrad[OUTPUT_LAYER_SIZE];
float outputLayerBiasGradTotal[OUTPUT_LAYER_SIZE];
float outputLayerActivation[OUTPUT_LAYER_SIZE];

namespace
{
    inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
}

void initializeNetwork()
{
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);

    auto randWeight = [&]() { return distribution(generator); };

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
            layer1Weights[i][j] = randWeight();
        layer1Bias[i] = 0.0f;

        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
            layer2Weights[i][j] = randWeight();
        layer2Bias[i] = 0.0f;
    }

    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
            outputLayerWeights[i][j] = randWeight();
        outputLayerBias[i] = 0.0f;
    }
}

void forwardPass(const Image& image)
{
    const float* inputLayerActivation = image.pixels;

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        layer1Activation[i] = 0.0f;
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
            layer1Activation[i] += layer1Weights[i][j] * inputLayerActivation[j];
        layer1Activation[i] = sigmoid(layer1Activation[i] + layer1Bias[i]);
    }

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        layer2Activation[i] = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
            layer2Activation[i] += layer2Weights[i][j] * layer1Activation[j];
        layer2Activation[i] = sigmoid(layer2Activation[i] + layer2Bias[i]);
    }

    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        outputLayerActivation[i] = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
            outputLayerActivation[i] += outputLayerWeights[i][j] * layer2Activation[j];
        outputLayerActivation[i] = sigmoid(outputLayerActivation[i] + outputLayerBias[i]);
    }
}

int getPredictedLabel()
{
    int index = 0;
    for (int i = 1; i < OUTPUT_LAYER_SIZE; i++)
    {
        if (outputLayerActivation[i] > outputLayerActivation[index])
            index = i;
    }
    return index;
}

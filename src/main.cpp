#include <iostream>
#include <random>
#include <vector>
#include <filesystem>
#include <string>
#include <cmath>

#include "image_loader.h"

constexpr unsigned int OUTPUT_LAYER_SIZE = 10;
constexpr unsigned int HIDDEN_LAYER_SIZE = 16;

float inputLayerActivation[INPUT_LAYER_SIZE];

float layer1Weights[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE]; // each node has 1 link to each pixel
float layer1WeightsGrad[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];
float layer1Bias[HIDDEN_LAYER_SIZE];
float layer1BiasGrad[HIDDEN_LAYER_SIZE];
float layer1Activation[HIDDEN_LAYER_SIZE];

float layer2Weights[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE]; // each node has 1 link to each node in layer1
float layer2WeightsGrad[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float layer2Bias[HIDDEN_LAYER_SIZE];
float layer2BiasGrad[HIDDEN_LAYER_SIZE];
float layer2Activation[HIDDEN_LAYER_SIZE];

float outputLayerWeights[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerWeightsGrad[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerBias[OUTPUT_LAYER_SIZE];
float outputLayerBiasGrad[OUTPUT_LAYER_SIZE];
float outputLayerActivation[OUTPUT_LAYER_SIZE];

void initializeNetwork()
{
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);

    auto randWeight = [&]() {
      return distribution(generator);
    };

    // layer 1/2
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        // layer 1
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
        {
            layer1Weights[i][j] = randWeight();
        }

        layer1Bias[i] = 0.0f;

        // layer 2
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            layer2Weights[i][j] = randWeight();
        }

        layer2Bias[i] = 0.0f;
    }

    // output layer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            outputLayerWeights[i][j] = randWeight();
        }

        outputLayerBias[i] = 0.0f;
    }  
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

void forwardPass(const Image& image)
{
    // set the input 
    for (int i = 0; i < INPUT_LAYER_SIZE; i++)
    {
        inputLayerActivation[i] = image.pixels[i];
    }

    // run layer 1
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        layer1Activation[i] = 0.0f;
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
        {
            layer1Activation[i] += layer1Weights[i][j] * inputLayerActivation[j];
        }

        layer1Activation[i] += layer1Bias[i];
        layer1Activation[i] = sigmoid(layer1Activation[i]);
    }

    // run layer 2
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        layer2Activation[i] = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            layer2Activation[i] += layer2Weights[i][j] * layer1Activation[j];
        }

        layer2Activation[i] += layer2Bias[i];
        layer2Activation[i] = sigmoid(layer2Activation[i]);
    }

    // output layer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        outputLayerActivation[i] = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            outputLayerActivation[i] += outputLayerWeights[i][j] * layer2Activation[j];
        }

        outputLayerActivation[i] += outputLayerBias[i];
        outputLayerActivation[i] = sigmoid(outputLayerActivation[i]);
    }
}

float calculateLoss(const Image& image)
{
    float loss = 0.0f;
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        float expected = i == image.label ? 1.0f : 0.0f;

        float diff = outputLayerActivation[i] - expected;

        loss += diff * diff;
    }

    return loss * 0.5; // used to avoid the 2.0f in the gradient 
}

void fillExpectedOutputVector(float* output, int label)
{
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        if (label == i)
            output[i] = 1.0f;
        else
            output[i] = 0.0f;
    }
}

// y is desired activation
// j = index in the current layer L
// k = index in the previous layer L - 1
// z_L is the activation of neuron j but without the sigmoid
// 
// Cost function: C_0 =  0.5 ((a1 - y1)^2 + (a2 - y2)^2 ... (an - yn)^2)
// * 1, 2, 3 is index of output layer neurons
// * _0 is training example 0
// * We add the 0.5 to cancel out the 2.0 from derived gradient
// 
// Derived gradients:
// 
// Weight gradient: 2.0f * (activation_L - y) * sigmoidDerivative(z_L) * activation_L_minus_1;
// Bias gradient: 2.0f * (activation_L - y) * sigmoidDerivative(z_L)
// 
void calculateGradientBackPropagation(float* expectedResult)
{
    // output layer
    for (int currIndex = 0; currIndex < OUTPUT_LAYER_SIZE; currIndex++)
    {
        float y = expectedResult[currIndex];
        float activation = outputLayerActivation[currIndex];
        float activationGrad = (activation - y);

        // sigmoid derivative: sigmoid(z_L) * (1.0f - sigmoid(z_L))
        // 
        // simplification: activation_L is already sigmoid(z_L) so no need to re-calculate it 

        float sigmoidDerivative = activation * (1.0f - activation);
        float biasGrad = activationGrad * sigmoidDerivative;

        outputLayerBiasGrad[currIndex] = biasGrad;

        for (int prevIndex = 0; prevIndex < HIDDEN_LAYER_SIZE; prevIndex++)
        {
            float activationPrev = layer2Activation[prevIndex];
            float weightGrad = biasGrad * activationPrev;
            
            outputLayerWeightsGrad[currIndex][prevIndex] = weightGrad;
        }
    }

    // hidden layer 2
    for (int currIndex = 0; currIndex < HIDDEN_LAYER_SIZE; currIndex++)
    {
        // first calculate the derivative of cost function wrt our own activation dC / da_k

        float activationGrad = 0.0f;

        for (int nextIndex = 0; nextIndex < OUTPUT_LAYER_SIZE; nextIndex++)
        {
            // float y = expectedResult[j];
            // float nextLayerActivation = outputLayerActivation[j];
            // float sigmoidDerivative = nextLayerActivation * (1.0f - nextLayerActivation);
            // result += (nextLayerActivation - y) * sigmoidDerivative * outputLayerWeights[j][k];

            // The above simplifies a lot because output layer biasGrad is already most of the calculcation
            // MAJOR KEY: this is where we propagate the gradient
            activationGrad += outputLayerBiasGrad[nextIndex] * outputLayerWeights[nextIndex][currIndex];
        }

        float activation = layer2Activation[currIndex];

        float sigmoidDerivative = activation * (1.0f - activation);

        float biasGrad = activationGrad * sigmoidDerivative;
        
        layer2BiasGrad[currIndex] = biasGrad;

        for (int prevIndex = 0; prevIndex < HIDDEN_LAYER_SIZE; prevIndex++)
        {
            float activationPrev = layer1Activation[prevIndex];
            float weightGrad = biasGrad * activationPrev;

            layer2WeightsGrad[currIndex][prevIndex] = weightGrad;
        }
    }

    // hidden layer 1
    for (int currIndex = 0; currIndex < HIDDEN_LAYER_SIZE; currIndex++)
    {
        // first calculate the derivative of cost function wrt our own activation dC / da_k

        float activationGrad = 0.0f;

        for (int nextIndex = 0; nextIndex < HIDDEN_LAYER_SIZE; nextIndex++)
        {
            activationGrad += layer2BiasGrad[nextIndex] * layer2Weights[nextIndex][currIndex];
        }

        float activation = layer1Activation[currIndex];

        float sigmoidDerivative = activation * (1.0f - activation);

        float biasGrad = activationGrad * sigmoidDerivative;

        layer1BiasGrad[currIndex] = biasGrad;

        for (int prevIndex = 0; prevIndex < INPUT_LAYER_SIZE; prevIndex++)
        {
            float activationPrev = inputLayerActivation[prevIndex];
            float weightGrad = biasGrad * activationPrev;

            layer1WeightsGrad[currIndex][prevIndex] = weightGrad;
        }
    }
}

int main()
{
    std::vector<Image> images;
    const std::string currentWorkingDirectory = std::filesystem::current_path().string();
    std::cout << "Working directory: " << currentWorkingDirectory << "\n";

    if (!loadDataset(
            "mnist_dataset/train-images.idx3-ubyte",
            "mnist_dataset/train-labels.idx1-ubyte",
            images))
    {
        std::cerr << "Could not load MNIST images\n";
        return 1;
    }

    initializeNetwork();

    Image& image = images[12345];

    forwardPass(image);

    float expectedResult[OUTPUT_LAYER_SIZE];

    fillExpectedOutputVector(expectedResult, image.label);

    calculateGradientBackPropagation(expectedResult);

    std::cout << "Loaded " << images.size() << " images\n";
    return 0;
}

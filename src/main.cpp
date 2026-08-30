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
float layer1Bias[HIDDEN_LAYER_SIZE];
float layer1Activation[HIDDEN_LAYER_SIZE];

float layer2Weights[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE]; // each node has 1 link to each node in layer1
float layer2Bias[HIDDEN_LAYER_SIZE];
float layer2Activation[HIDDEN_LAYER_SIZE];

float outputLayerWeights[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerBias[OUTPUT_LAYER_SIZE];
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

    return loss;
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

    float loss = calculateLoss(image);

    std::cout << "Loss: " << loss;

    std::cout << "Loaded " << images.size() << " images\n";
    return 0;
}

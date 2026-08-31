#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <filesystem>
#include <string>
#include <cmath>

#include "image_loader.h"

constexpr unsigned int OUTPUT_LAYER_SIZE = 10;
constexpr unsigned int HIDDEN_LAYER_SIZE = 64;
constexpr unsigned int NUM_TRAINING_IMAGES = 60000;
constexpr unsigned int NUM_TEST_IMAGES = 10000;
constexpr float LEARNING_RATE = 3.0f;
constexpr unsigned int NUM_EPOCHS = 30;
constexpr unsigned int BATCH_SIZE = 10;

float inputLayerActivation[INPUT_LAYER_SIZE];

float layer1Weights[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE]; // each node has 1 link to each pixel
float layer1WeightsGrad[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];
float layer1WeightsGradTotal[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];
float layer1Bias[HIDDEN_LAYER_SIZE];
float layer1BiasGrad[HIDDEN_LAYER_SIZE];
float layer1BiasGradTotal[HIDDEN_LAYER_SIZE];
float layer1Activation[HIDDEN_LAYER_SIZE];

float layer2Weights[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE]; // each node has 1 link to each node in layer1
float layer2WeightsGrad[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float layer2WeightsGradTotal[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float layer2Bias[HIDDEN_LAYER_SIZE];
float layer2BiasGrad[HIDDEN_LAYER_SIZE];
float layer2BiasGradTotal[HIDDEN_LAYER_SIZE];
float layer2Activation[HIDDEN_LAYER_SIZE];

float outputLayerWeights[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerWeightsGrad[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerWeightsGradTotal[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float outputLayerBias[OUTPUT_LAYER_SIZE];
float outputLayerBiasGrad[OUTPUT_LAYER_SIZE];
float outputLayerBiasGradTotal[OUTPUT_LAYER_SIZE];
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

void getExpectedOutputVector(float* output, int label)
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

void clearGradientTotals()
{
    // layer 1/2
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        // layer 1
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
        {
            layer1WeightsGradTotal[i][j] = 0.0f;
        }

        layer1BiasGradTotal[i] = 0.0f;

        // layer 2
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            layer2WeightsGradTotal[i][j] = 0.0f;
        }

        layer2BiasGradTotal[i] = 0.0f;
    }

    // output layer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            outputLayerWeightsGradTotal[i][j] = 0.0f;
        }

        outputLayerBiasGradTotal[i] = 0.0f;
    }
}

void accumGradient(float scaleFactor)
{
    // layer 1/2
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        // layer 1
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
        {
            layer1WeightsGradTotal[i][j] += layer1WeightsGrad[i][j] * scaleFactor;
        }

        layer1BiasGradTotal[i] += layer1BiasGrad[i] * scaleFactor;

        // layer 2
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            layer2WeightsGradTotal[i][j] += layer2WeightsGrad[i][j] * scaleFactor;
        }

        layer2BiasGradTotal[i] += layer2BiasGrad[i] * scaleFactor;
    }

    // output layer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            outputLayerWeightsGradTotal[i][j] += outputLayerWeightsGrad[i][j] * scaleFactor;
        }

        outputLayerBiasGradTotal[i] += outputLayerBiasGrad[i] * scaleFactor;
    }
}

void applyGradientStep()
{
    // layer 1/2
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        // layer 1
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
        {
            layer1Weights[i][j] -= layer1WeightsGradTotal[i][j] * LEARNING_RATE;
        }

        layer1Bias[i] -= layer1BiasGradTotal[i] * LEARNING_RATE;

        // layer 2
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            layer2Weights[i][j] -= layer2WeightsGradTotal[i][j] * LEARNING_RATE;
        }

        layer2Bias[i] -= layer2BiasGradTotal[i] * LEARNING_RATE;
    }

    // output layer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            outputLayerWeights[i][j] -= outputLayerWeightsGradTotal[i][j] * LEARNING_RATE;
        }

        outputLayerBias[i] -= outputLayerBiasGradTotal[i] * LEARNING_RATE;
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

int main()
{
    std::vector<Image> trainingImages;
    std::vector<Image> testImages;
    const std::string currentWorkingDirectory = std::filesystem::current_path().string();
    std::cout << "Working directory: " << currentWorkingDirectory << "\n";

    if (!loadDataset(
            "mnist_dataset/train-images.idx3-ubyte",
            "mnist_dataset/train-labels.idx1-ubyte",
        trainingImages))
    {
        std::cerr << "Could not load MNIST traning images\n";
        return 1;
    }

    if (!loadDataset(
        "mnist_dataset/t10k-images.idx3-ubyte",
        "mnist_dataset/t10k-labels.idx1-ubyte",
        testImages))
    {
        std::cerr << "Could not load MNIST test images\n";
        return 1;
    }

    initializeNetwork();

    // training
    std::random_device rd;
    std::mt19937 g(rd());
    const auto trainingStart = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        std::shuffle(trainingImages.begin(), trainingImages.end(), g);
        int epochCorrect = 0;
        float epochLoss = 0.0f;

        for (int i = 0; i < NUM_TRAINING_IMAGES; i += BATCH_SIZE)
        {
            const int currBatchSize = std::min(BATCH_SIZE, NUM_TRAINING_IMAGES - i);

            clearGradientTotals();

            for (int j = i; j < (i + BATCH_SIZE) && j < NUM_TRAINING_IMAGES; j++)
            {
                Image& image = trainingImages[j];

                float expectedResult[OUTPUT_LAYER_SIZE];
                getExpectedOutputVector(expectedResult, image.label);

                forwardPass(image);

                int predictedLabel = getPredictedLabel();
                if (predictedLabel == image.label)
                    epochCorrect++;

                epochLoss += calculateLoss(image);

                calculateGradientBackPropagation(expectedResult);

                accumGradient(1.0f / currBatchSize);
            }
            // take a gradient step
            applyGradientStep();
        }

        const float epochAccuracy = 100.0f * epochCorrect / NUM_TRAINING_IMAGES;
        const float averageEpochLoss = epochLoss / NUM_TRAINING_IMAGES;
        std::cout << "Epoch " << (epoch + 1)
            << ": loss = " << averageEpochLoss
            << ", accuracy = " << epochAccuracy << "%\n";
    }

    const auto trainingEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double> trainingDuration = trainingEnd - trainingStart;
    std::cout << "Training time: " << trainingDuration.count() << " seconds\n";

    // test evaluation
    int testCorrect = 0;

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        Image& image = testImages[i];

        forwardPass(image);

        int predictedLabel = getPredictedLabel();
        if (predictedLabel == image.label)
            testCorrect++;
    }

    const float testAccuracy = 100.0f * testCorrect / NUM_TEST_IMAGES;
    std::cout << "\nTest Accuracy: " << testAccuracy << "\n";

    return 0;
}

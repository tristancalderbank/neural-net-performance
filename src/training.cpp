#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <filesystem>
#include <string>
#include <cmath>

#include "network.h"
#include "model_io.h"

constexpr unsigned int NUM_TRAINING_IMAGES = 60000;
constexpr unsigned int NUM_TEST_IMAGES = 10000;
constexpr float LEARNING_RATE = 1.5f;
constexpr unsigned int NUM_EPOCHS = 30;
constexpr unsigned int BATCH_SIZE = 10;

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
void calculateGradientBackPropagation(const Image& image)
{
    const float* inputLayerActivation = image.pixels;

    // output layer
    for (int currIndex = 0; currIndex < OUTPUT_LAYER_SIZE; currIndex++)
    {
        float y = currIndex == image.label ? 1.0f : 0.0f;
        float activation = outputLayerActivation[currIndex];
        float activationGrad = (activation - y);

        // sigmoid derivative: sigmoid(z_L) * (1.0f - sigmoid(z_L))
        // 
        // simplification: activation_L is already sigmoid(z_L) so no need to re-calculate it 

        float sigmoidDerivative = activation * (1.0f - activation);
        float biasGrad = activationGrad * sigmoidDerivative;

        outputLayerBiasGrad[currIndex] = biasGrad;
        outputLayerBiasGradTotal[currIndex] += biasGrad;

        for (int prevIndex = 0; prevIndex < HIDDEN_LAYER_SIZE; prevIndex++)
        {
            float activationPrev = layer2Activation[prevIndex];
            float weightGrad = biasGrad * activationPrev;
            
            outputLayerWeightsGradTotal[currIndex][prevIndex] += weightGrad;
        }
    }

    // hidden layer 2
    float layer2ActivationGrad[HIDDEN_LAYER_SIZE] = {};
    for (int nextIndex = 0; nextIndex < OUTPUT_LAYER_SIZE; nextIndex++)
    {
        for (int currIndex = 0; currIndex < HIDDEN_LAYER_SIZE; currIndex++)
        {
            // float y = expectedResult[j];
            // float nextLayerActivation = outputLayerActivation[j];
            // float sigmoidDerivative = nextLayerActivation * (1.0f - nextLayerActivation);
            // result += (nextLayerActivation - y) * sigmoidDerivative * outputLayerWeights[j][k];

            // The above simplifies a lot because output layer biasGrad is already most of the calculcation
            // MAJOR KEY: this is where we propagate the gradient
            layer2ActivationGrad[currIndex] += outputLayerBiasGrad[nextIndex] * outputLayerWeights[nextIndex][currIndex];
        }
    }

    for (int currIndex = 0; currIndex < HIDDEN_LAYER_SIZE; currIndex++)
    {
        // first calculate the derivative of cost function wrt our own activation dC / da_k

        float activationGrad = layer2ActivationGrad[currIndex];

        float activation = layer2Activation[currIndex];

        float sigmoidDerivative = activation * (1.0f - activation);

        float biasGrad = activationGrad * sigmoidDerivative;
        
        layer2BiasGrad[currIndex] = biasGrad;
        layer2BiasGradTotal[currIndex] += biasGrad;

        for (int prevIndex = 0; prevIndex < HIDDEN_LAYER_SIZE; prevIndex++)
        {
            float activationPrev = layer1Activation[prevIndex];
            float weightGrad = biasGrad * activationPrev;

            layer2WeightsGradTotal[currIndex][prevIndex] += weightGrad;
        }
    }

    // hidden layer 1
    float layer1ActivationGrad[HIDDEN_LAYER_SIZE] = {};
    for (int nextIndex = 0; nextIndex < HIDDEN_LAYER_SIZE; nextIndex++)
    {
        for (int currIndex = 0; currIndex < HIDDEN_LAYER_SIZE; currIndex++)
        {
            layer1ActivationGrad[currIndex] += layer2BiasGrad[nextIndex] * layer2Weights[nextIndex][currIndex];
        }
    }


    for (int currIndex = 0; currIndex < HIDDEN_LAYER_SIZE; currIndex++)
    {
        // first calculate the derivative of cost function wrt our own activation dC / da_k

        float activationGrad = layer1ActivationGrad[currIndex];

        float activation = layer1Activation[currIndex];

        float sigmoidDerivative = activation * (1.0f - activation);

        float biasGrad = activationGrad * sigmoidDerivative;

        layer1BiasGradTotal[currIndex] += biasGrad;

        for (int prevIndex = 0; prevIndex < INPUT_LAYER_SIZE; prevIndex++)
        {
            float activationPrev = inputLayerActivation[prevIndex];
            float weightGrad = biasGrad * activationPrev;

            layer1WeightsGradTotal[currIndex][prevIndex] += weightGrad;
        }
    }
}

void applyGradientStepAndClearTotals(float batchScale)
{
    const float updateScale = batchScale * LEARNING_RATE;

    // layer 1/2
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        // layer 1
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
        {
            layer1Weights[i][j] -= layer1WeightsGradTotal[i][j] * updateScale;
            layer1WeightsGradTotal[i][j] = 0.0f;
        }

        layer1Bias[i] -= layer1BiasGradTotal[i] * updateScale;
        layer1BiasGradTotal[i] = 0.0f;

        // layer 2
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            layer2Weights[i][j] -= layer2WeightsGradTotal[i][j] * updateScale;
            layer2WeightsGradTotal[i][j] = 0.0f;
        }

        layer2Bias[i] -= layer2BiasGradTotal[i] * updateScale;
        layer2BiasGradTotal[i] = 0.0f;
    }

    // output layer
    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            outputLayerWeights[i][j] -= outputLayerWeightsGradTotal[i][j] * updateScale;
            outputLayerWeightsGradTotal[i][j] = 0.0f;
        }

        outputLayerBias[i] -= outputLayerBiasGradTotal[i] * updateScale;
        outputLayerBiasGradTotal[i] = 0.0f;
    }
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
        const auto epochStart = std::chrono::steady_clock::now();
        std::shuffle(trainingImages.begin(), trainingImages.end(), g);
        int epochCorrect = 0;
        float epochLoss = 0.0f;

        for (int i = 0; i < NUM_TRAINING_IMAGES; i += BATCH_SIZE)
        {
            const int currBatchSize = std::min(BATCH_SIZE, NUM_TRAINING_IMAGES - i);

            for (int j = i; j < (i + BATCH_SIZE) && j < NUM_TRAINING_IMAGES; j++)
            {
                Image& image = trainingImages[j];

                forwardPass(image);

                int predictedLabel = getPredictedLabel();
                if (predictedLabel == image.label)
                    epochCorrect++;

                epochLoss += calculateLoss(image);

                calculateGradientBackPropagation(image);
            }

            applyGradientStepAndClearTotals(1.0f / currBatchSize);
        }

        const auto epochEnd = std::chrono::steady_clock::now();
        const std::chrono::duration<double> epochDuration = epochEnd - epochStart;
        const float epochAccuracy = 100.0f * epochCorrect / NUM_TRAINING_IMAGES;
        const float averageEpochLoss = epochLoss / NUM_TRAINING_IMAGES;
        std::cout << "Epoch " << (epoch + 1)
            << ": loss = " << averageEpochLoss
            << ", accuracy = " << epochAccuracy << "%"
            << ", time = " << epochDuration.count() << " seconds\n";
    }

    const auto trainingEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double> trainingDuration = trainingEnd - trainingStart;
    std::cout << "Training time: " << trainingDuration.count() << " seconds\n";

    // test evaluation
    int testCorrect = 0;
    const auto inferenceStart = std::chrono::steady_clock::now();

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        Image& image = testImages[i];

        forwardPass(image);

        int predictedLabel = getPredictedLabel();
        if (predictedLabel == image.label)
            testCorrect++;
    }

    const auto inferenceEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double> inferenceDuration = inferenceEnd - inferenceStart;
    const float testAccuracy = 100.0f * testCorrect / NUM_TEST_IMAGES;
    std::cout << "\nTest Accuracy: " << testAccuracy
        << "% (" << NUM_TEST_IMAGES << " images, "
        << inferenceDuration.count() << " seconds)\n";

    if (saveModel("model.bin"))
        std::cout << "Model saved to model.bin\n";
    else
        std::cerr << "Could not save model to model.bin\n";

    return 0;
}

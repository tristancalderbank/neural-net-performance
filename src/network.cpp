#include "network.h"

#include <cmath>
#include <random>

#if defined(NNP_PLATFORM_MACOS)
#include <arm_neon.h>
#elif defined(NNP_PLATFORM_WINDOWS)
#include <immintrin.h>
#endif

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

inline float dotProductSIMD(const float* a, const float* b, int size)
{
#if defined(NNP_PLATFORM_WINDOWS)
    constexpr int batch = sizeof(__m256) / sizeof(float);

    __m256 acc = _mm256_setzero_ps(); // zero accumulator [0, 0, 0, 0, 0, 0, 0, 0]

    int i = 0;

    for (; i + batch <= size; i += batch)
    {
        __m256 vecA = _mm256_loadu_ps(a + i); // load in values from a
        __m256 vecB = _mm256_loadu_ps(b + i); // load in values from b

        acc = _mm256_fmadd_ps(vecA, vecB, acc); // do FMA
    }

    // now we have everything summed but its still 8 floats in the acc
    __m128 low = _mm256_castps256_ps128(acc); // extract lower lanes
    __m128 high = _mm256_extractf128_ps(acc, 1); // extract upper lanes

    __m128 sum = _mm_add_ps(low, high); // add low and high

    sum = _mm_hadd_ps(sum, sum); // horizontal add, does [x0 + x1, x2 + x3, x0 + x1, x2 + x3]
    sum = _mm_hadd_ps(sum, sum); // now result is in lane 0

    float total = _mm_cvtss_f32(sum); // extract lowest lane
#elif defined(NNP_PLATFORM_MACOS)
    constexpr int batch = sizeof(float32x4_t) / sizeof(float);
    
    float32x4_t acc = vdupq_n_f32(0.0f); // zero accumulator [0, 0, 0, 0]
    
    int i = 0;
    
    for (; i + batch <= size; i += batch)
    {
        float32x4_t vecA = vld1q_f32(a + i); // load in values from a
        float32x4_t vecB = vld1q_f32(b + i); // load in values from b
        
        acc = vfmaq_f32(acc, vecA, vecB); // do FMA
    }
    
    // now we have everything summed but its still 4 floats in the acc
    float total = vaddvq_f32(acc); // sum 4 vals
#endif
    
    
    // handle tail less than 8
    for (; i < size; i++)
    {
        total += a[i] * b[i];
    }

    return total;
}

void forwardPass(const Image& image)
{
    const float* inputLayerActivation = image.pixels;

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        layer1Activation[i] = dotProductSIMD(layer1Weights[i], inputLayerActivation, INPUT_LAYER_SIZE);
        layer1Activation[i] = sigmoid(layer1Activation[i] + layer1Bias[i]);
    }

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        layer2Activation[i] = dotProductSIMD(layer2Weights[i], layer1Activation, HIDDEN_LAYER_SIZE);
        layer2Activation[i] = sigmoid(layer2Activation[i] + layer2Bias[i]);
    }

    for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
    {
        outputLayerActivation[i] = dotProductSIMD(outputLayerWeights[i], layer2Activation, HIDDEN_LAYER_SIZE);
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

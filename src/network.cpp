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
    constexpr int batch = (sizeof(__m256) / sizeof(float)) * 8;

    __m256 acc0 = _mm256_setzero_ps(); // zero accumulator [0, 0, 0, 0, 0, 0, 0, 0]
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    int i = 0;

    for (; i + batch <= size; i += batch)
    {
        __m256 vec0A = _mm256_loadu_ps(a + i); // load in values from a
        __m256 vec1A = _mm256_loadu_ps(a + i + 8);
        __m256 vec2A = _mm256_loadu_ps(a + i + 16);
        __m256 vec3A = _mm256_loadu_ps(a + i + 24);
        __m256 vec4A = _mm256_loadu_ps(a + i + 32);
        __m256 vec5A = _mm256_loadu_ps(a + i + 40);
        __m256 vec6A = _mm256_loadu_ps(a + i + 48);
        __m256 vec7A = _mm256_loadu_ps(a + i + 56);


        __m256 vec0B = _mm256_loadu_ps(b + i); // load in values from b
        __m256 vec1B = _mm256_loadu_ps(b + i + 8);
        __m256 vec2B = _mm256_loadu_ps(b + i + 16);
        __m256 vec3B = _mm256_loadu_ps(b + i + 24);
        __m256 vec4B = _mm256_loadu_ps(b + i + 32);
        __m256 vec5B = _mm256_loadu_ps(b + i + 40);
        __m256 vec6B = _mm256_loadu_ps(b + i + 48);
        __m256 vec7B = _mm256_loadu_ps(b + i + 56);


        acc0 = _mm256_fmadd_ps(vec0A, vec0B, acc0); // do FMA
        acc1 = _mm256_fmadd_ps(vec1A, vec1B, acc1);
        acc2 = _mm256_fmadd_ps(vec2A, vec2B, acc2);
        acc3 = _mm256_fmadd_ps(vec3A, vec3B, acc3);
        acc4 = _mm256_fmadd_ps(vec4A, vec4B, acc4);
        acc5 = _mm256_fmadd_ps(vec5A, vec5B, acc5);
        acc6 = _mm256_fmadd_ps(vec6A, vec6B, acc6);
        acc7 = _mm256_fmadd_ps(vec7A, vec7B, acc7);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc4 = _mm256_add_ps(acc4, acc5);
    acc6 = _mm256_add_ps(acc6, acc7);

    acc0 = _mm256_add_ps(acc0, acc2);
    acc4 = _mm256_add_ps(acc4, acc6);

    acc0 = _mm256_add_ps(acc0, acc4);

    // now we have everything summed but its still 8 floats in the acc
    __m128 low = _mm256_castps256_ps128(acc0); // extract lower lanes
    __m128 high = _mm256_extractf128_ps(acc0, 1); // extract upper lanes

    __m128 sum = _mm_add_ps(low, high); // add low and high

    sum = _mm_hadd_ps(sum, sum); // horizontal add, does [x0 + x1, x2 + x3, x0 + x1, x2 + x3]
    sum = _mm_hadd_ps(sum, sum); // now result is in lane 0

    float total = _mm_cvtss_f32(sum); // extract lowest lane
#elif defined(NNP_PLATFORM_MACOS)
    constexpr int batch = sizeof(float32x4_t) / sizeof(float) * 4;
    
    float32x4_t acc0 = vdupq_n_f32(0.0f); // zero accumulator [0, 0, 0, 0]
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    
    int i = 0;
    
    for (; i + batch <= size; i += batch)
    {
        float32x4_t vecA0 = vld1q_f32(a + i); // load in values from a
        float32x4_t vecA1 = vld1q_f32(a + i + 4);
        float32x4_t vecA2 = vld1q_f32(a + i + 8);
        float32x4_t vecA3 = vld1q_f32(a + i + 12);
        
        float32x4_t vecB0 = vld1q_f32(b + i); // load in values from b
        float32x4_t vecB1 = vld1q_f32(b + i + 4);
        float32x4_t vecB2 = vld1q_f32(b + i + 8);
        float32x4_t vecB3 = vld1q_f32(b + i + 12);
        
        acc0 = vfmaq_f32(acc0, vecA0, vecB0); // do FMA
        acc1 = vfmaq_f32(acc1, vecA1, vecB1);
        acc2 = vfmaq_f32(acc2, vecA2, vecB2);
        acc3 = vfmaq_f32(acc3, vecA3, vecB3);
    }
    
    // now we have everything summed but its still 4 floats in the acc
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    
    acc0 = vaddq_f32(acc0, acc2);
    
    float total = vaddvq_f32(acc0);
#endif
    
    
    // handle tail less than batch size
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

#pragma once

#include "image_loader.h"

constexpr unsigned int OUTPUT_LAYER_SIZE = 10;
constexpr unsigned int HIDDEN_LAYER_SIZE = 256;

extern float layer1Weights[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];
extern float layer1WeightsGradTotal[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];
extern float layer1Bias[HIDDEN_LAYER_SIZE];
extern float layer1BiasGradTotal[HIDDEN_LAYER_SIZE];
extern float layer1Activation[HIDDEN_LAYER_SIZE];

extern float layer2Weights[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
extern float layer2WeightsGradTotal[HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
extern float layer2Bias[HIDDEN_LAYER_SIZE];
extern float layer2BiasGrad[HIDDEN_LAYER_SIZE];
extern float layer2BiasGradTotal[HIDDEN_LAYER_SIZE];
extern float layer2Activation[HIDDEN_LAYER_SIZE];

extern float outputLayerWeights[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
extern float outputLayerWeightsGradTotal[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
extern float outputLayerBias[OUTPUT_LAYER_SIZE];
extern float outputLayerBiasGrad[OUTPUT_LAYER_SIZE];
extern float outputLayerBiasGradTotal[OUTPUT_LAYER_SIZE];
extern float outputLayerActivation[OUTPUT_LAYER_SIZE];

void initializeNetwork();
void forwardPass(const Image& image);
int getPredictedLabel();

#pragma once

#include <cstdint>
#include <vector>

constexpr unsigned int IMAGE_DIM = 28;
constexpr unsigned int INPUT_LAYER_SIZE = IMAGE_DIM * IMAGE_DIM;

struct Image
{
    float pixels[INPUT_LAYER_SIZE];
    std::uint8_t label;
};

bool loadDataset(const char* imagesPath, const char* labelsPath, std::vector<Image>& images);

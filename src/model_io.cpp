#include "model_io.h"

#include "network.h"

#include <cstdint>
#include <fstream>

namespace
{
    constexpr std::uint32_t MODEL_MAGIC = 0x4E4E4D31; // "NNM1"
    constexpr std::uint32_t MODEL_VERSION = 1;

    struct ModelHeader
    {
        std::uint32_t magic;
        std::uint32_t version;
        std::uint32_t inputSize;
        std::uint32_t hiddenSize;
        std::uint32_t outputSize;
    };

    template <typename T>
    bool writeValue(std::ofstream& file, const T& value)
    {
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        return static_cast<bool>(file);
    }

    template <typename T>
    bool readValue(std::ifstream& file, T& value)
    {
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(file);
    }

    template <typename T, std::size_t N>
    bool writeArray(std::ofstream& file, const T (&values)[N])
    {
        file.write(reinterpret_cast<const char*>(values), sizeof(values));
        return static_cast<bool>(file);
    }

    template <typename T, std::size_t N>
    bool readArray(std::ifstream& file, T (&values)[N])
    {
        file.read(reinterpret_cast<char*>(values), sizeof(values));
        return static_cast<bool>(file);
    }
}

bool saveModel(const char* path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        return false;

    const ModelHeader header{
        MODEL_MAGIC,
        MODEL_VERSION,
        INPUT_LAYER_SIZE,
        HIDDEN_LAYER_SIZE,
        OUTPUT_LAYER_SIZE};

    if (!writeValue(file, header))
        return false;

    return writeArray(file, layer1Weights) &&
           writeArray(file, layer1Bias) &&
           writeArray(file, layer2Weights) &&
           writeArray(file, layer2Bias) &&
           writeArray(file, outputLayerWeights) &&
           writeArray(file, outputLayerBias);
}

bool loadModel(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return false;

    ModelHeader header{};
    if (!readValue(file, header) ||
        header.magic != MODEL_MAGIC ||
        header.version != MODEL_VERSION ||
        header.inputSize != INPUT_LAYER_SIZE ||
        header.hiddenSize != HIDDEN_LAYER_SIZE ||
        header.outputSize != OUTPUT_LAYER_SIZE)
    {
        return false;
    }

    return readArray(file, layer1Weights) &&
           readArray(file, layer1Bias) &&
           readArray(file, layer2Weights) &&
           readArray(file, layer2Bias) &&
           readArray(file, outputLayerWeights) &&
           readArray(file, outputLayerBias);
}

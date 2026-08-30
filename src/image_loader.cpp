#include "image_loader.h"

#include <cstdint>
#include <fstream>

namespace
{
    struct ImageFileHeader
    {
        std::uint32_t magicNumber;
        std::uint32_t imageCount;
        std::uint32_t rowCount;
        std::uint32_t columnCount;
    };

    struct LabelFileHeader
    {
        std::uint32_t magicNumber;
        std::uint32_t labelCount;
    };

    std::uint32_t readBigEndianUint32(std::ifstream& file)
    {
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), sizeof(bytes));

        return (static_cast<std::uint32_t>(bytes[0]) << 24) |
               (static_cast<std::uint32_t>(bytes[1]) << 16) |
               (static_cast<std::uint32_t>(bytes[2]) << 8) |
               static_cast<std::uint32_t>(bytes[3]);
    }
}

bool loadDataset(const char* imagesPath, const char* labelsPath, std::vector<Image>& images)
{
    std::ifstream imagesFile(imagesPath, std::ios::binary);
    std::ifstream labelsFile(labelsPath, std::ios::binary);

    if (!imagesFile || !labelsFile)
    {
        return false;
    }

    ImageFileHeader header;
    header.magicNumber = readBigEndianUint32(imagesFile);
    header.imageCount = readBigEndianUint32(imagesFile);
    header.rowCount = readBigEndianUint32(imagesFile);
    header.columnCount = readBigEndianUint32(imagesFile);

    LabelFileHeader labelHeader;
    labelHeader.magicNumber = readBigEndianUint32(labelsFile);
    labelHeader.labelCount = readBigEndianUint32(labelsFile);

    if (!imagesFile || header.magicNumber != 2051 ||
        header.rowCount != IMAGE_DIM || header.columnCount != IMAGE_DIM ||
        !labelsFile || labelHeader.magicNumber != 2049 ||
        labelHeader.labelCount != header.imageCount)
    {
        return false;
    }

    images.reserve(images.size() + header.imageCount);

    for (std::uint32_t imageIndex = 0; imageIndex < header.imageCount; imageIndex++)
    {
        Image image;
        std::uint8_t rawPixels[INPUT_LAYER_SIZE];

        imagesFile.read(
            reinterpret_cast<char*>(rawPixels),
            sizeof(rawPixels));

        if (!imagesFile)
        {
            return false;
        }

        labelsFile.read(reinterpret_cast<char*>(&image.label), sizeof(image.label));

        if (!labelsFile)
        {
            return false;
        }

        for (unsigned int pixelIndex = 0; pixelIndex < INPUT_LAYER_SIZE; pixelIndex++)
        {
            image.pixels[pixelIndex] = rawPixels[pixelIndex] / 255.0f;
        }

        images.push_back(image);
    }

    return true;
}

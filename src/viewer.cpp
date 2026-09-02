#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "model_io.h"
#include "network.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <string>
#include <vector>

namespace
{
    constexpr int DRAWING_DIMENSION = 28;
    constexpr int DRAWING_PIXEL_COUNT = DRAWING_DIMENSION * DRAWING_DIMENSION;
    constexpr int BRUSH_RADIUS = 1;
    constexpr float BRUSH_WEIGHTS[3][3] = {
        {0.25f, 0.50f, 0.25f},
        {0.50f, 1.00f, 0.50f},
        {0.25f, 0.50f, 0.25f}};
}

int main()
{
    if (!glfwInit())
        return 1;

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Neural Network Viewer", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    float drawingPixels[DRAWING_PIXEL_COUNT] = {};
    Image drawingImage{};
    const bool modelLoaded = loadModel("model.bin");
    std::vector<Image> testImages;
    const bool testImagesLoaded = loadDataset(
        "mnist_dataset/t10k-images.idx3-ubyte",
        "mnist_dataset/t10k-labels.idx1-ubyte",
        testImages);
    int selectedTestImage = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiWindowFlags fixedWindowFlags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->WorkPos);
        ImGui::SetNextWindowSize(ImGui::GetMainViewport()->WorkSize);
        ImGui::Begin("Neural Network", nullptr, fixedWindowFlags);
        ImGui::Text("Draw a digit (28 x 28 pixels)");

        if (ImGui::Button("Clear"))
            std::fill(std::begin(drawingPixels), std::end(drawingPixels), 0.0f);

        constexpr float canvasPixelSize = 16.0f;
        const ImVec2 canvasSize(
            DRAWING_DIMENSION * canvasPixelSize,
            DRAWING_DIMENSION * canvasPixelSize);
        const ImVec2 canvasPosition = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton("drawing_canvas", canvasSize);

        const bool drawingCanvasHovered = ImGui::IsItemHovered();
        if (drawingCanvasHovered && ImGui::IsMouseDown(ImGuiMouseButton_Left))
        {
            const ImVec2 mousePosition = ImGui::GetIO().MousePos;
            const int pixelX = static_cast<int>((mousePosition.x - canvasPosition.x) / canvasPixelSize);
            const int pixelY = static_cast<int>((mousePosition.y - canvasPosition.y) / canvasPixelSize);

            for (int offsetY = -BRUSH_RADIUS; offsetY <= BRUSH_RADIUS; offsetY++)
            {
                for (int offsetX = -BRUSH_RADIUS; offsetX <= BRUSH_RADIUS; offsetX++)
                {
                    const int brushX = pixelX + offsetX;
                    const int brushY = pixelY + offsetY;
                    if (brushX >= 0 && brushX < DRAWING_DIMENSION &&
                        brushY >= 0 && brushY < DRAWING_DIMENSION)
                    {
                        const float brushValue = BRUSH_WEIGHTS[
                            offsetY + BRUSH_RADIUS][offsetX + BRUSH_RADIUS];
                        float& pixel = drawingPixels[brushY * DRAWING_DIMENSION + brushX];
                        pixel = std::max(pixel, brushValue);
                    }
                }
            }
        }

        if (modelLoaded)
        {
            std::copy(
                std::begin(drawingPixels),
                std::end(drawingPixels),
                std::begin(drawingImage.pixels));
            forwardPass(drawingImage);
        }

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        for (int y = 0; y < DRAWING_DIMENSION; y++)
        {
            for (int x = 0; x < DRAWING_DIMENSION; x++)
            {
                const float value = drawingPixels[y * DRAWING_DIMENSION + x];
                const ImVec2 topLeft(
                    canvasPosition.x + x * canvasPixelSize,
                    canvasPosition.y + y * canvasPixelSize);
                const ImVec2 bottomRight(
                    topLeft.x + canvasPixelSize,
                    topLeft.y + canvasPixelSize);

                const int intensity = static_cast<int>(value * 255.0f);
                drawList->AddRectFilled(
                    topLeft,
                    bottomRight,
                    IM_COL32(intensity, intensity, intensity, 255));
                drawList->AddRect(
                    topLeft,
                    bottomRight,
                    IM_COL32(70, 70, 70, 255));
            }
        }

        constexpr float outputBoxSize = 40.0f;
        constexpr float outputBoxSpacing = 4.0f;
        const float outputBoxesX = canvasPosition.x + canvasSize.x + 24.0f;
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
        {
            const float activation = std::clamp(outputLayerActivation[i], 0.0f, 1.0f);
            const int intensity = static_cast<int>(activation * 255.0f);
            const ImVec2 boxTopLeft(
                outputBoxesX,
                canvasPosition.y + i * (outputBoxSize + outputBoxSpacing));
            const ImVec2 boxBottomRight(
                boxTopLeft.x + outputBoxSize,
                boxTopLeft.y + outputBoxSize);

            drawList->AddRectFilled(
                boxTopLeft,
                boxBottomRight,
                IM_COL32(intensity, intensity, intensity, 255));
            drawList->AddRect(
                boxTopLeft,
                boxBottomRight,
                IM_COL32(180, 180, 180, 255));

            const ImVec2 labelPosition(boxTopLeft.x + 16.0f, boxTopLeft.y + 12.0f);
            const ImU32 labelColor = intensity > 128
                ? IM_COL32(0, 0, 0, 255)
                : IM_COL32(255, 255, 255, 255);
            drawList->AddText(labelPosition, labelColor, std::to_string(i).c_str());
        }

        if (testImagesLoaded && !testImages.empty())
        {
            ImGui::Dummy(ImVec2(0.0f, 24.0f));
            ImGui::SliderInt(
                "Test image",
                &selectedTestImage,
                0,
                static_cast<int>(testImages.size()) - 1);
            int testPrediction = -1;
            if (modelLoaded)
            {
                forwardPass(testImages[selectedTestImage]);
                testPrediction = getPredictedLabel();
            }
            ImGui::Text(
                "MNIST test image (label: %d, prediction: %d)",
                testImages[selectedTestImage].label,
                testPrediction);
            const bool testPredictionWrong =
                modelLoaded && testPrediction != testImages[selectedTestImage].label;

            const ImVec2 testCanvasPosition(
                outputBoxesX + outputBoxSize + 24.0f,
                canvasPosition.y);

            for (int y = 0; y < DRAWING_DIMENSION; y++)
            {
                for (int x = 0; x < DRAWING_DIMENSION; x++)
                {
                    const float value = testImages[selectedTestImage].pixels[y * DRAWING_DIMENSION + x];
                    const int intensity = static_cast<int>(value * 255.0f);
                    const ImVec2 topLeft(
                        testCanvasPosition.x + x * canvasPixelSize,
                        testCanvasPosition.y + y * canvasPixelSize);
                    const ImVec2 bottomRight(
                        topLeft.x + canvasPixelSize,
                        topLeft.y + canvasPixelSize);

                    drawList->AddRectFilled(
                        topLeft,
                        bottomRight,
                        IM_COL32(intensity, intensity, intensity, 255));
                    drawList->AddRect(
                        topLeft,
                        bottomRight,
                        IM_COL32(70, 70, 70, 255));
                }
            }

            const float testOutputBoxesX = testCanvasPosition.x + canvasSize.x + 24.0f;
            for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
            {
                const float activation = std::clamp(outputLayerActivation[i], 0.0f, 1.0f);
                const int intensity = static_cast<int>(activation * 255.0f);
                const ImVec2 boxTopLeft(
                    testOutputBoxesX,
                    testCanvasPosition.y + i * (outputBoxSize + outputBoxSpacing));
                const ImVec2 boxBottomRight(
                    boxTopLeft.x + outputBoxSize,
                    boxTopLeft.y + outputBoxSize);
                const bool isWrongPredictionBox = testPredictionWrong && i == testPrediction;

                drawList->AddRectFilled(
                    boxTopLeft,
                    boxBottomRight,
                    isWrongPredictionBox
                        ? IM_COL32(255, 0, 0, 255)
                        : IM_COL32(intensity, intensity, intensity, 255));
                drawList->AddRect(
                    boxTopLeft,
                    boxBottomRight,
                    isWrongPredictionBox
                        ? IM_COL32(255, 0, 0, 255)
                        : IM_COL32(180, 180, 180, 255));

                const ImVec2 labelPosition(boxTopLeft.x + 16.0f, boxTopLeft.y + 12.0f);
                const ImU32 labelColor = intensity > 128
                    ? IM_COL32(0, 0, 0, 255)
                    : IM_COL32(255, 255, 255, 255);
                drawList->AddText(labelPosition, labelColor, std::to_string(i).c_str());
            }
        }

        ImGui::End();

        ImGui::Render();
        int displayWidth = 0;
        int displayHeight = 0;
        glfwGetFramebufferSize(window, &displayWidth, &displayHeight);
        glViewport(0, 0, displayWidth, displayHeight);
        glClearColor(0.10f, 0.10f, 0.10f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

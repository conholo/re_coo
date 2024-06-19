#pragma once

#include "renderer/camera.h"
#include "renderer/vulkan/vulkan_descriptors.h"
#include <vulkan/vulkan.h>

struct GlobalUbo
{
    glm::mat4 Projection{1.0f};
    glm::mat4 View{1.0f};
    glm::mat4 InvView{1.0f};
    glm::mat4 InvProjection{1.0f};
    glm::vec4 CameraPosition{0.0f};
    glm::ivec4 ScreenResolution_NumRaysPerPixel_FrameNumber{-1};
};
#pragma once

#include <glm/glm.hpp>
#include "core/buffer.h"

#include <vector>

struct RayTracingMaterial
{
    glm::vec4 Color_Smoothness;
    glm::vec4 EmissionColor_Strength;
    glm::vec4 SpecularColor_Probability;
};

struct Sphere
{
    glm::vec4 Position_Radius;
    RayTracingMaterial Material;

    static Buffer SpheresToBuffer(std::vector<Sphere>& spheres);
};


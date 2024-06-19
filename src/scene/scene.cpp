#include "scene/scene.h"

Buffer Sphere::SpheresToBuffer(std::vector<Sphere>& spheres)
{
    Sphere* data = spheres.data();
    uint64_t sizeInBytes = spheres.size() * sizeof(Sphere);
    Buffer buffer(data, sizeInBytes);
    return buffer;
}

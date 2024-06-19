#version 450

layout (location = 0) in vec2 v_UV;
layout (location = 1) in vec3 v_Ro;
layout (location = 2) in vec3 v_Rd;

layout (location = 0) out vec4 o_Color;

layout(set = 0, binding = 0) uniform GlobalUBO
{
    mat4 Projection;
    mat4 View;
    mat4 InvView;
    mat4 InvProjection;
    vec4 CameraPosition;
    ivec4 ScreenResolution_NumRaysPerPixel_FrameNumber;
}  u_UBO;


struct RayTracingMaterial
{
    vec4 Color_Smoothness;
    vec4 EmissionColor_Strength;
    vec4 SpecularColor_Probability;
};

struct Sphere
{
    vec4 Position_Radius;
    RayTracingMaterial Material;
};

layout(std430, set = 1, binding = 0) buffer Spheres
{
    Sphere Spheres[];
} u_Spheres;

struct Ray
{
    vec3 Origin;
    vec3 Dir;
};

struct HitInfo
{
    bool DidHit;
    float Distance;
    vec3 HitPoint;
    vec3 Normal;
    RayTracingMaterial Material;
};
// --- RNG Stuff ---

const float PI = 3.1415;

// PCG (permuted congruential generator). Thanks to:
// www.pcg-random.org and www.shadertoy.com/view/XlGcRh
uint NextRandom(inout uint state)
{
    state = state * 747796405 + 2891336453;
    uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    result = (result >> 22) ^ result;
    return result;
}

float RandomValue(inout uint state)
{
    return NextRandom(state) / 4294967295.0; // 2^32 - 1
}

// Random value in normal distribution (with mean=0 and sd=1)
float RandomValueNormalDistribution(inout uint state)
{
    // Thanks to https://stackoverflow.com/a/6178290
    float theta = 2 * 3.1415926 * RandomValue(state);
    float rho = sqrt(-2 * log(RandomValue(state)));
    return rho * cos(theta);
}

// Calculate a random direction
vec3 RandomDirection(inout uint state)
{
    // Thanks to https://math.stackexchange.com/a/1585996
    float x = RandomValueNormalDistribution(state);
    float y = RandomValueNormalDistribution(state);
    float z = RandomValueNormalDistribution(state);
    return normalize(vec3(x, y, z));
}

vec2 RandomPointInCircle(inout uint rngState)
{
    float angle = RandomValue(rngState) * 2 * PI;
    vec2 pointOnCircle = vec2(cos(angle), sin(angle));
    return pointOnCircle * sqrt(RandomValue(rngState));
}

HitInfo RaySphere(Ray ray, vec3 sphereCentre, float sphereRadius)
{
    HitInfo hitInfo;
    vec3 offsetRayOrigin = ray.Origin - sphereCentre;
    // From the equation: sqrLength(rayOrigin + rayDir * dst) = radius^2
    // Solving for dst results in a quadratic equation with coefficients:
    float a = dot(ray.Dir, ray.Dir); // a = 1 (assuming unit vector)
    float b = 2 * dot(offsetRayOrigin, ray.Dir);
    float c = dot(offsetRayOrigin, offsetRayOrigin) - sphereRadius * sphereRadius;
    // Quadratic discriminant
    float discriminant = b * b - 4 * a * c;

    // No solution when d < 0 (ray misses sphere)
    if (discriminant >= 0)
    {
        // Distance to nearest intersection point (from quadratic formula)
        float dst = (-b - sqrt(discriminant)) / (2 * a);

        // Ignore intersections that occur behind the ray
        if (dst >= 0)
        {
            hitInfo.DidHit = true;
            hitInfo.Distance = dst;
            hitInfo.HitPoint = ray.Origin + ray.Dir * dst;
            hitInfo.Normal = normalize(hitInfo.HitPoint - sphereCentre);
        }
    }
    return hitInfo;
}

// Find the first point that the given ray collides with, and return hit info
HitInfo CalculateRayCollision(Ray ray)
{
    HitInfo closestHit;
    // We haven't hit anything yet, so 'closest' hit is infinitely far away
    closestHit.Distance = 1e6;

    // Raycast against all spheres and keep info about the closest hit
    for (int i = 0; i < u_Spheres.Spheres.length(); i ++)
    {
        Sphere sphere = u_Spheres.Spheres[i];
        vec4 position_radius = sphere.Position_Radius;
        HitInfo hitInfo = RaySphere(ray, position_radius.xyz, position_radius.w);

        if (hitInfo.DidHit && hitInfo.Distance < closestHit.Distance)
        {
            closestHit = hitInfo;
            closestHit.Material = sphere.Material;
        }
    }
    return closestHit;
}


vec3 Trace(Ray ray, inout uint rngState)
{
    vec3 incomingLight = vec3(0.0);
    vec3 rayColor = vec3(1.0);

    HitInfo hitInfo = CalculateRayCollision(ray);

    if(!hitInfo.DidHit)
    {
        return vec3(0.0);
    }

    RayTracingMaterial mat = hitInfo.Material;
    int isSpecularBounce = mat.SpecularColor_Probability.w >= RandomValue(rngState) ? 1 : 0;

    ray.Origin = hitInfo.HitPoint;
    vec3 diffuseDir = normalize(hitInfo.Normal + RandomDirection(rngState));
    vec3 specularDir = reflect(ray.Dir, hitInfo.Normal);
    ray.Dir = normalize(mix(diffuseDir, specularDir, mat.Color_Smoothness.w * isSpecularBounce));

    // Update light calculations
    vec3 emittedLight = mat.EmissionColor_Strength.xyz * mat.EmissionColor_Strength.w;
    incomingLight += emittedLight * rayColor;
    rayColor *= mix(mat.Color_Smoothness.xyz, mat.SpecularColor_Probability.xyz, isSpecularBounce);

    return incomingLight;
}

void main()
{
    Ray ray;
    ray.Origin = v_Ro;
    ray.Dir = normalize(v_Rd);

    ivec2 numPixels = u_UBO.ScreenResolution_NumRaysPerPixel_FrameNumber.xy;
    ivec2 pixelCoord = ivec2(v_UV) * numPixels;
    uint pixelIndex = pixelCoord.y * numPixels.x + pixelCoord.x;

    int frameNumber = u_UBO.ScreenResolution_NumRaysPerPixel_FrameNumber.w;
    uint rngState = pixelIndex + frameNumber * 719393;

    vec3 cameraRight = vec3(u_UBO.View[0][0], u_UBO.View[1][0], u_UBO.View[2][0]);
    vec3 cameraUp = vec3(u_UBO.View[0][1], u_UBO.View[1][1], u_UBO.View[2][1]);
    vec3 cameraForward = -vec3(u_UBO.View[0][2], u_UBO.View[1][2], u_UBO.View[2][2]);

    vec3 incomingLight = vec3(0.0);
    int raysPerPixel = u_UBO.ScreenResolution_NumRaysPerPixel_FrameNumber.z;
    for(int i = 0; i < raysPerPixel; i++)
    {
        incomingLight += Trace(ray, rngState);
    }

    vec3 color = incomingLight / raysPerPixel;

    o_Color = vec4(color, 1.0);
}
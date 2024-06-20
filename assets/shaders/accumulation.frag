#version 450

layout (location = 0) in vec2 v_UV;
layout (location = 0) out vec4 o_Next;

layout(set = 0, binding = 0) uniform GlobalUBO
{
    mat4 Projection;
    mat4 View;
    mat4 InvView;
    mat4 InvProjection;
    vec4 CameraPosition;
    ivec4 ScreenResolution_NumRaysPerPixel_FrameNumber;
}  u_UBO;

layout (input_attachment_index = 0, set = 1, binding = 1) uniform subpassInput u_Current;
layout (input_attachment_index = 1, set = 1, binding = 2) uniform subpassInput u_Previous;

void main()
{
    o_Next = vec4(0.0, 1.0, 0.0, 1.0);
}
#version 450

layout (location = 0) in vec2 v_UV;

layout (location = 1) out vec4 o_Attachment1;
layout (location = 2) out vec4 o_Attachment2;

layout(push_constant) uniform FrameConstants
{
    int FrameNumber;
} u_FrameData;

void main()
{
    if(u_FrameData.FrameNumber % 2 == 0)
        o_Attachment1 = vec4(0.0, 1.0, 0.0, 1.0);
    else
        o_Attachment2 = vec4(1.0, 0.0, 0.0, 1.0);
}
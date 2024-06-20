#version 450

layout (location = 0) out vec2 v_UV;
layout (location = 1) out vec3 v_Ro;
layout (location = 2) out vec3 v_Rd;

layout(set = 0, binding = 0) uniform GlobalUBO
{
    mat4 Projection;
    mat4 View;
    mat4 InvView;
    mat4 InvProjection;
    vec4 CameraPosition;
    ivec4 ScreenResolution_NumRaysPerPixel_FrameNumber;
}  u_UBO;

void main()
{
    v_UV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

    vec3 ndcCoords = vec3(v_UV * 2.0f - 1.0f, 0.0);

    v_Ro = u_UBO.CameraPosition.xyz;

    vec3 viewPosition = (u_UBO.InvProjection * vec4(ndcCoords, 1.0)).xyz;
    v_Rd = (u_UBO.InvView * vec4(viewPosition, 0.0)).xyz;

    gl_Position = vec4(ndcCoords, 1.0f);
}

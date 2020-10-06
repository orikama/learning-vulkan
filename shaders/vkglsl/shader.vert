#version 450
#extension GL_ARB_separate_shader_objects : enable


in layout(location = 0) vec2 in_position;
in layout(location = 1) vec3 in_color;

out layout(location = 0) vec3 out_fragColor;

uniform layout(binding = 0) ubo_MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo_mvp;


void main()
{
    out_fragColor = in_color;
    gl_Position = ubo_mvp.projection * ubo_mvp.view * ubo_mvp.model * vec4(in_position, 0.0, 1.0);
}

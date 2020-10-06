#version 450
#extension GL_ARB_separate_shader_objects : enable


in layout(location = 0) vec3 in_fragColor;

out layout(location = 0) vec4 out_color;


void main()
{
    out_color = vec4(in_fragColor, 1.0);
}

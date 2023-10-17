#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
layout(binding = 1, set = 0, scalar) buffer storageBuffer
{
  vec3 imageData[];
};

void main() 
{
  	const uvec2 resolution = uvec2(1920, 1080);
	const uvec2 pixel = gl_LaunchIDEXT.xy;
	uint linearIndex       = resolution.x * pixel.y + pixel.x;
  	imageData[linearIndex] = vec3(0.1, 0.1, 0.8);
}

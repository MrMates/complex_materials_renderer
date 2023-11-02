#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.shader"

layout(location = 0) rayPayloadInEXT hitPayload prd;
hitAttributeEXT vec3 attribs;

void main()
{
  // if (attribs.x < 0.5) {
  //   ignoreIntersectionEXT;
  // }
  // hitValue = attribs;
}
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.shader"

layout(location = 0) rayPayloadEXT hitPayload prd;

void main()
{
      // set color to black
  prd.directLight = vec3(0.8, 0.0, 0.0);
  // no more reflections
//   prd.nextRayOrigin = vec3(0.0, 0.0, 0.0);
//   prd.nextRayDirection = vec3(0.0, 0.0, 0.0);
}
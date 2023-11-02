#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.shader"

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
layout(binding = 1, set = 0, scalar) buffer storageBuffer
{
	vec3 imageData[];
};

layout(location = 0) rayPayloadEXT hitPayload prd;


void main()
{
  const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
  const vec2 inUV        = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
  vec2       d           = inUV * 2.0 - 1.0;

  vec3 origin    = vec3(-0.001, 2.0, 10.0);

  const float fovVerticalSlope = 1.0 / 5.0;

  const vec2 screenUV = vec2((2.0 * pixelCenter.x - gl_LaunchSizeEXT.x) / gl_LaunchSizeEXT.y,    //
                               -(2.0 * pixelCenter.y - gl_LaunchSizeEXT.y) / gl_LaunchSizeEXT.y);  // Flip the y axis
  vec3 direction = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
  direction = normalize(direction);

  uint  rayFlags = gl_RayFlagsOpaqueEXT;
  float tMin     = 0.001;
  float tMax     = 10000.0;

  traceRayEXT(topLevelAS,     // acceleration structure
              rayFlags,       // rayFlags
              0xFF,           // cullMask
              0,              // sbtRecordOffset
              0,              // sbtRecordStride
              0,              // missIndex
              origin,     // ray origin
              tMin,           // ray min range
              direction,  // ray direction
              tMax,           // ray max range
              0               // payload (location = 0)
  );

  uint linearIndex = gl_LaunchSizeEXT.x * gl_LaunchIDEXT.y + gl_LaunchIDEXT.x;
  imageData[linearIndex] = prd.directLight;
}





// void main() 
// {
// 	const uvec2 pixel = gl_LaunchIDEXT.xy;
// 	const uvec2 resolution = gl_LaunchSizeEXT.xy;

//   	const vec2 pixelCenter = vec2(pixel) + vec2(0.5);
//     const vec2 inUV = pixelCenter/vec2(resolution);
//     vec2 d = inUV * 2.0 - 1.0;

// 	const vec3 cameraOrigin = vec3(-0.001, 2.0, 10.0);
// 	const float fovVerticalSlope = 1.0 / 5.0;

	

// 	const vec2 screenUV = vec2((2.0 * pixelCenter.x - resolution.x) / resolution.y,    //
//                                -(2.0 * pixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis

// 	vec3 rayDirection = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
//     rayDirection      = normalize(rayDirection);

// 	// Ray setup
// 	uint  rayFlags = gl_RayFlagsNoneEXT;
//   	float tMin     = 0.001;
//   	float tMax     = 10000.0;

// 	// Payload setup
// 	prd.nextRayOrigin    = cameraOrigin;
// 	prd.nextRayDirection = rayDirection;
// 	prd.nextFactor = vec3(1.0);
// 	prd.rngState = resolution.x * pixel.y + pixel.x;
	
// 	// Trace setup
// 	vec3 accumulatedColor = vec3(0.0);
// 	vec3 contribution = vec3(1.0);

// 	// Limit the kernel to trace at most 32 segments.
// 	int level = 0;
// 	int maxLevel = 1;
//     while (level < maxLevel && length(prd.nextRayDirection) > 0.1 && length(contribution) > 0.001 )
// 	{
// 		prd.level = level;

// 		traceRayEXT(topLevelAS, 	// acceleration structure
// 			rayFlags,       		// rayFlags
// 			0xFF,           		// cullMask
// 			0,              		// sbtRecordOffset
// 			0,              		// sbtRecordStride
// 			0,              		// missIndex
// 			prd.nextRayOrigin,		// ray origin
// 			tMin,           		// ray min range
// 			prd.nextRayDirection,	// ray direction
// 			tMax,           		// ray max range
// 			0               		// payload (location = 0)
// 		);

// 		accumulatedColor += contribution * prd.directLight;
// 		contribution *= prd.nextFactor;
// 		level++;
// 	}

// 	uint linearIndex       = resolution.x * pixel.y + pixel.x;
// 	imageData[linearIndex] = accumulatedColor;
// }

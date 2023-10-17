#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE
// note: this is a polyglot file for C++ and GLSL, so I'm using regular header guards
// for GLSL compatibility

#ifdef __cplusplus
#include "nvmath/nvmath.h"
// GLSL Types
using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;
using mat4 = nvmath::mat4f;
using uint = unsigned int;
#endif

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
#define START_BINDING(a) enum a {
#define END_BINDING() }
#else
#define START_BINDING(a)  const uint
#define END_BINDING() 
#endif

START_BINDING(SceneBindings)
eGlobals = 0,  // Global uniform containing camera matrices
eObjDescs = 1,  // Access to the object descriptions
eTextures = 2   // Access to textures
END_BINDING();

START_BINDING(RtxBindings)
eTlas = 0,  // Top-level acceleration structure
eOutBuffer = 1   // Ray tracer output buffer
END_BINDING();
// clang-format on

struct MediaCoeffs
{
	vec3 absorption;
	vec3 scattering;
};

#endif // END COMMON_HOST_DEVICE
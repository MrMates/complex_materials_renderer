#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require

#include "raycommon.shader"


layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(buffer_reference, buffer_reference_align=8, scalar) buffer VertexBuffer {
    vec3 v[];
};

layout(buffer_reference, buffer_reference_align=8, scalar) buffer IndexBuffer {
    uint i[];
};

layout(shaderRecordEXT, std430) buffer SBT {
    VertexBuffer verts;
    IndexBuffer indices;
};

// Barycentric coords from intersection shader
hitAttributeEXT vec2 attribs;




void main()
{
  // Indices of the triangle
  const uint i0 = indices.i[3 * gl_PrimitiveID + 0];
  const uint i1 = indices.i[3 * gl_PrimitiveID + 1];
  const uint i2 = indices.i[3 * gl_PrimitiveID + 2];

  // Vertex of the triangle
  vec3 v0 = verts.v[i0];
  vec3 v1 = verts.v[i1];
  vec3 v2 = verts.v[i2];

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Computing the coordinates of the hit position
  const vec3 pos      = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

  // Computing the normal at hit position
  vec3 nrm      = cross(v1 - v0, v2 - v0);
  nrm = nrm * barycentrics.x + nrm * barycentrics.y + nrm * barycentrics.z;
  const vec3 worldNrm = normalize(vec3(nrm * gl_WorldToObjectEXT));  // Transforming the normal to world space

  // Vector toward the light

  vec3 lDir      = vec3(-0.001, 2.0, 10.0) - worldPos;
  float lightDistance  = length(lDir);
  float lightIntensity = 1000 / (lightDistance * lightDistance);
  vec3  L              = normalize(lDir);

  // Diffuse
  vec3 color = vec3(0.8);
  // Lambertian
  float dotNL = max(dot(worldNrm, L), 0.0);
  vec3 diffuse = color * dotNL;

  float attenuation = 1;


    // Compute specular only if not in shadow
  const float kPi        = 3.14159265;
  const float kShininess = 60.0;

  // Specular
  const float kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
  vec3        V                   = normalize(-gl_WorldRayDirectionEXT);
  vec3        R                   = reflect(-L, worldNrm);
  float       spec            = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

  vec3 specular = vec3(spec);

  // Tracing shadow ray only if the light is visible from the surface
  // if(dot(worldNrm, L) > 0)
  // {
  //   float tMin   = 0.001;
  //   float tMax   = lightDistance;
  //   vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  //   vec3  rayDir = L;
  //   uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  //   isShadowed   = true;
  //   traceRayEXT(topLevelAS,  // acceleration structure
  //               flags,       // rayFlags
  //               0xFF,        // cullMask
  //               0,           // sbtRecordOffset
  //               0,           // sbtRecordStride
  //               1,           // missIndex
  //               origin,      // ray origin
  //               tMin,        // ray min range
  //               rayDir,      // ray direction
  //               tMax,        // ray max range
  //               1            // payload (location = 1)
  //   );

  //   if(isShadowed)
  //   {
  //     attenuation = 0.3;
  //   }
  //   else
  //   {
  //     // Specular
  //     specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, worldNrm);
  //   }
  // }

  prd.directLight = vec3(lightIntensity * attenuation * (diffuse + specular));
}









// void main()
// {
//   // Get the indices of the vertices of the triangle
//   const uint i0 = indices.i[3 * gl_PrimitiveID + 0];
//   const uint i1 = indices.i[3 * gl_PrimitiveID + 1];
//   const uint i2 = indices.i[3 * gl_PrimitiveID + 2];

//   // Get the vertices of the triangle
//   const vec3 v0 = verts.v[i0];
//   const vec3 v1 = verts.v[i1];
//   const vec3 v2 = verts.v[i2];


//   // Triangle vertices to local
//   vec3 localNormal = normalize( cross( v1 - v0, v2 - v0 ) );
//   vec3 localPosition = v0 * attribs.x + v1 * attribs.y + v2 * attribs.z;

//   // Local to World
//   vec3 normal = normalize((localNormal * gl_WorldToObjectEXT).xyz);  //gl_ObjectToWorldEXT * vec4(localNormal, 1.0);
//   vec3 position = gl_ObjectToWorldEXT * vec4(localPosition, 1.0);

//   prd.nextRayOrigin = position - 0.0001 * sign(dot(prd.nextRayDirection, normal)) * normal;

//   // LAMBERTIAN MODEL
//   // For a random diffuse bounce direction, we follow the approach of
//   // Ray Tracing in One Weekend, and generate a random point on a sphere
//   // of radius 1 centered at the normal. This uses the random_unit_vector
//   // function from chapter 8.5:
//   const float theta = 6.2831853 * stepAndOutputRNGFloat(prd.rngState);   // Random in [0, 2pi]
//   const float u     = 2.0 * stepAndOutputRNGFloat(prd.rngState) - 1.0;  // Random in [-1, 1]
//   const float r     = sqrt(1.0 - u * u);
//   prd.nextRayDirection  = normal + vec3(r * cos(theta), r * sin(theta), u);
//   // Then normalize the ray direction:
//   prd.nextRayDirection = normalize(prd.nextRayDirection);

//   vec3 color = vec3(0.8, 0.8, 0.8);

//   const float dotX = dot(normal, vec3(1.0, 0.0, 0.0));
//   if(dotX > 0.99)
//   {
//     color = vec3(0.8, 0.0, 0.0);
//   }
//   else if(dotX < -0.99)
//   {
//     color = vec3(0.0, 0.8, 0.0);
//   }

//   prd.directLight *= color;
//   prd.nextFactor = color;

// }

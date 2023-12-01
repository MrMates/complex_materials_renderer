#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_debug_printf : enable

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
layout(binding = 0, set = 0, scalar) buffer storageBuffer
{
  vec3 imageData[];
};
layout(binding = 1, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 2, set = 0, scalar) buffer Vertices
{
  vec3 vertices[];
};
layout(binding = 3, set = 0, scalar) buffer Indices
{
  uint indices[];
};
layout(binding = 4, set = 0, scalar) buffer MatIDs
{
  uint matIds[];
};

struct Medium
{
  vec3 absorption; // sigma a
  vec3 scattering; // sigma s
  vec3 anisotropy; // g
};

// Credit: https://github.com/mitsuba-renderer/mitsuba/blob/10af06f365886c1b6dd8818e0a3841078a62f283/src/medium/materials.h#L34
// "Apple", { 2.29f, 2.39f, 1.97f }, { 0.0030f, 0.0034f, 0.046f  }, { 0.0f, 0.0f, 0.0f }
// "Sprite" { 0.00011f, 0.00014f, 0.00014f }, { 0.00189f, 0.00183f, 0.00200f }, { 0.94300f, 0.95300f, 0.95200f }
Medium appleMedium = Medium(vec3(2.29, 2.39, 1.97), vec3(0.0030, 0.0034, 0.046), vec3(0));
Medium spriteMedium = Medium(vec3(0.00011, 0.00014, 0.00014), vec3(0.00189, 0.00183, 0.00200), vec3(0.94300, 0.95300, 0.95200));
// The camera is located at (-0.001, 0, 53).
const vec3 cameraOrigin = vec3(-0.001, 1.0, 6.0);

// Returns the color of the sky in a given direction (in linear color space)
vec3 skyColor(vec3 direction)
{
  // +y in world space is up, so:
  if(direction.y > 0.0f)
  {
    return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
  }
  else
  {
    return vec3(0.03f);
  }
}

struct HitInfo
{
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
  uint matID;
};

vec3 getTransmittance(float dist)
{
  vec3 extinction = spriteMedium.scattering + spriteMedium.absorption; // sigma t

  return exp(dist * -extinction);
}

// L(x, omega) - radiance, light; here it's the resulting color
vec3 calculateMediaInteraction(HitInfo hitInfo)
{

// //   Simplified transmittance for homogeneous media
//   vec3 transmittance = exp((hitInfo.worldPosition - cameraOrigin) * extinction);

//   if (max(max(transmittance.r, transmittance.g), transmittance.b) < 1e-20) 
//   {
//     transmittance = vec3(0);
//   }

//   vec3 radiance = transmittance; //* appleMedium.scattering; // * 10 /*direct light*/; // TODO: * delta t (interval samplingu)

//   return radiance;
  return vec3(1);
}

HitInfo getObjectHitInfo(rayQueryEXT rayQuery, bool commited)
{
  HitInfo result;

  // Get the ID of the triangle
  int primitiveID;
  if (commited)
  {
    primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
  }
  else
  {
    primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
  }

  result.matID = matIds[primitiveID];

  // Get the indices of the vertices of the triangle
  const uint i0 = indices[3 * primitiveID + 0];
  const uint i1 = indices[3 * primitiveID + 1];
  const uint i2 = indices[3 * primitiveID + 2];

  // Get the vertices of the triangle
  const vec3 v0 = vertices[i0];
  const vec3 v1 = vertices[i1];
  const vec3 v2 = vertices[i2];

  // Get the barycentric coordinates of the intersection
  vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
  barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;

  // Compute the coordinates of the intersection
  const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // Transform from object space to world space:
  const mat4x3 objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
  result.worldPosition       = objectToWorld * vec4(objectPos, 1.0f);

  // Compute the normal of the triangle in object space, using the right-hand rule:
  //    v2      .
  //    |\      .
  //    | \     .
  //    |/ \    .
  //    /   \   .
  //   /|    \  .
  //  L v0---v1 .
  // n
  const vec3 objectNormal = cross(v1 - v0, v2 - v0);
  // Get object to world inverse matrix
  const mat4x3 objectToWorldInverse = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
  // For the main tutorial, object space is the same as world space:
  result.worldNormal = normalize((objectNormal * objectToWorldInverse).xyz);

  result.color = vec3(0.8f);

  const float dotX = dot(result.worldNormal, vec3(1.0, 0.0, 0.0));
  const float dotY = dot(result.worldNormal, vec3(0.0, 1.0, 0.0));
  if(dotX > 0.99)
  {
    result.color = vec3(0.8, 0.0, 0.0);
  }
  else if(dotX < -0.99)
  {
    result.color = vec3(0.0, 0.8, 0.0);
  }

  return result;
}

// Random number generation using pcg32i_random_t, using inc = 1. Our random state is a uint.
uint stepRNG(uint rngState)
{
  return rngState * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rngState  = stepRNG(rngState);
  uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}

uint seed = stepRNG(123456);
float sampleDistance()
{
  float rand = stepAndOutputRNGFloat(seed);
  float sampleDensity = max(max(spriteMedium.scattering.r, spriteMedium.scattering.g), spriteMedium.scattering.b);
  float sampled = -1.0;

  if (rand < 0.5)
  {
    sampled = -log(1-rand) / sampleDensity;
    // debugPrintfEXT("My float is %f", sampled);
  }

  return sampled;
  //TODO: check jestli nejsme pryč z média
}

void main()
{
  // The resolution of the buffer, which in this case is a hardcoded vector
  // of 2 unsigned integers:
  const uvec2 resolution = uvec2(1920, 1080);

  // Get the coordinates of the pixel for this invocation:
  //
  // .-------.-> x
  // |       |
  // |       |
  // '-------'
  // v
  // y
  const uvec2 pixel = gl_GlobalInvocationID.xy;

  // If the pixel is outside of the image, don't do anything:
  if((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
  {
    return;
  }
  // State of the random number generator.
  uint rngState = resolution.x * pixel.y + pixel.x;  // Initial seed

  // This scene uses a right-handed coordinate system like the OBJ file format, where the
  // +x axis points right, the +y axis points up, and the -z axis points into the screen.

  // Define the field of view by the vertical slope of the topmost rays:
  const float fovVerticalSlope = 1.0 / 5.0;

  // The sum of the colors of all of the samples.
  vec3 summedPixelColor = vec3(0.0);

  // Limit the kernel to trace at most 64 samples.
  const int NUM_SAMPLES = 64;
  for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
  {
    // Rays always originate at the camera for now. In the future, they'll
    // bounce around the scene.
    vec3 rayOrigin = cameraOrigin;
    // Compute the direction of the ray for this pixel. To do this, we first
    // transform the screen coordinates to look like this, where a is the
    // aspect ratio (width/height) of the screen:
    //           1
    //    .------+------.
    //    |      |      |
    // -a + ---- 0 ---- + a
    //    |      |      |
    //    '------+------'
    //          -1
    const vec2 randomPixelCenter = vec2(pixel) + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    const vec2 screenUV          = vec2((2.0 * randomPixelCenter.x - resolution.x) / resolution.y,    //
                               -(2.0 * randomPixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis
    // Create a ray direction:
    vec3 rayDirection = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
    rayDirection      = normalize(rayDirection);

    vec3 accumulatedRayColor = vec3(1.0);  // The amount of light that made it to the end of the current ray.

    // Limit the kernel to trace at most 32 segments.
    for(int tracedSegments = 0; tracedSegments < 32; tracedSegments++)
    {
      // Trace the ray and see if and where it intersects the scene!
      // First, initialize a ray query object:
      rayQueryEXT rayQuery;
      rayQueryInitializeEXT(rayQuery,              // Ray query
                            tlas,                  // Top-level acceleration structure
                            gl_RayFlagsNoneEXT,    // Ray flags
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            rayOrigin,             // Ray origin
                            0.0,                   // Minimum t-value
                            rayDirection,          // Ray direction
                            10000.0);              // Maximum t-value

      // Start traversal, and loop over all ray-scene intersections. When this finishes,
      // rayQuery stores a "committed" intersection, the closest intersection (if any).
      bool inObject = false;
      float dist;

      while(rayQueryProceedEXT(rayQuery))
      {
        // Commits the closest hit
        if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
        {
          HitInfo hitInfo = getObjectHitInfo(rayQuery, false);
          if(hitInfo.matID == 1) // Media intersection
          {
            dist = sampleDistance();
            if (dist > 0)
            {
              rayQueryGenerateIntersectionEXT(rayQuery, dist);
            }
          }
          else // Commits opaque (non-media) intersections right away
          {
            rayQueryConfirmIntersectionEXT(rayQuery);
          }
        }
      }

      // Get the type of committed (true) intersection - nothing, a triangle, or
      // a generated object
      if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
      {
        // Ray hit a triangle
        HitInfo hitInfo = getObjectHitInfo(rayQuery, true);


        // Start a new ray at the hit position, but offset it slightly along
        // the normal against rayDirection:
        rayOrigin = hitInfo.worldPosition - 0.0001 * sign(dot(rayDirection, hitInfo.worldNormal)) * hitInfo.worldNormal;

        // if(hitInfo.matID == 1) // Media
        // {
        //   accumulatedRayColor *= calculateMediaInteraction(hitInfo);
        // }
        // else
        // {
        //   // Apply color absorption
        //   accumulatedRayColor *= hitInfo.color;
        // }

        accumulatedRayColor *= hitInfo.color;


        // LAMBERTIAN MODEL
        // For a random diffuse bounce direction, we follow the approach of
        // Ray Tracing in One Weekend, and generate a random point on a sphere
        // of radius 1 centered at the normal. This uses the random_unit_vector
        // function from chapter 8.5:
        const float theta = 6.2831853 * stepAndOutputRNGFloat(rngState);   // Random in [0, 2pi]
        const float u     = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;  // Random in [-1, 1]
        const float r     = sqrt(1.0 - u * u);
        rayDirection      = hitInfo.worldNormal + vec3(r * cos(theta), r * sin(theta), u);
        // Then normalize the ray direction:
        rayDirection = normalize(rayDirection);

        if(hitInfo.matID == 0)
        {
          // light someday hopefully
        }
        // MIRROR REFLECTION
        // Reflect the direction of the ray using the triangle normal:
        // rayDirection = reflect(rayDirection, hitInfo.worldNormal);
      }
      else if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionGeneratedEXT){
        accumulatedRayColor *= spriteMedium.scattering * getTransmittance(dist);
      }
      else
      {
        // Ray hit the sky
        accumulatedRayColor *= skyColor(rayDirection);
        
        // Sum this with the pixel's other samples.
        // (Note that we treat a ray that didn't find a light source as if it had
        // an accumulated color of (0, 0, 0)).
        summedPixelColor += accumulatedRayColor;
    
        break;
      }
    }
  }

  // Get the index of this invocation in the buffer:
  uint linearIndex       = resolution.x * pixel.y + pixel.x;
  imageData[linearIndex] = summedPixelColor / float(NUM_SAMPLES);  // Take the average
}
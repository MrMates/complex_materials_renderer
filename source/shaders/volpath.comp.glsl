#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_debug_printf : enable

const float INV_FOURPI = 0.07957747154594767;
const float PI = 3.14159265359;
const float TWOPI = 6.28318530718;

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

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
  vec3 scattering; // sigma s
  vec3 absorption; // sigma a
  vec3 anisotropy; // g
};

struct RayPayload
{
  int depth;
  float t;
};

// Credit: https://github.com/mitsuba-renderer/mitsuba/blob/10af06f365886c1b6dd8818e0a3841078a62f283/src/medium/materials.h#L34
// "Apple", { 2.29f, 2.39f, 1.97f }, { 0.0030f, 0.0034f, 0.046f  }, { 0.0f, 0.0f, 0.0f }
// "Sprite" { 0.00011f, 0.00014f, 0.00014f }, { 0.00189f, 0.00183f, 0.00200f }, { 0.94300f, 0.95300f, 0.95200f }
// "Regular Milk",                { 18.2052f, 20.3826f, 22.3698f }, { 0.00153f, 0.00460f, 0.01993f }, { 0.75000f, 0.71400f, 0.68100f }
// "Ketchup",                    { 0.18f, 0.07f, 0.03f }, { 0.061f,  0.97f,   1.45f   }, { 0.0f, 0.0f, 0.0f }},
Medium appleMedium = Medium(vec3(2.29, 2.39, 1.97), vec3(0.0030, 0.0034, 0.046), vec3(0));
Medium spriteMedium = Medium(vec3(0.00011, 0.00014, 0.00014), vec3(0.00189, 0.00183, 0.00200), vec3(0.94300, 0.95300, 0.95200));
Medium milkMedium = Medium(vec3(18.2052, 20.3826, 22.3698), vec3(0.00153, 0.00460, 0.01993), vec3(0.75000, 0.71400, 0.68100));
Medium ketchupMedium = Medium(vec3(0.18, 0.07, 0.03), vec3(0.061, 0.97, 1.45), vec3(0));
Medium presso = Medium(vec3(7.78262, 8.13050, 8.53875), vec3(4.79838, 6.57512, 8.84925), vec3(0.90700, 0.89600, 0.88000));

Medium selectedMedium = presso;

// The camera is located at (-0.001, 1.0, 6.0).
const vec3 cameraOrigin = vec3(-0.001, 1.5, 8.0);

const vec3 lightPos = vec3(-1.001, 1.5, 8.0);
const vec3 lightColor = vec3(0.8, 0.8, 0.6);
const vec3 lightIntensity = lightColor * 10.0;

struct HitInfo
{
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
  uint matID;
};

struct MediumSample
{
  float t;
  vec3 point;
  vec3 absorption;
  vec3 scattering;
  float probFail;
  float probSuccess;
  vec3 transmittance;
};

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
  vec3 barycentrics;
  if (commited)
  {
    barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
  }
  else
  {
    barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, false));
  }
  barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;

  // Compute the coordinates of the intersection
  const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // Transform from object space to world space:
  mat4x3 objectToWorld;
  if (commited)
  {
    objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
  }
  else
  {
    objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, false);
  }
  result.worldPosition       = objectToWorld * vec4(objectPos, 1.0f);

  // Compute the normal of the triangle in object space, using the right-hand rule
  const vec3 objectNormal = cross(v1 - v0, v2 - v0);
  // Get object to world inverse matrix
  mat4x3 objectToWorldInverse;
  if (commited)
  {
    objectToWorldInverse = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
  }
  else
  {
    objectToWorldInverse = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, false);
  }
  // For the main tutorial, object space is the same as world space:
  result.worldNormal = normalize((objectNormal * objectToWorldInverse).xyz);

  result.color = vec3(0.8f);

  const float dotX = dot(result.worldNormal, vec3(1.0, 0.0, 0.0));
  const float dotY = dot(result.worldNormal, vec3(0.0, 1.0, 0.0));
  if(matIds[primitiveID] == 2 && dotX > 0.99)
  {
    result.color = vec3(0.8, 0.0, 0.0);
  }
  else if(matIds[primitiveID] == 2 && dotX < -0.99)
  {
    result.color = vec3(0.0, 0.8, 0.0);
  }
  else if (matIds[primitiveID] == 0)
  {
    result.color = vec3(0.2, 0.2, 0.2);
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
bool shown = false;

vec3 evalTransmittance(float dist)
{
  vec3 extinction = selectedMedium.absorption + selectedMedium.scattering;
  vec3 transmittance = exp(extinction * (-dist));
  return transmittance;
}


vec3 sampleDirectLight(vec3 point, vec3 normal, inout uint rngState)
{
  vec3 lightDir = lightPos - point;
  float lightDist = length(lightDir);
  float invDist = 1.0 / lightDist;

  vec3 lightValue = lightIntensity * invDist * invDist;

  lightDir = normalize(lightDir);
  vec3 transmittance = vec3(1.0);

  rayQueryEXT lightRayQuery;
  rayQueryInitializeEXT(lightRayQuery,              // Ray query
                        tlas,                  // Top-level acceleration structure
                        gl_RayFlagsNoneEXT,    // Ray flags
                        0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                        point + normal * 0.001,             // Ray origin
                        0.0,                   // Minimum t-value
                        lightDir,          // Ray direction
                        lightDist);              // Maximum t-value

  while(rayQueryProceedEXT(lightRayQuery))
  {
    rayQueryConfirmIntersectionEXT(lightRayQuery);
  }

  if(rayQueryGetIntersectionTypeEXT(lightRayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
  {
    HitInfo hitInfo = getObjectHitInfo(lightRayQuery, true);

    if (hitInfo.matID != 1) // not medium
    {
      transmittance = vec3(0); // light is fully occluded
    }

    // TODO: Sample distance only to the end of media (if it's needed)
    vec3 mediumTransmittance = evalTransmittance(lightDist);
    transmittance *= mediumTransmittance;
  }

  lightValue *= transmittance;
  return lightValue;
}

struct PhaseFunctionSample {
  vec3 inDir;
  vec3 outDir;
};

// Implementation of the Henyey-Greenstein phase function via Mitsuba
// Credit: https://github.com/mitsuba-renderer/mitsuba/blob/master/src/phase/hg.cpp
float evalPhaseFunction(PhaseFunctionSample phase)
{
  // Using the average of the three channels from g
  float g = dot(selectedMedium.anisotropy, vec3(1.0f)) / 3.0f;

  float tmp = 1.0f + g * g + 2.0f * g * dot(phase.inDir, phase.outDir);
  return INV_FOURPI * (1.0f - g * g) / (tmp * sqrt(tmp));
}

float samplePhaseFunction(inout PhaseFunctionSample phase, inout uint seed)
{
  float g = dot(selectedMedium.anisotropy, vec3(1.0f)) / 3.0f;

  float x = stepAndOutputRNGFloat(seed);
  float y = stepAndOutputRNGFloat(seed);

  float temp = (1.0f - g * g) / (1.0f - g + 2.0f * g * x);
  float cosTheta = (1.0f + g * g - temp * temp) / (2.0f * g);

  float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
  float phi = TWOPI * y;
  float sinPhi = sin(phi);
  float cosPhi = cos(phi);

  vec3 localDir = vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

  vec3 s,t;

  // Using incident direction vector as normal for coordinate system
  vec3 normal = -phase.inDir;
  if (abs(normal.x) > abs(normal.y)) {
    float temp = 1.0f / sqrt(normal.x * normal.x + normal.z * normal.z);
    t = vec3(normal.z * temp, 0.0f, -normal.x * temp);
  } else {
    float temp = 1.0f / sqrt(normal.y * normal.y + normal.z * normal.z);
    t = vec3(0.0f, normal.z * temp, -normal.y * temp);
  }
  s = cross(t, normal);

  // Converting localDir to world space
  vec3 worldDir = s * localDir.x + t * localDir.y + normal * localDir.z;
  phase.outDir = worldDir;

  return 1.0f;
}


bool sampleDistance(inout MediumSample mSample, float dist, inout uint seed)
{
  float rand = stepAndOutputRNGFloat(seed);
  vec3 extinction = mSample.absorption + mSample.scattering;
  // Strategy: Single - select the lowest variance channel for density
  float sampleDensity = min(min(extinction.r, extinction.g), extinction.b);

  float sampled;

  float sampleWeight = -1.0;
  for (int i = 0; i < 3; i++)
  {
    float albedo = mSample.scattering[i] / extinction[i];
    if (albedo > sampleWeight)
    {
      sampleWeight = albedo;
    }
  }

  if (sampleWeight > 0)
  {
    sampleWeight = max(sampleWeight, 0.5);
  }

  if (rand < sampleWeight)
  {
    rand /= sampleWeight;
    sampled = -log(1-rand) / sampleDensity;
    // debugPrintfEXT("My float is %f", sampled);
  }
  else
  {
    sampled = 1.0 / 0.0; // +inf (infinite distance, no interaction)
  }

  bool success = true;

  if (sampled < dist) // Checking if sampled distance is still in media
  {
    mSample.t = sampled; // How deep in the media we are
  }
  else
  {
    sampled = dist;
    success = false;
  }

  mSample.probFail = exp(-sampleDensity * sampled);
  mSample.probSuccess = sampleDensity * mSample.probFail;

  mSample.transmittance = exp(extinction * (-sampled));

  mSample.probSuccess *= sampleWeight;
  mSample.probFail = sampleWeight * mSample.probFail + (1 - sampleWeight);

  if (max(max(mSample.transmittance.r, mSample.transmittance.g), mSample.transmittance.b) < 1e-20)
  {
    mSample.transmittance = vec3(0);
  }

  return success;
}

void main()
{
  // The resolution of the buffer, which in this case is a hardcoded vector
  // of 2 unsigned integers:
  const uvec2 resolution = uvec2(1920, 1080);

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
  vec3 summedPixelColor = vec3(0.0);

  // Limit the kernel to trace at most 64 samples.
  const int NUM_SAMPLES = 512;
  for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
  {
    // Starts at camera
    vec3 rayOrigin = cameraOrigin;

    // Randomly sample a point on the pixel
    const vec2 randomPixelCenter = vec2(pixel) + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));

    // Convert the pixel coordinates to screen coordinates:
    const vec2 screenUV          = vec2((2.0 * randomPixelCenter.x - resolution.x) / resolution.y,    //
                               -(2.0 * randomPixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis
    // Create a ray direction:
    vec3 rayDirection = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
    rayDirection      = normalize(rayDirection);

    vec3 accumulatedRayColor = vec3(0.0);  // The amount of light that made it to the end of the current ray.
    vec3 throughput = vec3(1.0);

    RayPayload payload = {0,0};

    float maxT = 10000.0;
    bool inMedium = false;
    vec3 nextPoint;
    
    bool scattered = false;
    // Limit the kernel to trace at most 32 segments.
    while(payload.depth < 32)
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
                            maxT);              // Maximum t-value

      float dist;
      while(rayQueryProceedEXT(rayQuery))
      {
          rayQueryConfirmIntersectionEXT(rayQuery);
      }

      // Get the type of committed (true) intersection
      if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
      {
        // Ray hit a triangle
        HitInfo hitInfo = getObjectHitInfo(rayQuery, true);
        
        vec3 newDirection = normalize(rayDirection);

        // Cast a ray to determine distance to the medium end
        rayQueryEXT rayQueryDist;
        rayQueryInitializeEXT(rayQueryDist,              // Ray query
                                    tlas,                  // Top-level acceleration structure
                                    gl_RayFlagsTerminateOnFirstHitEXT,    // Ray flags
                                    0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                                    hitInfo.worldPosition + newDirection * 0.0001,             // Ray origin
                                    0.0,                   // Minimum t-value
                                    newDirection,          // Ray direction
                                    10000.0);              // Maximum t-value

        rayQueryProceedEXT(rayQueryDist);

        dist = rayQueryGetIntersectionTEXT(rayQueryDist, false);
        MediumSample mSample = {0, vec3(0), selectedMedium.absorption, selectedMedium.scattering, 0, 0, vec3(0)};
        if(hitInfo.matID == 1 && sampleDistance(mSample, dist, rngState))
        {
          throughput *= mSample.scattering * mSample.transmittance / mSample.probSuccess;


          // Direct light
          vec3 lightValue = sampleDirectLight(hitInfo.worldPosition, hitInfo.worldNormal, rngState);
          PhaseFunctionSample phase = {-rayDirection, vec3(0)};
          float phaseEval = evalPhaseFunction(phase);
          accumulatedRayColor += throughput * lightValue * phaseEval;

          // Sample phase function
          float phaseVal = samplePhaseFunction(phase, rngState);
          if (phaseVal < 1e-20)
          {
            break;
          }
          throughput *= phaseVal;



          // Set the new ray direction
          rayDirection = phase.outDir;
          rayOrigin = hitInfo.worldPosition + rayDirection * mSample.t;
          scattered = true;
        } 
        else
        {
          if (hitInfo.matID == 1)
          {
            throughput *= mSample.transmittance / mSample.probFail;

            // Need to trace what's behind the medium
            rayQueryEXT rayQueryDist;
            rayQueryInitializeEXT(rayQueryDist,              // Ray query
                                    tlas,                  // Top-level acceleration structure
                                    gl_RayFlagsNoneEXT,    // Ray flags
                                    0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                                    hitInfo.worldPosition + dist*rayDirection + rayDirection * 0.0002,             // Ray origin
                                    0.0,                   // Minimum t-value
                                    rayDirection,          // Ray direction
                                    10000.0);              // Maximum t-value

            while(rayQueryProceedEXT(rayQueryDist))
            {
              rayQueryConfirmIntersectionEXT(rayQueryDist);
            }

            if(rayQueryGetIntersectionTypeEXT(rayQueryDist, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
            {
              HitInfo backgroundHitInfo = getObjectHitInfo(rayQueryDist, true);
              if (backgroundHitInfo.matID != 1)
              {
                throughput *= backgroundHitInfo.color;
              }
              // We want to set the origin to the same point as rayQueryDist to repeat the same query in the next iteration
              // That way we can even handle the whole media interaction for the next segment
              rayOrigin = hitInfo.worldPosition + dist*rayDirection + rayDirection * 0.0002;
            }
            payload.depth++;
            continue;
          }

          vec3 colorValue = vec3(1.0);
          if (hitInfo.matID != 1)
          {
            throughput *= hitInfo.color;
            colorValue = hitInfo.color;
          }

          vec3 lightValue = sampleDirectLight(hitInfo.worldPosition, hitInfo.worldNormal, rngState);
          accumulatedRayColor += throughput * lightValue * colorValue;

          // Lambertian model
          const float theta = 6.2831853 * stepAndOutputRNGFloat(rngState);   // Random in [0, 2pi]
          const float u     = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;  // Random in [-1, 1]
          const float r     = sqrt(1.0 - u * u);
          rayDirection      = hitInfo.worldNormal + vec3(r * cos(theta), r * sin(theta), u);
          // Then normalize the ray direction:
          rayDirection = normalize(rayDirection);
          rayOrigin = hitInfo.worldPosition + rayDirection * dist;
        }
      }
      payload.depth++;
      if (payload.depth > 16)
      {
        // Russian roulette
        float throughputMax = max(max(throughput.r, throughput.g), throughput.b);
        float q = min(throughputMax, 0.95);
        if (stepAndOutputRNGFloat(rngState) > q)
        {
          break;
        }
        throughput /= q;
      }
    }
    summedPixelColor += accumulatedRayColor;
  }


  float exposure = 1;
  vec3 hdrColor = summedPixelColor / float(NUM_SAMPLES);

  //Reinhard toneamp
  vec3 mappedColor = hdrColor / (hdrColor + vec3(1.0));
  mappedColor = pow(mappedColor, vec3(1.0 / exposure));

  // Get the index of this invocation in the buffer:
  uint linearIndex       = resolution.x * pixel.y + pixel.x;
  imageData[linearIndex] = mappedColor;  // Take the average
}
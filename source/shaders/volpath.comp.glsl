#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_debug_printf : enable

const float INV_FOURPI = 0.07957747154594767;
const float PI = 3.14159265359;
const float INV_PI = 0.31830988618;
const float TWOPI = 6.28318530718;

// Scaling factor for the media coefficients
// 1: 1 scene unit = 1 mm; 10: 1 scene unit = 1 cm; 1000: 1 scene unit = 1 m
const int scale = 10;

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

struct MediumBuffer
{
  float matID;
  vec3 scattering; // sigma s
  vec3 absorption; // sigma a
  vec3 anisotropy; // g
};

layout(binding = 0, set = 0, rgba32f) uniform image2D storageImage;
layout(binding = 1, set = 0) uniform accelerationStructureEXT tlas;
// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
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
layout(binding = 5, set = 0, scalar) buffer Media
{
  float mediaSize;
  MediumBuffer media[];
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

Medium none = Medium(vec3(0), vec3(0), vec3(0));

float airIOR = 1.00;
float mediaIOR = 1.55;
vec3 reflectance = vec3(0.8);

// The camera location in world space
const vec3 cameraOrigin = vec3(0.2, 4.2, 6.5);

const vec3 lightPos = vec3(-1.001, 5.0, 6.0);
const vec3 lightColor = vec3(0.8, 0.8, 0.6);
const vec3 lightIntensity = lightColor * 100.0;

struct HitInfo
{
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
  uint matID;
  bool hasMedium;
  Medium medium;
};

struct MediumSample
{
  float t;
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

  result.hasMedium = false;
  for (int i = 0; i < uint(mediaSize); i++)
  {
    if (result.matID == uint(media[i].matID))
    {
      result.medium = Medium(media[i].scattering * scale, media[i].absorption * scale, media[i].anisotropy);
      result.hasMedium = true;
      break;
    }
  }

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
  // Object space is the same as world space in this implementation
  result.worldNormal = normalize((objectNormal * objectToWorldInverse).xyz);

  result.color = vec3(0.8f);

  // Checkerboard pattern
  if((  int(floor(result.worldPosition.x)) % 2 == 0 && int(floor(result.worldPosition.y)) % 2 == 0) 
    || (int(floor(result.worldPosition.x)) % 2 != 0 && int(floor(result.worldPosition.y)) % 2 != 0))
  {
    result.color = vec3(0.8f, 0.8f, 0.8f);
  }
  else
  {
    result.color = vec3(0.3f, 0.3f, 0.3f);
  }

  // Cornell sides
  // const float dotX = dot(result.worldNormal, vec3(1.0, 0.0, 0.0));
  // const float dotY = dot(result.worldNormal, vec3(0.0, 1.0, 0.0));
  // if(dotX > 0.99)
  // {
  //   result.color = vec3(0.8, 0.0, 0.0);
  // }
  // else if(dotX < -0.99)
  // {
  //   result.color = vec3(0.0, 0.8, 0.0);
  // }

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

vec3 evalTransmittance(float dist, Medium medium)
{
  vec3 extinction = medium.absorption + medium.scattering;
  vec3 transmittance = exp(extinction * (-dist));
  return transmittance;
}

vec3 diffuseEval(vec3 wi, vec3 wo)
{
  if(wi.z <= 0.0 || wo.z <= 0.0)
  {
    return vec3(0.0);
  }

  return reflectance * (INV_PI * wo.z);
}

vec3 diffuseSample(vec3 wi, inout vec3 wo, inout uint rngState)
{
  if(wi.z <= 0)
  {
    return vec3(0.0);
  }

  // Dave Cline & Mitsuba implementation
  // (https://github.com/mitsuba-renderer/mitsuba/blob/master/src/libcore/warp.cpp#L81) 
  float r1 = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;
  float r2 = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;

  float phi, r;
  if(r1 == 0.0 && r2 == 0.0)
  {
    phi = 0.0;
    r = 0.0;
  }
  else if(r1 * r1 > r2 * r2)
  {
    r = r1;
    phi = (PI / 4.0) * (r2 / r1);
  }
  else
  {
    r = r2;
    phi = (PI / 2.0) - (r1 / r2) * (PI / 4.0);
  }

  float cosPhi = cos(phi);
  float sinPhi = sin(phi);

  vec2 diskCoords = vec2(r * cosPhi, r * sinPhi);

  float temp = 1.0 - diskCoords.x * diskCoords.x - diskCoords.y * diskCoords.y;
  if(temp <= 0.0)
  {
    wo = vec3(diskCoords.x, diskCoords.y, 1e-10);
  }
  else
  {
    wo = vec3(diskCoords.x, diskCoords.y, sqrt(temp));
  }

  return reflectance;
}

float getFresnelR(float n1, float n2, vec3 inDir, vec3 normal, bool fast)
{
  if (fast) {
    // Schlick's approximation
    float f = ((1.0 - n1 / n2)*(1.0 - n1 / n2)) / ((1.0 + n1 / n2)*(1.0 + n1 / n2));
    return f + (1.0 - f) * pow(1.0 - abs(dot(normalize(inDir), normalize(normal))), 5.0);
  }

  // Full Fresnel
  float theta1 = acos(dot(normalize(inDir), normalize(normal)));
  if (dot(inDir, normal) < 0.0) {
    theta1 = PI - theta1;
  }
  float theta2 = asin(n1/n2 * sin(theta1));


  float cosTheta1 = cos(theta1);
  float cosTheta2 = cos(theta2);

  float rs = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
  float rp = (n1 * cosTheta2 - n2 * cosTheta1) / (n1 * cosTheta2 + n2 * cosTheta1);

  return (rs * rs + rp * rp) / 2.0;
}

vec3 sampleDirectLight(vec3 point, vec3 normal, inout uint rngState, Medium medium)
{
  vec3 lightDir = lightPos - point;
  float lightDist = length(lightDir);
  float invDist = 1.0 / lightDist;

  vec3 lightValue = lightIntensity * invDist * invDist;

  // Shadow smoothing
  lightDir.x += 0.05 * (stepAndOutputRNGFloat(rngState) - 0.5);
  lightDir.y += 0.05 * (stepAndOutputRNGFloat(rngState) - 0.5);
  lightDir.z += 0.05 * (stepAndOutputRNGFloat(rngState) - 0.5);

  lightDir = normalize(lightDir);
  vec3 transmittance = vec3(1.0);

  rayQueryEXT lightRayQuery;
  rayQueryInitializeEXT(lightRayQuery,
                        tlas,
                        gl_RayFlagsNoneEXT,
                        0xFF,
                        point,
                        0.0001,
                        lightDir,
                        lightDist);

  while(rayQueryProceedEXT(lightRayQuery))
  {
    rayQueryConfirmIntersectionEXT(lightRayQuery);
  }

  if(rayQueryGetIntersectionTypeEXT(lightRayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
  {
    HitInfo hitInfo = getObjectHitInfo(lightRayQuery, true);

    if (!hitInfo.hasMedium) // not medium
    {
      // light is fully occluded
      return vec3(0.0);
    }

    lightDist -= rayQueryGetIntersectionTEXT(lightRayQuery, true);

    rayQueryEXT distRayQuery;
    rayQueryInitializeEXT(distRayQuery,
                          tlas,
                          gl_RayFlagsNoneEXT,
                          0xFF,
                          hitInfo.worldPosition,
                          0.0001,
                          lightDir,
                          lightDist);

    while(rayQueryProceedEXT(distRayQuery))
    {
      rayQueryConfirmIntersectionEXT(distRayQuery);
    }

    if(rayQueryGetIntersectionTypeEXT(distRayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
    {
      HitInfo mediumHitInfo = getObjectHitInfo(distRayQuery, true);
      if (!mediumHitInfo.hasMedium)
      {
        // light is fully occluded
        return vec3(0.0);
      }
      vec3 mediumTransmittance = evalTransmittance(rayQueryGetIntersectionTEXT(distRayQuery, true), mediumHitInfo.medium);
      transmittance *= mediumTransmittance;
    }
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
float evalPhaseFunction(PhaseFunctionSample phase, Medium medium)
{
  // Using the average of the three channels from g
  float g = dot(medium.anisotropy, vec3(1.0f)) / 3.0f;

  float tmp = 1.0f + g * g + 2.0f * g * dot(phase.inDir, phase.outDir);
  return INV_FOURPI * (1.0f - g * g) / (tmp * sqrt(tmp));
}

float samplePhaseFunction(inout PhaseFunctionSample phase, Medium medium, inout uint seed)
{
  float g = dot(medium.anisotropy, vec3(1.0f)) / 3.0f;

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
  }
  else
  {
    sampled = 500000; // no interaction
  }

  bool success = true;

  if (sampled < dist) // Checking if sampled distance is still in media
  {
    mSample.t = sampled; // How deep in the media we are
  }
  else
  {
    sampled = dist;
    mSample.t = dist;
    success = false;
  }

  mSample.probFail = exp(-sampleDensity * sampled);
  mSample.probSuccess = sampleDensity * mSample.probFail;

  mSample.transmittance = exp(extinction * (-sampled));

  mSample.probSuccess *= sampleWeight;
  mSample.probFail = sampleWeight * mSample.probFail + (1 - sampleWeight);

  if (max(max(mSample.transmittance.r, mSample.transmittance.g), mSample.transmittance.b) < 0.0001)
  {
    mSample.transmittance = vec3(0);
  }

  return success;
}

void main()
{
  // debugPrintfEXT("Size: %d; Mat %d: %f, %f, %f\n", uint(mediaSize), uint(media[0].matID), media[0].scattering.r, media[0].scattering.g, media[0].scattering.b);

  const ivec2 resolution = imageSize(storageImage);

  const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

  if((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
  {
    return;
  }

  uint rngState = uint(resolution.x * pixel.y + pixel.x);  // Initial seed

  // This scene uses a right-handed coordinate system like the OBJ file format, where the
  // +x axis points right, the +y axis points up, and the -z axis points into the screen.

  // Define the field of view by the vertical slope of the topmost rays:
  const float fovVerticalSlope = 1.0 / 5.0;
  vec3 summedPixelColor = vec3(0.0);

  const int NUM_SAMPLES = 2048;
  for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
  {
    vec3 rayOrigin = cameraOrigin;

    // Randomly sample a point on the pixel
    const vec2 randomPixelCenter = vec2(pixel) + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));

    // Convert the pixel coordinates to screen coordinates:
    const vec2 screenUV          = vec2((2.0 * randomPixelCenter.x - resolution.x) / resolution.y,    //
                               -(2.0 * randomPixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis

    vec3 rayDirection = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
    rayDirection      = normalize(rayDirection);

    vec3 accumulatedRayColor = vec3(0.0);  // The amount of light that made it to the end of the current ray.
    vec3 throughput = vec3(1.0);

    RayPayload payload = {0,0};

    // Limit the kernel to trace at most 32 segments.
    while(payload.depth < 128)
    {
      rayQueryEXT rayQuery;
      rayQueryInitializeEXT(rayQuery,              // Ray query
                            tlas,                  // Top-level acceleration structure
                            gl_RayFlagsNoneEXT,    // Ray flags
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            rayOrigin,             // Ray origin
                            0.0003,                   // Minimum t-value
                            rayDirection,          // Ray direction
                            10000.0);              // Maximum t-value


      float dist;
      while(rayQueryProceedEXT(rayQuery))
      {
          rayQueryConfirmIntersectionEXT(rayQuery);
      }

      // Get the type of committed (true) intersection
      if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
      {
        HitInfo hitInfo = getObjectHitInfo(rayQuery, true);

        if(hitInfo.hasMedium)
        {
          vec3 interactionNormal = hitInfo.worldNormal;
          float fromIOR = airIOR;
          float toIOR = mediaIOR;
          if (dot(rayDirection, hitInfo.worldNormal) > 0)
          {
            interactionNormal = -hitInfo.worldNormal;
            fromIOR = mediaIOR;
            toIOR = airIOR;
          }
          vec3 refractDir = refract(rayDirection, interactionNormal, fromIOR / toIOR);
          vec3 reflectDir = reflect(rayDirection, interactionNormal);

          float fresnelR = getFresnelR(fromIOR, toIOR, rayDirection, interactionNormal, false);
          float rand = stepAndOutputRNGFloat(rngState);

          if (rand < fresnelR)
          {
            // Reflect
            rayDirection = reflectDir;
            rayOrigin = hitInfo.worldPosition;// + rayDirection * 0.0001;
            payload.depth++;
            continue;
          }
          else
          {
            if (refractDir == vec3(0.0))
            {
              // Total internal reflection
              rayDirection = reflectDir;
            }
            else
            {
              // Refract
              rayDirection = refractDir;
            }
          }

        }

        vec3 newDirection = normalize(rayDirection);

        // Cast a ray to determine distance to the medium end
        rayQueryEXT rayQueryDist;
        rayQueryInitializeEXT(rayQueryDist,
                                    tlas,
                                    gl_RayFlagsTerminateOnFirstHitEXT,
                                    0xFF,
                                    hitInfo.worldPosition,
                                    0.0001,
                                    newDirection,
                                    10000.0);

        rayQueryProceedEXT(rayQueryDist);

        dist = rayQueryGetIntersectionTEXT(rayQueryDist, false);
        HitInfo mediumEndHitInfo = getObjectHitInfo(rayQueryDist, false);

        MediumSample mSample = {0, hitInfo.medium.absorption, hitInfo.medium.scattering, 0, 0, vec3(0)};
        if (hitInfo.hasMedium)
        {
          mSample.absorption = hitInfo.medium.absorption;
          mSample.scattering = hitInfo.medium.scattering;
        }
        if(hitInfo.hasMedium && sampleDistance(mSample, dist, rngState))
        {
          throughput *= mSample.scattering * mSample.transmittance / mSample.probSuccess;


          // Direct light
          vec3 lightValue = sampleDirectLight(hitInfo.worldPosition, hitInfo.worldNormal, rngState, hitInfo.medium);
          PhaseFunctionSample phase = {-rayDirection, vec3(0)};
          float phaseEval = evalPhaseFunction(phase, hitInfo.medium);

          accumulatedRayColor += throughput * lightValue * phaseEval;

          // Sample phase function
          float phaseVal = samplePhaseFunction(phase, hitInfo.medium, rngState);
          if (phaseVal < 1e-20)
          {
            break;
          }
          throughput *= phaseVal;

          // Set the new ray direction
          rayDirection = phase.outDir;
          rayOrigin = hitInfo.worldPosition + rayDirection * mSample.t;
        } 
        else
        {
          if (hitInfo.hasMedium)
          {
            // We're either out of media or RNG failed the sampling, so we just apply the transmittance 
            // and continue tracing behind the media
            throughput *= mSample.transmittance / mSample.probFail;

            // Move the origin to the end of the medium either way
            // Hit position + direction unaffected by reflection/refraction
            rayOrigin = hitInfo.worldPosition + rayDirection * mSample.t;
            
            vec3 interactionNormal = hitInfo.worldNormal;
            float fromIOR = mediaIOR;
            float toIOR = airIOR;
            if (dot(rayDirection, hitInfo.worldNormal) > 0)
            {
              interactionNormal = -hitInfo.worldNormal;
              fromIOR = airIOR;
              toIOR = mediaIOR;
            }

            vec3 refractDir = refract(rayDirection, interactionNormal, fromIOR / toIOR);
            vec3 reflectDir = reflect(rayDirection, interactionNormal);

            float fresnelR = getFresnelR(fromIOR, toIOR, rayDirection, interactionNormal, false);
            float rand = stepAndOutputRNGFloat(rngState);

            if (rand < fresnelR)
            {
              // Reflect
              rayDirection = reflectDir;
            }
            else
            {
              if (refractDir == vec3(0.0))
              {
                // Total internal reflection
                rayDirection = reflectDir;
              }
              else
              {
                // Refract
                rayDirection = refractDir;
                
              }
            }
            payload.depth++;
            continue;
          }

          // Shaded/lambertian section
          vec3 wo;

          if(dot(rayDirection, hitInfo.worldNormal) > 0)
          {
            rayDirection = -rayDirection;
          }

          vec3 bsdfVal = diffuseSample(-rayDirection, wo, rngState);
          if(bsdfVal == vec3(0.0))
          {
            break;
          }

          throughput *= bsdfVal;
          vec3 lightValue = sampleDirectLight(hitInfo.worldPosition, hitInfo.worldNormal, rngState, hitInfo.medium);

          accumulatedRayColor += throughput * lightValue * diffuseEval(-rayDirection, wo) * hitInfo.color;

          rayDirection = normalize(wo);
          rayOrigin = hitInfo.worldPosition + rayDirection * 0.0001;
        }
      }
      else
      {
        // Ray did not hit anything
        break;
      }
      payload.depth++;
      if (payload.depth > 64)
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


  float exposure = 1.0;
  vec3 hdrColor = summedPixelColor / float(NUM_SAMPLES);

  //Reinhard toneamp
  vec3 mappedColor = hdrColor / (hdrColor + vec3(1.0));
  mappedColor = pow(mappedColor, vec3(1.0 / exposure));

  imageStore(storageImage, pixel, vec4(mappedColor, 0.0));
}
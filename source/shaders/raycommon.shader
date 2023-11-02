struct hitPayload
{
  vec3 directLight;
  vec3 nextRayOrigin;
  vec3 nextRayDirection;
  vec3 nextFactor;
  bool shadowRayMiss;
  uint rngState;
  int level;
};

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
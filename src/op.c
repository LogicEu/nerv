
/*********************************************
 *   floating point functions and operations
 * ******************************************/

#include <nerv.h>
#include <math.h>

unsigned int ftou(float n)
{
    return (int)(n * 255.0);
}

float utof(unsigned int u)
{
    return (float)u / 255.0;
}

float clampf(float x, float min, float max)
{
    return x * (min <= x && x <= max) + max * (x > max) + min * (x < min);
}

float normf(float x)
{
    return clampf(x, 0.0, 1.0);
}

float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

float sigderiv(float sig)
{
    return sig * (1.0f - sig);
}

float dsigmoid(float x)
{
    return sigderiv(sigmoid(x));
}

float relu(float x)
{
    return x * (x > 0.0f);
}

float drelu(float x)
{
    return (float)(x > 0.0f);
}

float leaky_relu(float x, float slope)
{
    return x * (x >= 0.0f) + slope * (x < 0.0f);
}

float dleaky_relu(float x, float slope)
{
    return (float)(x >= 0.0f) + slope * (x < 0.0f);
}

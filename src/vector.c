
/*********************************************
 *      vector creation and management
 * ******************************************/

#include <nerv.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

Vec vector(int size)
{
    Vec v = {
        v.size = size,
        v.data = (float*)calloc(size, sizeof(float))
    };  return v;
}

Vec vector_create(int size, ...)
{
    Vec v = vector(size);
    float *f = v.data;
    
    va_list args;
    va_start(args, size);
    for (float* end = f + size; f != end; f++) {
        float n = va_arg(args, double);
        *f = n;
    }   
    va_end(args);
    
    return v;
}

Vec vector_uniform(int size, float val)
{
    Vec v = vector(size);
    float* f = v.data;
    
    for (float* end = f + v.size; f != end; f++) {
        *f = val;
    }   
    
    return v;
}

Vec vector_copy(const Vec* restrict src)
{
    Vec v = vector(src->size);
    memcpy(v.data, src->data, v.size * sizeof(float));
    return v;
}

void vector_free(Vec* vec)
{
    free(vec->data);
}

/*********************************************
 *          fully connected layers 
 * ******************************************/

#include <nerv.h>
#include <stdlib.h>

Layer layer_create(int layer_size, int next_layer_size)
{
    Layer layer;
    layer.w.columns = 0;
    layer.w.rows = 0;
    layer.w.data = NULL;

    layer.a = vector(layer_size);
    layer.b = vector(layer_size);
    layer.z = vector(layer_size);
    layer.d = vector(layer_size);
    if (!next_layer_size) return layer;

    layer.w = matrix(next_layer_size, layer_size);
    return layer;
}

Layer layer_copy(const Layer* restrict layer)
{
    Layer ret;
    ret.w.columns = 0;
    ret.w.rows = 0;
    ret.w.data = NULL;

    ret.a = vector_copy(&layer->a);
    ret.b = vector_copy(&layer->b);
    ret.z = vector_copy(&layer->z);
    ret.d = vector_copy(&layer->d);
    if (!layer->w.data) return ret;

    ret.w = matrix_copy(&layer->w);
    return ret;
}

void layer_matrix_free(Layer* layer)
{
    matrix_free(&layer->w);
}

void layer_vector_free(Layer* layer)
{
    vector_free(&layer->a);
    vector_free(&layer->b);
    vector_free(&layer->z);
    vector_free(&layer->d);
}

void layer_free(Layer* layer)
{
    layer_matrix_free(layer);
    layer_vector_free(layer);
}
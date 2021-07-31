
/*********************************************
 *    neural network model data structure 
 * ******************************************/

#include <nerv.h>
#include <stdarg.h>
#include <stdlib.h>

Model model_new(int layer_count)
{
    Model model = {
        layer_count,
        (Layer*)malloc(sizeof(Layer) * layer_count)
    };  return model;
}

Model model_create(int layer_count, ...)
{   
    int neuron_counts[layer_count];

    va_list args;
    va_start(args, layer_count);
    for (int i = 0; i < layer_count; i++) {
        neuron_counts[i] = va_arg(args, int);
    }
    va_end(args);

    Model model = model_new(layer_count);
    Layer* layer = model.layers;

    for (int i = 0; i < layer_count; i++) {
        int next = 0;
        if (i < layer_count - 1) next = neuron_counts[i + 1];
        *(layer++) = layer_create(neuron_counts[i], next);
    } 

    return model;
}

Model model_copy(Model* model)
{
    Model ret = model_new(model->layer_count);
    Layer* layer = ret.layers, *l = model->layers;
    
    for (int i = 0; i < ret.layer_count; i++) {
        *(layer++) = layer_copy(l++);
    } 

    return ret;
}

void model_free(Model* model)
{
    Layer* layer = model->layers;
    for (int i = 0; i < model->layer_count - 1; i++) {
        layer_free(layer);
        layer++;
    }

    layer_vector_free(layer);
    free(model->layers);
}
#include <nerv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Model model_load(char* path)
{
    Model model = {0, NULL};

    FILE* file = fopen(path, "rb");
    if (!file) {
        printf("Could not read nerv model file '%s'\n", path);
        return model;
    }

    fread(&model, sizeof(Model), 1, file);
    model.layers = (Layer*)malloc(sizeof(Layer) * model.layer_count);

    int layer_sizes[model.layer_count + 1];
    memset(&layer_sizes[0], 0, sizeof(int) * (model.layer_count + 1));
    fread(&layer_sizes[0], sizeof(int), model.layer_count, file);

    Layer* layer = model.layers;
    for (int i = 0; i < model.layer_count; i++) {
        int size = layer_sizes[i], next = layer_sizes[i + 1];
        *layer = layer_create(size, next);
        
        fread(layer->a.data, sizeof(float), layer->a.size, file);
        
        if (!next) break;
        fread(layer->w.data, sizeof(float), layer->w.rows * layer->w.columns, file);
        
        layer++;
    }
    
    fclose(file);
    printf("Succesfully loaded nerv model file '%s'\n", path);
    model_print_struct(&model);
    return model;
}

void model_save(char* path, Model* model)
{
    FILE* file = fopen(path, "wb");
    if (!file) {
        printf("Could not write nerv model file '%s'\n", path);
        return;
    }

    fwrite(model, sizeof(Model), 1, file);

    Layer* layer = model->layers;
    for (Layer* end = layer + model->layer_count; layer != end; layer++) {
        fwrite(&layer->a.size, sizeof(int), 1, file);
    }

    layer = model->layers;
    for (Layer* end = layer + model->layer_count; layer != end; layer++) {
        fwrite(layer->a.data, sizeof(float), layer->a.size, file);
        
        if (layer + 1 == end) break;
        fwrite(layer->w.data, sizeof(float), layer->w.rows * layer->w.columns, file);
    }

    fclose(file);
    printf("Succesfully saved nerv file '%s'\n", path);
}
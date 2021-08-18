#ifndef NERV_H
#define NERV_H

#ifdef __cplusplus
extern "C" {
#endif

/******************************
 * nerv tiny C tensor framework
 * ----------------------------
 ****************** @eulogic */

typedef struct Vec {
    int size;
    float* data;
} Vec;

typedef struct Mat {
    int rows, columns;
    float* data;
} Mat;

typedef struct {
    Mat w;
    Vec b, z, a, d;
} Layer;

typedef struct {
    int layer_count;
    Layer* layers;
} Model;

#define _ftou(n) (unsigned int)(int)((n) * 255)
#define _utof(u) ((float)(u) / 255.0f)
#define __clampf(x, min, max) (x * (min <= x && x <= max) + max * (x > max) + min * (x < min))
#define _clampf(x, min, max) __clampf((x), (min), (max)) 
#define _normf(x) _clampf(x, 0.0, 1.0)
#define _sigmoid(x) (1.0 / (1.0 + exp(-(x))))
#define _sigderiv(x) ((x) * (1.0 - (x)))
#define _dsigmoid(x) _sigderiv(_sigmoid(x))
#define _relu(x) ((x) * ((x) > 0.0f))
#define _drelu(x) (float)((x) > 0.0f)
#define __leaky_relu(x, slope) (x * (x >= 0.0f) + slope * (x < 0.0f))
#define _leaky_relu(x, slope) __leaky_relu((x), (slope))
#define __dleaky_relu(x, slope) (float)(x >= 0.0f) + slope * (x < 0.0f)
#define _dleaky_relu(x, slope) __dleaky_relu((x), (slope))

/*------------------------------------------*/

/*    PSEUDO RANDOM GENERATOR AND MATH      */

/*********************************************
 * pseudo random number distribution generator 
 * ******************************************/

void rand_seed(unsigned int seed);
unsigned int rand_uint();
double rand_gauss();
double rand_norm();
double rand_dist(double standard_deviation, double mean);

/*********************************************
 *   floating point functions and operations
 * ******************************************/

unsigned int ftou(float n);
float utof(unsigned int u);
float clampf(float x, float min, float max);
float normf(float x);
float sigmoid(float x);
float sigderiv(float sig);
float dsigmoid(float x);
float relu(float x);
float drelu(float x);
float leaky_relu(float x, float slope);
float dleaky_relu(float x, float slope);

/*------------------------------------------*/

/*  VECTOR DATA STRUCTURE AND OPERATIONS    */

/*********************************************
 *      vector creation and management
 * ******************************************/

Vec vector(int size);
Vec vector_create(int size, ...);
Vec vector_uniform(int size, float val);
Vec vector_copy(const Vec* src);
void vector_free(Vec* vector);

/*********************************************
 *     vector functions and operations
 * ******************************************/

void vector_scale(const Vec* v, float n);
void vector_add(const Vec* dst, const Vec* src);
void vector_sub(const Vec* dst, const Vec* src);
void vector_hadamard(const Vec* dst, const Vec* src);

Vec vector_by_matrix(const Mat* mat, const Vec* vec);
Vec vector_by_matrix_transposed(const Mat* mat, const Vec* vec);
Vec vector_sigmoid(const Vec* v);
Vec vector_dsigmoid(const Vec* v);
Vec vector_sigderiv(const Vec* v);
Vec vector_relu(const Vec* v);
Vec vector_drelu(const Vec* v);
Vec vector_leaky_relu(const Vec* v, float leak);
Vec vector_dleaky_relu(const Vec* v, float leak);
Vec vector_softmax(const Vec* v);

/*------------------------------------------*/

/*   MATRIX DATA STRUCTURE AND OPERATIONS   */

/*********************************************
 *       matrix creation and management
 * ******************************************/

Mat matrix(int rows, int columns);
Mat matrix_create(int rows, int columns, ...);
Mat matrix_identity(int size);
Mat matrix_uniform(int rows, int columns, float val);
Mat matrix_copy(const Mat* mat);
Mat matrix_vector(const Vec* v);
void matrix_free(Mat* mat);

/*********************************************
 *       matrix functions and operations
 * ******************************************/

Mat matrix_scale(const Mat* mat, float scale);
Mat matrix_multiply(const Mat* a, const Mat* b);
Mat matrix_hadamard(const Mat* a, const Mat* b);
Mat matrix_transpose(const Mat* m);

/*------------------------------------------*/

/*   NEURAL NETWORK AND LAYER STRUCTURES    */

//      z = w * a + b
//      a = f(z)
//      c = (a - y)^2
//      d = 2(a - y) * f'(z)
//      dw = a * d * g
//      db = d * g

/*********************************************
 *          fully connected layers 
 * ******************************************/

Layer layer_create(int layer_size, int next_layer_size);
Layer layer_copy(const Layer* layer);
void layer_vector_free(Layer* layer);
void layer_matrix_free(Layer* layer);
void layer_free(Layer* layer);

/*********************************************
 *    neural network model data structure 
 * ******************************************/

Model model_new(int layer_count);
Model model_create(int layer_count, ...);
Model model_copy(const Model* model);
void model_free(Model* model);

/*********************************************
 *    neural network model operations 
 * ******************************************/

void model_init(const Model* model);
void model_forward(const Model* model);
void model_backwards(const Model* model, const Vec* desired_output);
void model_update(const Model* model, float alpha);
float model_cost(const Model* model, const Vec* desired_output);

/*------------------------------------------*/

/*                 NERV IO                  */

/*********************************************
 *     serialize models - save and load
 * ******************************************/

Model model_load(char* path);
void model_save(char* path, const Model* model);

/*********************************************
 *   useful IO functions to print and scan
 * ******************************************/

Vec vector_scan();
void vector_print(const Vec* vec);

Mat matrix_scan();
void matrix_print(const Mat* mat);

Model model_scan();
void model_print(const Model* nm);
void model_print_input(const Model* nm);
void model_print_output(const Model* nm);
void model_print_struct(const Model* nm);

/*------------------------------------------*/

#ifdef __cplusplus
}
#endif
#endif

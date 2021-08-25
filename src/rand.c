
/*********************************************
 * pseudo random number distribution generator 
 * ******************************************/

#include <nerv.h>
#include <limits.h>
#include <math.h>

#define TWO_PI 6.28318530718

static unsigned int pseudo_random_seed = 0;

static unsigned int rand_seeded(unsigned int num)
{
    num = (num << 13) ^ num;
    return ((num * (num * num * 15731 + 789221) + 1376312589) & 0x7fffffff);
}

#define _randn() rand_seeded(pseudo_random_seed++)

unsigned int randn() 
{
    return rand_seeded(pseudo_random_seed++);
}

void rands(unsigned int seed) 
{
    pseudo_random_seed = seed;
}

double rand_norm()
{
    return (double)_randn() / (double)INT_MAX;
}

double rand_gauss()
{
	static double U, V;
	static int phase = 0;
	double Z;

	if (phase == 0) {
		U = (_randn() + 1.0) / (INT_MAX + 2.0);
		V = _randn() / (INT_MAX + 1.0);
		Z = sqrt(-2 * log(U)) * sin(TWO_PI * V);
	} else {
		Z = sqrt(-2 * log(U)) * cos(TWO_PI * V);
    }

	phase = 1 - phase;
	return Z;
}

double rand_dist(double standard_deviation, double mean)
{
    return rand_gauss() * standard_deviation + mean;
}
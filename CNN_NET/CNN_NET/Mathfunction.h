#ifndef __CNN_NET__Mathfunction__
#define __CNN_NET__Mathfunction__

#include <iostream>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "out_map.cpp"

using namespace std;

void convn(double **bottom, int bottom_h, int bottom_w, double **top,
           int h, int w, double **kernel, int sz, int stride, bool full);

void corre(double **bottom, int bottom_h, int bottom_w, double **top,
           int h, int w, double **kernel, int sz, int stride, bool full);

double linear(double **bottom, double **kernel, int h, int w);


#endif /* defined(__CNN_NET__Mathfunction__) */


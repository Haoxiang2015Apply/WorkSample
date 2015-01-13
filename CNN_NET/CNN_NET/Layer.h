#ifndef CNN_NET_Layer_h
#define CNN_NET_Layer_h
#include <iostream>
#include <cstdio>
#include <algorithm>
#define M 100
#include "out_map.cpp"

using namespace std;


struct Layer{
    int output_maps;
    int kernel_size;
    string type;
    
    Map *out_map;
    Kernel *kernel[M];
    
    Kernel pool_kernel;
    Layer(){}
    
    
    Layer(int size, string tp, int _kernel_size = 0){
        type = tp;    
        out_map = new Map[size];
        output_maps = size;
        kernel_size = _kernel_size;
    }
    
    
};


#endif

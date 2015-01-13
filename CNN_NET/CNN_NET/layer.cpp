
#include <iostream>
#include <cstdio>
#include <algorithm>
#define M 100
#include "out_map.cpp"

using namespace std;


struct Layer1{
    int output_maps;
    int kernel_size;
    string type;

    Map *out_map;
    Kernel *kernel[M];
    //Kernel *kernel;
    Layer1(){}
    

    Layer1(int size, string tp, int _kernel_size = 0){
        //cout<<"before"<<endl;
        out_map = new Map[size];
        //cout<<"after"<<endl;
        
        type = tp;
        output_maps = size;
        kernel_size = _kernel_size;
    }

};




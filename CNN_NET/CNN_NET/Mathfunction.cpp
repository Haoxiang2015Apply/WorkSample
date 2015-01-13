//
//  Mathfunction.cpp
//  CNN_NET
//
//  Created by imac41 on 14-7-20.
//  Copyright (c) 2014年 imac41. All rights reserved.
//

#include "Mathfunction.h"

void convn(double **bottom, int bottom_h, int bottom_w, double **top,
           int h, int w, double **kernel, int sz, int stride, bool full){
    
    //int sz = kernel.size;
    
    //int h = top.h, w = top.w;
    
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            double ret = 0.0;
            int x = 0, y = 0;
            if(full){
                for(int p = i; p > i - sz; p--){
                    y = 0;
                    for(int q = j; q > j - sz; q--){
                        if(p < 0 || p >= bottom_h || q < 0 || q >= bottom_w){
                            y++;
                            continue;
                        }
                        ret += bottom[p][q] * kernel[x][y++];
                    }
                    x++;
                }
                
            }
            else{
                
                for(int p = i * stride; p < i * stride + sz; p++){
                    y = 0;
                    for(int q = j * stride; q < j * stride + sz; q++)
                        ret += bottom[p][q] * kernel[sz-1-x][sz-1-y++];
                    x++;
                }
            }
            
            top[i][j] += ret;
        }
    }
}

void corre(double **bottom, int bottom_h, int bottom_w, double **top,
           int h, int w, double **kernel, int sz, int stride, bool full){
    
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            double ret = 0.0;
            int x = 0, y = 0;
            if(full){
                for(int p = i; p > i - sz; p--){
                    y = 0;
                    for(int q = j; q > j - sz; q--){
                        if(p < 0 || p >= bottom_h || q < 0 || q >= bottom_w){
                            y++;
                            continue;
                        }
                        ret += bottom[p][q] * kernel[sz-1-x][sz-1-y++];
                    }
                    x++;
                }
                
            }
            else{
                
                for(int p = i * stride; p < i * stride + sz; p++){
                    y = 0;
                    for(int q = j * stride; q < j * stride + sz; q++)
                        ret += bottom[p][q] * kernel[x][y++];
                    x++;
                }
            }
            
            top[i][j] += ret;
        }
    }
}

double linear(double **bottom, double **kernel, int h, int w){
    
    double ret = 0.0;
    for(int i = 0; i < h; i++){
        
        for(int j = 0; j < w; j++)
            ret += bottom[i][j] * kernel[i][j];
    }
    
    return ret;
    
}

#ifndef out_map_cpp
#define out_map_cpp
#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <algorithm>
#define N 500
using namespace std;
inline double Sigmod(double x) {return 1.0/(1 + exp(-x));}


struct Map{

private:
	int height;
	int width;

public:
	double *a[N];
	double *d[N];
    double bias;
    double db;
	Map(){}
	~Map(){}

    Map(int h, int w){
		height = h; width = w;
		for(int i = 0; i < h; i++){
			a[i] = new double[w];
			d[i] = new double[w];
			for(int j = 0; j < w; j++)
				a[i][j] = d[i][j] = 0;
		}
        bias = db = 0.0;
	}
    
    void release(){
        for(int i = 0; i < height; i++){
            delete d[i];
        }
    }
    
    void read_input(double **data, int id, int im_size){
        for(int i = 0; i < im_size; i++){
            for(int j = 0; j < im_size; j++){
                   a[i][j] = data[id][i * im_size + j];
            }
        }
        
        
    }
    
    void set(double **re){
        if(re == NULL){
            for(int i = 0; i < height; i++)
                memset(a[i], 0, sizeof(double) * width);
        }
    }
    
    void set_d(double **re){
        if(re == NULL){
            for(int i = 0; i < height; i++)
                memset(d[i], 0, sizeof(double) * width);
        }
    }
    
    double get_sum(){
        
        double ret = 0.0;
        for(int i = 0; i < height; i++){
            
            for(int j = 0; j < width; j++)
                ret += d[i][j];
        }
        return ret;
    }
    
    void sigmod(){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++)
                a[i][j] = Sigmod(a[i][j]);
        }
    }

    void add(double bias){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++)
                a[i][j] += bias;   
        }
    }
    
    void relu(){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++)
                a[i][j] = max(a[i][j], 0.0);
        }
    }
    void up(double **top){
        for(int i = 0; i < height; i += 2){
            for(int j = 0; j < width; j += 2){
                
                for(int p = i; p < i + 2; p++){
                    for(int q = j; q < j + 2; q++)
                        //d[p][q] = top[i/2][j/2] * 1/4 * (1 - a[p][q]) * a[p][q];
						d[p][q] = top[i/2][j/2] * 1/4 * (a[p][q] > 1e-15);

                }
            }
        }
    }
    
	int h(){return height;}
	int w(){return width;}

};


struct Kernel{

private:
	int Size;
	int Stride;
public:
	double *k[N], *dk[N];
	double b, db;
	Kernel(){}
	~Kernel(){}

	Kernel(int _size, int _stride, bool ispool){
		Size = _size;
        Stride = _stride;
        
		b = double(rand()%1000 - 500)/1000;
		for(int i = 0; i < _size; i++){
			k[i] = new double[_size];
            dk[i] = new double[_size];
            if(ispool){
                for(int j = 0; j < _size; j++)
                    k[i][j] = 0.25;
                continue;
            }
            
			for(int j = 0; j < _size; j++)
				k[i][j] = double(rand()%1000 - 500)/1000;
		}

	}
    
    void release(){
        
        for(int i = 0; i < Size; i++)
            delete dk[i];
    }

    
    void set_dk(double **re){
        
        if(re == NULL){
            for(int i = 0; i < Size; i++)
                memset(dk[i], 0, sizeof(double) * Size);
        }
        
    }
    
    void set_k(double **re){
        
        for(int i = 0; i < Size; i++){
            for(int j = 0; j < Size; j++)
                k[i][j] = re[i][j];
        }
        
    }
    
    
    void update_grad(double alpha){
        
        for(int i = 0; i < Size; i++){
            
            for(int j = 0; j < Size; j++)
                k[i][j] = k[i][j] + dk[i][j] * alpha;
        }
    }
    
	int size(){return Size;}
	int stride(){return Stride;}

};

#endif



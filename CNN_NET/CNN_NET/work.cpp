#include <iostream>
#include "test.h"
#include "Layer.h"
//#include <boost/thread.hpp>

#include "Mathfunction.h"

#define alpha -0.03
#define layer_num 6
#define im_size 28
using namespace std;

Layer *layer;

void cnn_setup(){
    
    layer = new Layer[10];
    srand(unsigned(time(NULL)));
    
    layer[0] = Layer(1, "in");
    layer[1] = Layer(6, "convn", 5);
    layer[2] = Layer(6, "pool", 2);
    layer[3] = Layer(12, "convn", 5);
    layer[4] = Layer(12, "pool", 2);
    layer[5] = Layer(10, "full", 4);
    
    
    int mapsize = 28, inputmaps = 1;

    //init
    for(int l = 0; l < layer_num; l++){
        int kernel_sz = layer[l].kernel_size;

        if(layer[l].type == "pool"){
            mapsize /= 2;
            layer[l].pool_kernel = Kernel(kernel_sz, 2, true);
            
            for(int i = 0; i < inputmaps; i++)
                layer[l].out_map[i] = Map(mapsize, mapsize);
        }
        
        else if(layer[l].type == "convn" || layer[l].type == "full"){
            int outputmaps = layer[l].output_maps;
            
            if(layer[l].type == "convn")
               mapsize = mapsize - kernel_sz + 1;
            else
               mapsize = 1;
            for(int i = 0; i < inputmaps; i++){
                layer[l].kernel[i] = new Kernel[outputmaps];
                for(int j = 0; j < outputmaps; j++)
                    layer[l].kernel[i][j] = Kernel(kernel_sz, 1, false);
            }
     
            for(int i = 0; i < outputmaps; i++){
                layer[l].out_map[i] = Map(mapsize, mapsize);
            }
            inputmaps =  outputmaps;
        }
        else{
            for(int i = 0; i < inputmaps; i++){
                layer[l].out_map[i] = Map(mapsize, mapsize);
            }
        }
        
    }

}


void cnn_ff(int id){
    
    //read data

    
    layer[0].out_map[0].read_input(data, id, im_size);
    
    for(int l = 1; l < layer_num; l++){
        
        if(layer[l].type == "convn"){
            for(int j = 0; j < layer[l].output_maps; j++){
                
                int h = layer[l].out_map[j].h(), w = layer[l].out_map[j].w();

                layer[l].out_map[j].set(NULL);
                
                double **top = layer[l].out_map[j].a;


                for(int i = 0; i < layer[l-1].output_maps; i++){
                    
                    int h1 = layer[l-1].out_map[i].h(), w1 = layer[l-1].out_map[i].w();
                    
                    
                    double **bottom = layer[l-1].out_map[i].a;

                    
                    int kernel_sz = layer[l].kernel[i][j].size();
                    
                    
                    double **kernel = layer[l].kernel[i][j].k;

                    corre(bottom, h1, w1, top, h, w, kernel, kernel_sz, 1, 0);
                    
                }

                layer[l].out_map[j].add(layer[l].out_map[j].bias);
                //layer[l].out_map[j].sigmod();
				layer[l].out_map[j].relu();
                
            }
        }
        else if(layer[l].type == "pool"){
            for(int i = 0; i < layer[l].output_maps; i++){
                
                int sz = layer[l].pool_kernel.size();
                
                
                double **kernel = layer[l].pool_kernel.k;
                
                
                int bottom_h = layer[l-1].out_map[i].h(), bottom_w = layer[l-1].out_map[i].w();
                
                
                double **bottom = layer[l-1].out_map[i].a;

                layer[l].out_map[i].set(NULL);
                int top_h = layer[l].out_map[i].h(), top_w = layer[l].out_map[i].w();
                
                double **top = layer[l].out_map[i].a;
                
                corre(bottom, bottom_h, bottom_w, top, top_h, top_w, kernel, sz, 2, 0);
                
                
            }
            
        }
        else{
            double Max = 0.0;
            double *tmp = new double[N];

            for(int j = 0; j < layer[l].output_maps; j++){
                double ret = 0.0;
                for(int i = 0; i < layer[l-1].output_maps; i++){
                    int bottom_h = layer[l-1].out_map[i].h(), bottom_w = layer[l-1].out_map[i].w();
                    
                    double **bottom = layer[l-1].out_map[i].a;
                    
                    double **kernel = layer[l].kernel[i][j].k;
                    ret += linear(bottom, kernel, bottom_h, bottom_w);
                }
                ret += layer[l].out_map[j].bias;
                tmp[j] = ret;
                if(ret > Max)
                   Max = ret;
            }

            double All = 0.0;
            for(int j = 0; j < layer[l].output_maps; j++){
                tmp[j] -= Max;
                All += exp(tmp[j]);
            }

            for(int j = 0; j < layer[l].output_maps; j++){
                layer[l].out_map[j].a[0][0] = exp(tmp[j])/All;
            }
            delete tmp;
        }
    }
    
}

double cost(int id){
	double ret = 0.0;
	for(int i = 0; i < 10; i++){
		if(id == i)
		   ret = -log(layer[layer_num-1].out_map[i].a[0][0]);
	}
	return ret;
}

void cnn_bp(int id){
    //last layer
    
    int l = layer_num - 1;
    
    double error = 0.0, out = 0.0;
    for(int i = 0; i < layer[l].output_maps; i++){
        out = layer[l].out_map[i].a[0][0];
        if(id == i)
            error = out - 1;
        else
            error = out;
        layer[l].out_map[i].d[0][0] = error;
    }
    
    //last second layer
    l = layer_num - 2;

    for(int i = 0; i < layer[l].output_maps; i++){
        
        for(int W = 0; W < layer[l].out_map[i].w(); W++){
            for(int H = 0; H < layer[l].out_map[i].h(); H++){
                double ret = 0.0;
                out = layer[l].out_map[i].a[W][H];
                for(int j = 0; j < layer[l + 1].output_maps; j++)
                    ret += layer[l+1].kernel[i][j].k[W][H] * layer[l+1].out_map[j].d[0][0];
                layer[l].out_map[i].d[W][H] = ret;
                
            }
        }
        
        
    }
    
    
    for(int l = layer_num - 3; l >= 1; l--){
        
        if(layer[l].type == "convn"){
            for(int i = 0; i < layer[l].output_maps; i++){
                double **top = layer[l+1].out_map[i].d;
                
                layer[l].out_map[i].up(top);
                
            }
            
        }
        else if(layer[l].type == "pool"){
            for(int i = 0; i < layer[l].output_maps; i++){
                
				int bottom_h = layer[l].out_map[i].h(), bottom_w = layer[l].out_map[i].w();
                
                layer[l].out_map[i].set_d(NULL);

                double **bottom = layer[l].out_map[i].d;
                
				for(int j = 0; j < layer[l+1].output_maps; j++){
                    
					int top_h = layer[l+1].out_map[j].h(), top_w = layer[l+1].out_map[j].w();
                    
                    double **top = layer[l+1].out_map[j].d;

					int kernel_sz = layer[l+1].kernel[i][j].size();
                    
                    double **kernel = layer[l+1].kernel[i][j].k;
                    
					convn(top, top_h, top_w, bottom, bottom_h, bottom_w, kernel, kernel_sz, 1, 1);
                    
				}
				
			}
            
            
        }
    }
    
    //update last layer
    
    l = layer_num - 1;
    
    for(int j = 0; j < layer[l].output_maps; ++j){
        
        //double ret = 0.0;
        
        for(int i = 0; i < layer[l-1].output_maps; i++){
            
            for(int p = 0; p < layer[l].kernel[i][j].size(); p++){
                for(int q = 0; q < layer[l].kernel[i][j].size(); q++)
                    layer[l].kernel[i][j].dk[p][q] = layer[l].out_map[j].d[0][0] * layer[l-1].out_map[i].a[p][q];
            }
            
        }
        layer[l].out_map[j].db = layer[l].out_map[j].d[0][0];
        
        
    }
    
    //update for k and bias
    
	for(l = layer_num - 2; l > 0; --l){
		if(layer[l].type == "convn"){
            
			for(int j = 0; j < layer[l].output_maps; j++){
                
                int sz = layer[l].out_map[j].w();
				
                double **top = layer[l].out_map[j].d;

                layer[l].out_map[j].db = layer[l].out_map[j].get_sum();
                
				for(int i = 0; i < layer[l-1].output_maps; i++){
                    
					int w = layer[l-1].out_map[i].w(), h = layer[l-1].out_map[i].h();
                    
					
                    double **bottom = layer[l-1].out_map[i].a;

					int kernel_sz = layer[l].kernel[i][j].size();
                    double **ret = layer[l].kernel[i][j].dk;
                    
					
                    layer[l].kernel[i][j].set_dk(NULL);
                    ret = layer[l].kernel[i][j].dk;
                    corre(bottom, w, h, ret, kernel_sz, kernel_sz, top, sz, 1, 0);
					
				}
                
            }
        }
        
        
	}
    
}


void apply_grad(){
    //bool f = true;
    for(int l = 1; l < layer_num; l++){
        
        if(layer[l].type == "convn" || layer[l].type == "full"){
            for(int j = 0; j < layer[l].output_maps; j++){
            
                for(int i = 0; i < layer[l-1].output_maps; i++){
                    
                    layer[l].kernel[i][j].update_grad(alpha);
                }

                layer[l].out_map[j].bias += layer[l].out_map[j].db * alpha;
            }
        }
    }
    
}

void check_grad(int id){
    double eps = 1e-6;
    double f1, f2;
    f1 = cost(id);
    
    
    for(int l = 1; l < layer_num; l++){
        
        
        for(int j = 0; j < layer[l].output_maps; j++){
            
            if(layer[l].type == "full" || layer[l].type == "convn"){
                layer[l].out_map[j].bias += eps;
                
                cnn_ff(1);
                f2 = cost(id);
                layer[l].out_map[j].bias -= 2*eps;
                cnn_ff(1);
                f1 = cost(id);
                double delta = (f2 - f1)/(2 * eps);
                if(abs(layer[l].out_map[j].db - delta) > eps){
                   cout<<"find a bug in bias"<<endl;
             
                   cout<<layer[l].out_map[j].db<<" "<<delta<<endl;
				}
                layer[l].out_map[j].bias += eps;
            }
            
            
            for(int i = 0; i < layer[l-1].output_maps; i++){
                
                if(layer[l].type == "full" || layer[l].type == "convn"){
                
                    int sz = layer[l].kernel[i][j].size();
                    for(int p = 0; p < sz; p++){
                        for(int q = 0; q < sz; q++){
                            layer[l].kernel[i][j].k[p][q] += eps;
                            
                            cnn_ff(1);
                            f1 = cost(id);
                            layer[l].kernel[i][j].k[p][q] -= 2*eps;
                            //cout<<f1<<" "<<f2<<" "<<f1 - f2<<endl;
                            cnn_ff(1);
                            f2 = cost(id);
                            
                            double delta = (f1 - f2)/ (2*eps);
                            //cout<<delta<<" "<<layer[l].kernel[i][j].dk[p][q]<<endl;
                            if(abs(delta - layer[l].kernel[i][j].dk[p][q]) > eps){
                                cout<<"find a bug"<<endl;
								cout<<delta<<" "<<layer[l].kernel[i][j].dk[p][q]<<endl;
							}
                            
                            layer[l].kernel[i][j].k[p][q] += eps;
                        }
                    }
                    
                }
                
            }
            
        }
        
        
    }
}

void cnn_test(){
    double ret = 0.0;
    for(int i = 0; i < 1000; i++){
        cnn_ff(i);
        cnn_bp((int)y[i]);
        ret += cost((int)y[i]);
        apply_grad();
    }
    cout<<ret/1000<<endl;
    
}

int predict(){
    
    double ret = 0.0;
    int id = 0;
    for(int i = 0; i < 10; i++){
        double result = layer[5].out_map[i].a[0][0];
        if(ret < result){
            ret = result;
            id = i;
        }
        
    }
    return id;
}

void finish(){
    
    for(int i = 0; i < layer_num; i++){
        if(layer[i].type == "convn" || layer[i].type == "full"){
            for(int j = 0; j < layer[i].output_maps; j++){
                layer[i].out_map[j].release();
                
                for(int k = 0; k < layer[i-1].output_maps; k++)
                    layer[i].kernel[k][j].release();
            }
        }
    }
}

int main()
{
    cnn_setup();
    
    cout<<"begin read data\n"<<endl;
    read();
    cout<<"end read data\n"<<endl;
    
    
    cnn_ff(1);
    cnn_bp((int)y[1]);
    check_grad((int)y[1]);
    cout<<"check end"<<endl;

	for(int i = 0; i < 30; i++)
		cnn_test();

	system("pause");
	return 0;

}






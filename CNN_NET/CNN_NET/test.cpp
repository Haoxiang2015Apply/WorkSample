#include "test.h"
#define input_num 20000
#define image_size 784

double *data[MAX];
double *y;


void work_str(string s, int n, bool f){
    string str;
    int k = 0;
    
    int l = (int)s.size();

    
    for(int i = 0; i < l; i++){
        if(s[i] == ','){
            //cout<<str<<endl;
            if(f){
                //cout<<str<<endl;
                y[n] = atoi(str.c_str());
                //cout<<"label is "<<y[n]<<endl;
                f = false;
            }
            else{
                data[n][k++] = atoi(str.c_str())/255.0;
            }
            str = "";
        }
        else{
            str += s[i];
        }
    }
    data[n][k++] = atoi(str.c_str())/255.0;
}



void read(){
    
    //input contains 20000 graphs, each graph is 32 * 32(1024) pixels
    y = new double[input_num];
    
    
    for(int i = 0; i < input_num; i++){
        data[i] = new double[image_size];
    }

    fstream fin("train.csv", ios::in);

	if(fin.fail())
	   cout<<"fail"<<endl;
    string tmp;
    getline(fin, tmp);
    
    
    int cnt = 0;
    while(getline(fin, tmp)){
        work_str(tmp, cnt, true);
        cnt++;
		if(cnt == 1000)
		   break;
    }

    //cout<<"cnt is "<<cnt<<endl;

    
    fin.close();
    
}

void write(){}





#ifndef __CNN_NET__test__
#define __CNN_NET__test__

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#define MAX 50000
using namespace std;

extern double *data[MAX];
extern double *y;


void work_str(string s, int n, bool f);
void read();
void write();


#endif /* defined(__CNN_NET__test__) */

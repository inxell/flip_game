#include<iostream>
#include<fstream>
#include<sstream>
#include<new>
#include<string>
#include<algorithm>
#include<map>
#include<set>
#include<unordered_map>
#include<unordered_set>
#include<vector>
#include<deque>
#include<random>
#include<ctime>

using namespace std;

string dir_path = "/home/ke/Documents/";
const int LEN = 25;

short nim(int k, const vector<short>& svec){
    bool prev = k&1, boo;
    int bit = 1;
    unordered_set<short> next_nims;
    for(int i = 1; i < LEN; ++i){
        bit <<= 1;
        boo = k&bit;
        if(prev && boo){
            next_nims.insert(svec[k - bit - bit/2]);
        }
        prev = boo;
    }
    for(short j = 0; j < LEN; ++j){
        if(next_nims.find(j) == next_nims.end()){
            return j;
        }
    }
}

short nim2D(int k, const vector<short>& svec){
    unordered_set<short> next_nims;
    // horizontal adjacent on bits
    for(int i = 0; i < 5; ++i){
        int bit = 1 << 5*i;
        bool prev = k & bit, boo;
        for(int j = 1; j < 5; ++j){
            bit <<= 1;
            boo = k & bit;
            if(prev && boo){
                next_nims.insert(svec[k - bit - bit/2]);
            }
            prev = boo;
        }
    }
    // vertical adjacent bits
    for(int i = 0; i < 4; ++i){
        int upper = 1 << 5*i, lower = 1 << 5*(i+1);
        for(int j = 0; j < 5; ++j){
            if((k & upper) && (k & lower)){
                next_nims.insert(svec[k - upper - lower]);
            }
            upper <<= 1;
            lower <<= 1;
        }
    }
    for(short j = 0; j < LEN; ++j){
        if(next_nims.find(j) == next_nims.end()){
            return j;
        }
    }
}


int main(){
    int total = 1 << LEN;
    string filename = "2D_len" + to_string(LEN) + ".txt";
    ofstream f(dir_path + filename);
    for(int i = 0; i < 3; ++i){
        f << 0 << '\n';
    }
    vector<short> res(total, 0);
    cout << "Start!" << endl;
    time_t starttime = time(nullptr);
    for(int i = 3; i < total; ++i){
        res[i] = nim2D(i, res);
        f << res[i] << '\n';
    }
    cout << "Finish in " << (time(nullptr) - starttime)/60 << " minutes.";

    return 0;
};
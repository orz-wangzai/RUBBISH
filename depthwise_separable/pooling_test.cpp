#define AP_INT_MAX_W 8192

#include <stdint.h>
#include <ap_int.h>
#include "pool2d.h"
#include "function.h"
#include "stream_tools.h"
#include "conv2d.h"

#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"



#define K      7
#define IN_ROW 7
#define IN_COL 7
#define IN_CH  1280
#define IN_BIT 2

int main(int argc, char const *argv[]){


	hls::stream<ap_uint<IN_CH*IN_BIT>> in;
	hls::stream<ap_uint<IN_CH*IN_BIT>> out;
	ap_uint<IN_CH*IN_BIT> temp;
	for(int i = 0;i<IN_ROW;i++){
		for(int j = 0; j<IN_COL;j++){
				temp = 21;
				in.write(temp);
		}
	}
	avg_pool2d<K,IN_ROW,IN_COL,IN_CH,IN_BIT>(in,out,1);
	cout << out.read()<<endl;

	return 0;
}

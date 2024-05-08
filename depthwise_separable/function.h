#pragma once
#include <hls_stream.h>
#include <ap_int.h>
// using namespace hls;
// #include <iostream>
using namespace std;
#include <assert.h>
#include "stream_tools.h"

/**
 *  padding 函数
 *  数据宽度�? IN_BIT * SIMD
 * 
 */ 
template <	unsigned IN_BIT, 
			unsigned SIMD,
			unsigned P>
void padding_var(
    // 将每�?数竖看成�?个元�?
	stream<ap_uint<IN_BIT * SIMD> >& in,
	stream<ap_uint<IN_BIT * SIMD> >& out,
	const unsigned in_row,				// 
	const unsigned in_col,				// 
	const unsigned in_simd_pre_ch,		// ch / simd
	const unsigned reps = 1)
{
    // const unsigned OUT_ROW = in_row + 2 * P;
    const unsigned OUT_COL = in_col + 2 * P;
	// const unsigned DATA_NUM_PRE_CH = in_ch / SIMD;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) {
				// 将一 ch 的数据置�?
				append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
			}
		}

		for (unsigned h = 0; h < in_row; h++) {

			for ( unsigned s = 0; s < OUT_COL; s++ ) {
// #pragma HLS PIPELINE II=1

				if ( (s < P) || (s >= OUT_COL-P) ) {
					// temp_out = 0;
					append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
				}
				else {
					// cout << "in size :" << in.size() << endl;
					stream_move<IN_BIT * SIMD>(in, out, in_simd_pre_ch);

				}
				// out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < P; h++) {
			for (unsigned i = 0; i < OUT_COL; i++) {
				append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
			}
		}

	}
}

/**
 *  padding 函数
 */ 
template <	unsigned IN_ROW,
			unsigned IN_COL,
            unsigned IN_CH,
			unsigned IN_BIT, 
			unsigned P>
void padding(
    // 将每一数竖看成一个元素
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps = 1)
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

	ap_uint<IN_CH*IN_BIT> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) { //添加P行0填充
				out.write(0);
			}
		}

		for (unsigned h = 0; h < IN_ROW; h++) {

			for ( unsigned s = 0; s < OUT_COL; s++ ) {
#pragma HLS PIPELINE II=1
				//S=前P个或者最后P个的时候填充0
				if ( (s < P) || (s >= OUT_COL-P) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}
				
				out.write(temp_out);
			}
		}
		//最后P行进行0填充
		for (unsigned h = 0; h < P; h++) {
			for (unsigned i = 0; i < OUT_COL; i++) {
				out.write(0);
			}
		}

	}

}


template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,

			unsigned DATA_BIT,
			unsigned W_BIT,
			unsigned L_SHIFT
			>
ap_uint<OUT_BIT> bn_qurelu( ap_int<IN_BIT> in,
                ap_int<INC_BIT> inc,
                ap_int<BIAS_BIT> bias ) {   

	const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

	ap_int<IN_BIT> bn_res = in * inc + bias;
	ap_uint<OUT_BIT> res;

	if (bn_res > 0) {
		bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
		if (bn_res > 15){
			res = 15;
		} else {
			res = bn_res;
		}
	} else {
		res = 0;
	}
	return res;
    
}
//BN层和激活层，不带量化
//这个BN过于简化了,在思考可不可以重写函数
template <	unsigned IN_BIT,//input width
			unsigned OUT_BIT,//output width
			unsigned INC_BIT,// the width of the inc
			unsigned BIAS_BIT//the width of the bias

			>
ap_uint<OUT_BIT> bn_qurelu_no( ap_int<IN_BIT> in,
                ap_int<INC_BIT> inc,
                ap_int<BIAS_BIT> bias ) {

	//const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

	ap_int<IN_BIT> bn_res = in * inc + bias; // this is the BN output?
	ap_uint<OUT_BIT> res;

	if (bn_res > 0) {
		//bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
		if (bn_res > 6){
			res = 6;
		} else {
			res = bn_res;
		}
	} else {
		res = 0;
	}
	//cout << " this is BN_ACT " << endl;
	return res;

}
//线性层
template <	unsigned IN_BIT,//input width
			unsigned OUT_BIT,//output width
			unsigned INC_BIT,// the width of the inc
			unsigned BIAS_BIT//the width of the bias

			>
ap_uint<OUT_BIT> bn_qurelu_linear( ap_int<IN_BIT> in,
                ap_int<INC_BIT> inc,
                ap_int<BIAS_BIT> bias ) {

	//const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

	ap_int<IN_BIT> bn_res = in * inc + bias; // this is the BN output?
	ap_uint<OUT_BIT> res;
	//cout << " this is linear " << endl;
	res = bn_res;
	return res;

}
template<
			unsigned BIT,

			unsigned CH

		>
ap_uint<CH*BIT> adder(
			ap_uint<CH*BIT> add1,
			ap_uint<CH*BIT> add2
		)
{
		ap_uint<CH*BIT> res;
		ap_uint<BIT> add1_temp;
		ap_uint<BIT> add2_temp;
		ap_uint<BIT> res_temp;
		//cout << "add1 = " <<bitset<CH*BIT>(add1) <<endl;
		//cout << "add2 = " <<bitset<CH*BIT>(add2) <<endl;

		for(int i = 0 ;i<CH;i++){
				add1_temp = add1(CH*BIT-1,(CH-1)*BIT);
				//cout << "add_1 temp" << bitset<BIT> (add1_temp) << endl;
				add2_temp = add2(CH*BIT-1,CH*BIT-BIT);

				res_temp  = add1_temp + add2_temp;
				//cout << "res temp" << bitset<BIT> (res_temp) << endl;
				res = res << BIT;
				add1 = add1 << BIT;
				add2 = add2 << BIT;
				res(BIT-1,0) = res_temp;//从最高位拿的就得从最低位读
		}
		return res;
}

#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "sliding_window_unit.h"
#include "stream_tools.h"

template <
			unsigned K,
			unsigned IN_CH,
			unsigned IN_BIT,
			unsigned VEC_NUMS
		 >
void avg_pool_cal(
    stream<ap_uint<IN_CH*IN_BIT>>& vec,
    stream<ap_uint<IN_CH*IN_BIT>>& out, // 修改输出为每次一个通道的平均值
    const unsigned reps = 1)
{
    ap_uint<16> sum[IN_CH]; // 使用数组来分别累加每个通道
    for(int i = 0;i <IN_CH;i++){
    	sum[i] = 0;
    }
    stream<ap_uint<IN_BIT>> out_temp;
#pragma HLS ARRAY_PARTITION complete dim=1 // 完全分区以提高并行性

    unsigned k_cnt = 0;

    for (unsigned rep = 0; rep < reps*VEC_NUMS; rep++) {
#pragma HLS PIPELINE II=1

        ap_uint<IN_CH*IN_BIT> temp_vec = vec.read();
       // cout << temp_vec << endl;

        for (unsigned c = 0; c < IN_CH; c++) {
#pragma HLS UNROLL
            ap_uint<IN_BIT> temp = temp_vec((c+1)*IN_BIT-1, c*IN_BIT);
            //cout << "sum ["<<c<<"]"<<sum[c]<<endl;
            sum[c] += temp; // 分别对每个通道进行累加

        }

        if (++k_cnt == K*K) {
            for (unsigned c = 0; c < IN_CH; c++) {
#pragma HLS UNROLL

                ap_uint<IN_BIT> avg = sum[c] / (K*K);
                //cout << "avg" << avg << endl;
                out_temp.write(avg); // 输出每个通道的平均值
                sum[c] = 0; // 重置累加器
            }
            k_cnt = 0; // 重置计数器
        }
    }
    StreamingDataWidthConverter_Batch<IN_BIT, IN_CH*IN_BIT, 1>(out_temp, out, reps);
}


/**
 * 平均池化
 * TODO 当前只给 K = 7, S = 1做优化
 */
template <	unsigned K,                 // kernel
			// unsigned S,                 // stride
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT>
void avg_pool2d(
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
    // TODO IN_ROW % S != 0
    // 暂时只处理特殊情�?
    const unsigned OUT_ROW = 1;
    const unsigned OUT_COL = 1;
    const unsigned S = 1;

    // 产生滑动窗口数据，可能并不需要
    //hls::stream<ap_uint<IN_CH*IN_BIT>> swu_out("swu_out");
    //SWU<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT>(in, swu_out, reps);
    //cout << in.read();
    // 处理数据
	// POOL<IN_ROW*IN_COL, Ibit, K, Cin, 1>(swu_out, out, reps);
    avg_pool_cal<K, IN_CH, IN_BIT, OUT_ROW*OUT_COL*K*K>(in, out, reps);
}

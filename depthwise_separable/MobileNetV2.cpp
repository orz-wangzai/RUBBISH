// #define DEBUG

#ifdef DEBUG
#include <iostream>
#include <fstream>
using namespace std;

#endif


#include <stdint.h>
#include <ap_int.h>
#include <hls_video.h>
#include "stream_tools.h"
#include "function.h"
#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "config.h"
#include "param.h"
#include "conv2d.h"
#include "pool2d.h"
#include "bn_qrelu2d.h"

//看上去是输入的图片尺寸，这里可以根据自己需求定义
#define IN_IMAGE_WIDTH  640
#define IN_IMAGE_HEIGHT 360

//应该是实际尺寸，对于MobileNetV2应该是224*224
#define RESIZE_IMAGE_WIDTH 224
#define RESIZE_IMAGE_HEIGHT 224

//stream_to_mat
//??hls::Mat ?????
void stream_to_mat (hls::stream<ap_uint<24>>&in, 
		 hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> & raw_img)  { //HLS_8UC3是什么
    
	for (int i=0; i<IN_IMAGE_HEIGHT; i++) {
		for (int j=0; j<IN_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8>> pix;
            ap_uint<24> in_data = in.read();
            for (unsigned int p=0; p < 3; p ++) {
                
                pix.val[p] = in_data(8*p+7, 8*p);
            }
			raw_img << pix;
		}	
	}

}
//mat_to_stream
void mat_to_stream (hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> & resize_img,
                    hls::stream<ap_uint<24>> & out ) {
    
	for (int i=0; i<RESIZE_IMAGE_HEIGHT; i++) {
		for (int j=0; j<RESIZE_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8>> pix; //3通道个8位的值
            resize_img >> pix;
            ap_uint<24> out_data;
            for (unsigned int p=0; p < 3; p ++) {
                out_data(8*p+7, 8*p) = pix.val[p];
            }
            out.write(out_data);
		}	
	}

}

//这就是为什么需要把数据转换成Mat形式
void resize(hls::stream<ap_uint<24>> &in, hls::stream<ap_uint<24>> & out) {
#pragma HLS dataflow
    hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> raw_img;

#pragma HLS STREAM variable=raw_img depth = 128 dim=1
    hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> resize_img;

#pragma HLS STREAM variable=resize_img depth = 128 dim=1
    stream_to_mat(in, raw_img);
    // hls::Resize(raw_img, resize_img, HLS_INTER_LINEAR);
    hls::Resize_opr_linear(raw_img, resize_img);
    mat_to_stream(resize_img, out);
}


//对多张图进行resize操作
void resize_batch(hls::stream<ap_uint<24>> &in, hls::stream<ap_uint<24>> & out, unsigned int reps) {
    for (unsigned int rep=0; rep < reps; rep ++) {
        resize(in, out);
    }
}


//使用自定义数据类型当作输入输出
void do_compute(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps) {
#pragma HLS DATAFLOW
		//?????????????????????64?
    const unsigned int num_per_rep = IN_IMAGE_WIDTH * IN_IMAGE_HEIGHT * 3 * 8 / 64;
    
    hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
    //extract data
#pragma HLS STREAM variable=in_stream_extract depth=16 dim=1
	ExtractPixels<64, num_per_rep> (in, in_stream_extract, reps);

	//pack up data to from 3 64bit to 3*64bit
    hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable=in_stream0 depth=16 dim=1
    StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract, in_stream0, reps);


	hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream1("in_stream1"); //这里是IN_PUT_WEIGHT*IN_PUT_HEIGHT
#pragma HLS STREAM variable=in_stream1 depth=16 dim=1
    StreamingDataWidthConverter_Batch<64 * 3, CONV_0_IN_BIT * CONV_0_IFM_CH, num_per_rep / 3> (in_stream0, in_stream1, reps);
    //这里得到的是一个320*640 个 8*3位宽的数据
#ifdef DEBUG
    cout << "in_stream1 size " << in_stream1.size() << endl;

#endif
    //这里可以得到一个160*320 个 8*3位宽的数据
    hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream2("in_stream2");
#pragma HLS STREAM variable=in_stream2 depth=16 dim=1
    resize_batch(in_stream1, in_stream2, reps);
#ifdef DEBUG
    cout << "in_stream2 size " << in_stream2.size() << endl;
    // hls::stream<ap_uint<8>> res("res");
    // StreamingDataWidthConverter_Batch<CONV_0_IN_BIT * CONV_0_IFM_CH, 8, 320*3>(in_stream, res, 1);
    // int data[3][320][3];
    // for (int n=0; n < 3; n ++)
    //     for (int i=0; i < 320; i ++) {
    //         for (int j=0; j < 3; j ++)
    //             data[n][i][j] = res.read();
    //     }

    // for (int n=0; n < 3; n ++)
    //     for (int i=0; i < 3; i ++) {
    //         for (int j=0; j < 3; j ++)
    //             cout << data[n][i][j] << " ";
    //     }
    // return;
    // 杈撳叆鏁版澁娌￠棶锟???
#endif
    //MobileNetV2 第一层 输出通道是32 步长为2
    hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OFM_CH>>  conv_0_out("conv_0_out");
#pragma HLS STREAM variable=conv_0_out depth=128 dim=1
    conv3x3_bn<
                    CONV_0_IFM_ROW,
                    CONV_0_IFM_COL,
                    CONV_0_IFM_CH,
                    CONV_0_IN_BIT,

                    CONV_0_OFM_CH,
                    CONV_0_OUT_BIT,

                    CONV_0_W_BIT,
                    32,                     
                    CONV_0_INC_BIT,
                    CONV_0_BIAS_BIT,

                    CONV_0_SIMD,
                    CONV_0_PE,
					CONV_0_S
               >
    			(
                in_stream2,
                conv_0_w,
                conv_0_inc,
                conv_0_bias,
                conv_0_out,
                reps
				);
#ifdef DEBUG
    cout << "conv_0_out size " << conv_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第二层 IN_PUT 此时是112*112*32的4位数据
 hls::stream<ap_uint <Bottle_neck_0_OUT_BIT * Bottle_neck_0_OFM_CH>>  Bottle_neck_0_out("Bottle_neck_0_out");
#pragma HLS STREAM variable = Bottle_neck_0_out depth=128 dim=1
   Inverted<
               Bottle_neck_0_IFM_ROW,
               Bottek_neck_0_IFM_COL,
               Bottle_neck_0_IFM_IN_CH,
               Bottle_neck_0_IN_BIT,

               Bottle_neck_0_OFM_CH,
               Bottle_neck_0_OUT_BIT,

               Bottle_neck_0_W_BIT,
               32,
               Bottle_neck_0_INC_BIT,
               Bottle_neck_0_BIAS_BIT,

               Bottle_neck_0_expand_ratio,
               Bottle_neck_0_PE,
               Bottle_neck_0_S
               >
    		(
    			conv_0_out,
				Bottle_neck_0_weights_conv3, //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_0_weights_conv1,//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_0_weigths_conv1_dp,//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_0_inc_conv3,
    			Bottle_neck_0_bias_3,

    			Bottle_neck_0_inc_pw_1, //for normal conv1x1
    			Bottle_neck_0_bias_pw_1,

				Bottle_neck_0_inc_pw_dp,
				Bottle_necK_0_bias_pw_dp,

    			Bottle_0_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
#ifdef DEBUG
    cout << "Bottle_neck_0_out " << Bottle_neck_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第三层 t c n s 为 6,24,2,2
hls::stream<ap_uint <Bottle_neck_1_OUT_BIT * Bottle_neck_1_OFM_CH>>  Bottle_neck_1_out("Bottle_neck_1_out");
#pragma HLS STREAM variable=Bottle_neck_1_out depth=128 dim=1
hls::stream<ap_uint <Bottle_neck_1_OUT_BIT * Bottle_neck_1_OFM_CH>>  Bottle_neck_1_temp("Bottle_neck_1_out");
for(int i = 0; i < Bottle_neck_1_loop; i++)
{
   if (i == 0)
   {
      /* code */
      Inverted<
               Bottle_neck_1_IFM_ROW,
               Bottek_neck_1_IFM_COL,
               Bottle_neck_1_IFM_IN_CH,
               Bottle_neck_1_IN_BIT,

               Bottle_neck_1_OFM_CH,
               Bottle_neck_1_OUT_BIT,

               Bottle_neck_1_W_BIT,
               32,
               Bottle_neck_1_INC_BIT,
               Bottle_neck_1_BIAS_BIT,
               
               Bottle_neck_1_expand_ratio,
               Bottle_neck_1_PE,
               Bottle_neck_1_S
               >
    		(
    			Bottle_neck_0_out,
				Bottle_neck_1_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_1_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_1_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_1_inc_conv3[i],
    			Bottle_neck_1_bias_3[i],

    			Bottle_neck_1_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_1_bias_pw_1[i],

				Bottle_neck_1_inc_pw_dp[i],
				Bottle_necK_1_bias_pw_dp[i],

    			Bottle_neck_1_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   }
   else if (i == Bottle_neck_1_loop - 1)
   {
      /* code */
      Inverted<
               Bottle_neck_1_IFM_ROW,
               Bottek_neck_1_IFM_COL,
               Bottle_neck_1_IFM_IN_CH,
               Bottle_neck_1_IN_BIT,

               Bottle_neck_1_OFM_CH,
               Bottle_neck_1_OUT_BIT,

               Bottle_neck_1_W_BIT,
               32,
               Bottle_neck_1_INC_BIT,
               Bottle_neck_1_BIAS_BIT,
               
               Bottle_neck_1_expand_ratio,
               Bottle_neck_1_PE,
               Bottle_neck_1_S
               >
    		(
    			Bottle_neck_1_temp,
				Bottle_neck_1_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_1_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_1_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_1_inc_conv3[i],
    			Bottle_neck_1_bias_3[i],

    			Bottle_neck_1_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_1_bias_pw_1[i],

				Bottle_neck_1_inc_pw_dp[i],
				Bottle_necK_1_bias_pw_dp[i],

    			Bottle_neck_1_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
      
   }
   else
   {
      /* code */
      Inverted<
               Bottle_neck_1_IFM_ROW,
               Bottek_neck_1_IFM_COL,
               Bottle_neck_1_IFM_IN_CH,
               Bottle_neck_1_IN_BIT,

               Bottle_neck_1_OFM_CH,
               Bottle_neck_1_OUT_BIT,

               Bottle_neck_1_W_BIT,
               32,
               Bottle_neck_1_INC_BIT,
               Bottle_neck_1_BIAS_BIT,
               
               Bottle_neck_1_expand_ratio,
               Bottle_neck_1_PE,
               Bottle_neck_1_S
               >
    		(
    			Bottle_neck_1_temp,
				Bottle_neck_1_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_1_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_1_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_1_inc_conv3[i],
    			Bottle_neck_1_bias_3[i],

    			Bottle_neck_1_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_1_bias_pw_1[i],

				Bottle_neck_1_inc_pw_dp[i],
				Bottle_necK_1_bias_pw_dp[i],

    			Bottle_neck_1_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   } 
}

#ifdef DEBUG
    cout << "Bottle_neck_0_out " << Bottle_neck_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第四层 t c n s 为 6 32 3 2
hls::stream<ap_uint <Bottle_neck_2_OUT_BIT * Bottle_neck_2_OFM_CH>>  Bottle_neck_2_out("Bottle_neck_2_out");
#pragma HLS STREAM variable=Bottle_neck_2_out depth=128 dim=1
hls::stream<ap_uint <Bottle_neck_2_OUT_BIT * Bottle_neck_2_OFM_CH>>  Bottle_neck_2_temp("Bottle_neck_2_out");
for(int i = 0; i < Bottle_neck_2_loop; i++)
{
   if (i == 0)
   {
      /* code */
      Inverted<
               Bottle_neck_2_IFM_ROW,
               Bottek_neck_2_IFM_COL,
               Bottle_neck_2_IFM_IN_CH,
               Bottle_neck_2_IN_BIT,

               Bottle_neck_2_OFM_CH,
               Bottle_neck_2_OUT_BIT,

               Bottle_neck_2_W_BIT,
               32,
               Bottle_neck_2_INC_BIT,
               Bottle_neck_2_BIAS_BIT,
               
               Bottle_neck_2_expand_ratio,
               Bottle_neck_2_PE,
               Bottle_neck_2_S
               >
    		(
    			Bottle_neck_1_out,
				Bottle_neck_2_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_2_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_2_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_2_inc_conv3[i],
    			Bottle_neck_2_bias_3[i],

    			Bottle_neck_2_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_2_bias_pw_1[i],

				Bottle_neck_2_inc_pw_dp[i],
				Bottle_necK_2_bias_pw_dp[i],

    			Bottle_neck_2_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   }
   else if (i == Bottle_neck_2_loop - 1)
   {
      /* code */
      Inverted<
               Bottle_neck_2_IFM_ROW,
               Bottek_neck_2_IFM_COL,
               Bottle_neck_2_IFM_IN_CH,
               Bottle_neck_2_IN_BIT,

               Bottle_neck_2_OFM_CH,
               Bottle_neck_2_OUT_BIT,

               Bottle_neck_2_W_BIT,
               32,
               Bottle_neck_2_INC_BIT,
               Bottle_neck_2_BIAS_BIT,
               
               Bottle_neck_2_expand_ratio,
               Bottle_neck_2_PE,
               Bottle_neck_2_S
               >
    		(
    			Bottle_neck_2_temp,
				Bottle_neck_2_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_2_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_2_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_2_inc_conv3[i],
    			Bottle_neck_2_bias_3[i],

    			Bottle_neck_2_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_2_bias_pw_1[i],

				Bottle_neck_2_inc_pw_dp[i],
				Bottle_necK_2_bias_pw_dp[i],

    			Bottle_neck_2_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
      
   }
   else
   {
      /* code */
      Inverted<
               Bottle_neck_2_IFM_ROW,
               Bottek_neck_2_IFM_COL,
               Bottle_neck_2_IFM_IN_CH,
               Bottle_neck_2_IN_BIT,

               Bottle_neck_2_OFM_CH,
               Bottle_neck_2_OUT_BIT,

               Bottle_neck_2_W_BIT,
               32,
               Bottle_neck_2_INC_BIT,
               Bottle_neck_2_BIAS_BIT,
               
               Bottle_neck_2_expand_ratio,
               Bottle_neck_2_PE,
               Bottle_neck_2_S
               >
    		(
    			Bottle_neck_2_temp,
				Bottle_neck_2_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_2_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_2_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_2_inc_conv3[i],
    			Bottle_neck_2_bias_3[i],

    			Bottle_neck_2_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_2_bias_pw_1[i],

				Bottle_neck_2_inc_pw_dp[i],
				Bottle_necK_2_bias_pw_dp[i],

    			Bottle_neck_2_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   } 
}

#ifdef DEBUG
    cout << "Bottle_neck_2_out " << Bottle_neck_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第五层 t c n s 为 6 64 4 2
hls::stream<ap_uint <Bottle_neck_3_OUT_BIT * Bottle_neck_3_OFM_CH>>  Bottle_neck_3_out("Bottle_neck_3_out");
#pragma HLS STREAM variable=Bottle_neck_3_out depth=128 dim=1
hls::stream<ap_uint <Bottle_neck_3_OUT_BIT * Bottle_neck_3_OFM_CH>>  Bottle_neck_3_temp("Bottle_neck_3_out");
for(int i = 0; i < Bottle_neck_3_loop; i++)
{
   if (i == 0)
   {
      /* code */
      Inverted<
               Bottle_neck_3_IFM_ROW,
               Bottek_neck_3_IFM_COL,
               Bottle_neck_3_IFM_IN_CH,
               Bottle_neck_3_IN_BIT,

               Bottle_neck_3_OFM_CH,
               Bottle_neck_3_OUT_BIT,

               Bottle_neck_3_W_BIT,
               32,
               Bottle_neck_3_INC_BIT,
               Bottle_neck_3_BIAS_BIT,
               
               Bottle_neck_3_expand_ratio,
               Bottle_neck_3_PE,
               Bottle_neck_3_S
               >
    		(
    			Bottle_neck_2_out,
				Bottle_neck_3_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_3_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_3_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_3_inc_conv3[i],
    			Bottle_neck_3_bias_3[i],

    			Bottle_neck_3_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_3_bias_pw_1[i],

				Bottle_neck_3_inc_pw_dp[i],
				Bottle_necK_3_bias_pw_dp[i],

    			Bottle_neck_3_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   }
   else if (i == Bottle_neck_3_loop - 1)
   {
      /* code */
      Inverted<
               Bottle_neck_3_IFM_ROW,
               Bottek_neck_3_IFM_COL,
               Bottle_neck_3_IFM_IN_CH,
               Bottle_neck_3_IN_BIT,

               Bottle_neck_3_OFM_CH,
               Bottle_neck_3_OUT_BIT,

               Bottle_neck_3_W_BIT,
               32,
               Bottle_neck_3_INC_BIT,
               Bottle_neck_3_BIAS_BIT,
               
               Bottle_neck_3_expand_ratio,
               Bottle_neck_3_PE,
               Bottle_neck_3_S
               >
    		(
    			Bottle_neck_3_temp,
				Bottle_neck_3_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_3_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_3_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_3_inc_conv3[i],
    			Bottle_neck_3_bias_3[i],

    			Bottle_neck_3_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_3_bias_pw_1[i],

				Bottle_neck_3_inc_pw_dp[i],
				Bottle_necK_3_bias_pw_dp[i],

    			Bottle_neck_3_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
      
   }
   else
   {
      /* code */
      Inverted<
               Bottle_neck_3_IFM_ROW,
               Bottek_neck_3_IFM_COL,
               Bottle_neck_3_IFM_IN_CH,
               Bottle_neck_3_IN_BIT,

               Bottle_neck_3_OFM_CH,
               Bottle_neck_3_OUT_BIT,

               Bottle_neck_3_W_BIT,
               32,
               Bottle_neck_3_INC_BIT,
               Bottle_neck_3_BIAS_BIT,
               
               Bottle_neck_3_expand_ratio,
               Bottle_neck_3_PE,
               Bottle_neck_3_S
               >
    		(
    			Bottle_neck_3_temp,
				Bottle_neck_3_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_3_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_3_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_3_inc_conv3[i],
    			Bottle_neck_3_bias_3[i],

    			Bottle_neck_3_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_3_bias_pw_1[i],

				Bottle_neck_3_inc_pw_dp[i],
				Bottle_necK_3_bias_pw_dp[i],

    			Bottle_neck_3_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   } 
}

#ifdef DEBUG
    cout << "Bottle_neck_3_out " << Bottle_neck_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第六层 t c n s 为 6 96 3 1
hls::stream<ap_uint <Bottle_neck_4_OUT_BIT * Bottle_neck_4_OFM_CH>>  Bottle_neck_4_out("Bottle_neck_4_out");
#pragma HLS STREAM variable=Bottle_neck_4_out depth=128 dim=1
hls::stream<ap_uint <Bottle_neck_4_OUT_BIT * Bottle_neck_4_OFM_CH>>  Bottle_neck_4_temp("Bottle_neck_4_out");
for(int i = 0; i < Bottle_neck_4_loop; i++)
{
   if (i == 0)
   {
      /* code */
      Inverted<
               Bottle_neck_4_IFM_ROW,
               Bottek_neck_4_IFM_COL,
               Bottle_neck_4_IFM_IN_CH,
               Bottle_neck_4_IN_BIT,

               Bottle_neck_4_OFM_CH,
               Bottle_neck_4_OUT_BIT,

               Bottle_neck_4_W_BIT,
               32,
               Bottle_neck_4_INC_BIT,
               Bottle_neck_4_BIAS_BIT,
               
               Bottle_neck_4_expand_ratio,
               Bottle_neck_4_PE,
               Bottle_neck_4_S
               >
    		(
    			Bottle_neck_3_out,
				Bottle_neck_4_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_4_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_4_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_4_inc_conv3[i],
    			Bottle_neck_4_bias_3[i],

    			Bottle_neck_4_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_4_bias_pw_1[i],

				Bottle_neck_4_inc_pw_dp[i],
				Bottle_necK_4_bias_pw_dp[i],

    			Bottle_neck_4_temp, //输出是OUT_CH IN_ROW*IN_CH
				);
   }
   else if (i == Bottle_neck_4_loop - 1)
   {
      /* code */
      Inverted<
               Bottle_neck_4_IFM_ROW,
               Bottek_neck_4_IFM_COL,
               Bottle_neck_4_IFM_IN_CH,
               Bottle_neck_4_IN_BIT,

               Bottle_neck_4_OFM_CH,
               Bottle_neck_4_OUT_BIT,

               Bottle_neck_4_W_BIT,
               32,
               Bottle_neck_4_INC_BIT,
               Bottle_neck_4_BIAS_BIT,
               
               Bottle_neck_4_expand_ratio,
               Bottle_neck_4_PE,
               Bottle_neck_4_S
               >
    		(
    			Bottle_neck_4_temp,
				Bottle_neck_4_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_4_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_4_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_4_inc_conv3[i],
    			Bottle_neck_4_bias_3[i],

    			Bottle_neck_4_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_4_bias_pw_1[i],

				Bottle_neck_4_inc_pw_dp[i],
				Bottle_necK_4_bias_pw_dp[i],

    			Bottle_neck_4_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
      
   }
   else
   {
      /* code */
      Inverted<
               Bottle_neck_4_IFM_ROW,
               Bottek_neck_4_IFM_COL,
               Bottle_neck_4_IFM_IN_CH,
               Bottle_neck_4_IN_BIT,

               Bottle_neck_4_OFM_CH,
               Bottle_neck_4_OUT_BIT,

               Bottle_neck_4_W_BIT,
               32,
               Bottle_neck_4_INC_BIT,
               Bottle_neck_4_BIAS_BIT,
               
               Bottle_neck_4_expand_ratio,
               Bottle_neck_4_PE,
               Bottle_neck_4_S
               >
    		(
    			Bottle_neck_4_temp,
				Bottle_neck_4_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_4_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_4_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_4_inc_conv3[i],
    			Bottle_neck_4_bias_3[i],

    			Bottle_neck_4_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_4_bias_pw_1[i],

				Bottle_neck_4_inc_pw_dp[i],
				Bottle_necK_4_bias_pw_dp[i],

    			Bottle_neck_4_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   } 

}
#ifdef DEBUG
    cout << "Bottle_neck_4_out " << Bottle_neck_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第七层 t c n s 为 6 160 3 2
hls::stream<ap_uint <Bottle_neck_5_OUT_BIT * Bottle_neck_5_OFM_CH>>  Bottle_neck_5_out("Bottle_neck_5_out");
#pragma HLS STREAM variable=Bottle_neck_5_out depth=128 dim=1
hls::stream<ap_uint <Bottle_neck_5_OUT_BIT * Bottle_neck_5_OFM_CH>>  Bottle_neck_5_temp("Bottle_neck_5_out");
for(int i = 0; i < Bottle_neck_5_loop; i++)
{
   if (i == 0)
   {
      /* code */
      Inverted<
               Bottle_neck_5_IFM_ROW,
               Bottek_neck_5_IFM_COL,
               Bottle_neck_5_IFM_IN_CH,
               Bottle_neck_5_IN_BIT,

               Bottle_neck_5_OFM_CH,
               Bottle_neck_5_OUT_BIT,

               Bottle_neck_5_W_BIT,
               32,
               Bottle_neck_5_INC_BIT,
               Bottle_neck_5_BIAS_BIT,
               
               Bottle_neck_5_expand_ratio,
               Bottle_neck_5_PE,
               Bottle_neck_5_S
               >
    		(
    			Bottle_neck_4_out,
				Bottle_neck_5_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_5_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_5_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_5_inc_conv3[i],
    			Bottle_neck_5_bias_3[i],

    			Bottle_neck_5_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_5_bias_pw_1[i],

				Bottle_neck_5_inc_pw_dp[i],
				Bottle_necK_5_bias_pw_dp[i],

    			Bottle_neck_5_temp, //输出是OUT_CH IN_ROW*IN_CH
				);
   }
   else if (i == Bottle_neck_5_loop - 1)
   {
      /* code */
      Inverted<
               Bottle_neck_5_IFM_ROW,
               Bottek_neck_5_IFM_COL,
               Bottle_neck_5_IFM_IN_CH,
               Bottle_neck_5_IN_BIT,

               Bottle_neck_5_OFM_CH,
               Bottle_neck_5_OUT_BIT,

               Bottle_neck_5_W_BIT,
               32,
               Bottle_neck_5_INC_BIT,
               Bottle_neck_5_BIAS_BIT,
               
               Bottle_neck_5_expand_ratio,
               Bottle_neck_5_PE,
               Bottle_neck_5_S
               >
    		(
    			Bottle_neck_5_temp,
				Bottle_neck_5_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_5_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_5_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重

    			Bottle_neck_5_inc_conv3[i],
    			Bottle_neck_5_bias_3[i],

    			Bottle_neck_5_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_5_bias_pw_1[i],

				Bottle_neck_5_inc_pw_dp[i],
				Bottle_necK_5_bias_pw_dp[i],

    			Bottle_neck_5_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
      
   }
   else
   {
      /* code */
      Inverted<
               Bottle_neck_5_IFM_ROW,
               Bottek_neck_5_IFM_COL,
               Bottle_neck_5_IFM_IN_CH,
               Bottle_neck_5_IN_BIT,

               Bottle_neck_5_OFM_CH,
               Bottle_neck_5_OUT_BIT,

               Bottle_neck_5_W_BIT,
               32,
               Bottle_neck_5_INC_BIT,
               Bottle_neck_5_BIAS_BIT,
               
               Bottle_neck_5_expand_ratio,
               Bottle_neck_5_PE,
               Bottle_neck_5_S
               >
    		(
    			Bottle_neck_5_temp,
				Bottle_neck_5_weights_conv3[i], //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_5_weights_conv1[i],//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_5_weigths_conv1_dp[i],//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重
    			Bottle_neck_5_inc_conv3[i],
    			Bottle_neck_5_bias_3[i],

    			Bottle_neck_5_inc_pw_1[i], //for normal conv1x1
    			Bottle_neck_5_bias_pw_1[i],

				Bottle_neck_5_inc_pw_dp[i],
				Bottle_necK_5_bias_pw_dp[i],

    			Bottle_neck_5_temp, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
   } 

}
#ifdef DEBUG
    cout << "Bottle_neck_5_out " << Bottle_neck_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第八层 t c n s 6 320 1 1
//思考这里怎么改变才能让位宽下降
 hls::stream<ap_uint <Bottle_neck_6_OUT_BIT * Bottle_neck_6_OFM_CH>>  Bottle_neck_6_out("Bottle_neck_6_out");
#pragma HLS STREAM variable = Bottle_neck_6_out depth=128 dim=1
   Inverted<
               Bottle_neck_6_IFM_ROW,
               Bottek_neck_6_IFM_COL,
               Bottle_neck_6_IFM_IN_CH,
               Bottle_neck_6_IN_BIT,

               Bottle_neck_6_OFM_CH,
               Bottle_neck_6_OUT_BIT,

               Bottle_neck_6_W_BIT,
               32,
               Bottle_neck_6_INC_BIT,
               Bottle_neck_6_BIAS_BIT,

               Bottle_neck_6_expand_ratio,
               Bottle_neck_6_PE,
               Bottle_neck_6_S
               >
    		(
    			Bottle_neck_5_out,
				Bottle_neck_6_weights_conv3, //const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9]
    			Bottle_neck_6_weights_conv1,//const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE]
    			Bottle_neck_6_weigths_conv1_dp,//const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重
    			Bottle_neck_6_inc_conv3,
    			Bottle_neck_6_bias_3,

    			Bottle_neck_6_inc_pw_1, //for normal conv1x1
    			Bottle_neck_6_bias_pw_1,

				Bottle_neck_6_inc_pw_dp,
				Bottle_necK_6_bias_pw_dp,

    			Bottle_6_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			reps
				);
#ifdef DEBUG
    cout << "Bottle_neck_6_out " << Bottle_neck_6_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第九层 conv2d1x1 c = 1280 
hls::stream<ap_uint <CONV_9_OUT_BIT * CONV_9_OFM_CH>>  CONV_9_out("CONV_9_out");
conv1x1_dp_bn_act<
                  CONV_9_IFM_ROW,
                  CONV_9_IFM_COL,
                  CONV_9_IFM_CH,
                  CONV_9_IN_BIT,

                  CONV_9_OFM_CH,
                  CONV_9_OUT_BIT,
                  
                  CONV_9_INC_BIT,
                  CONV_9_BIAS_BIT,

                  CONV_9_W_BIT,
                  32,
                  CONV_9_PE
                  >
                  (
                     Bottle_neck_6_out,
                     CONV_9_weights,
                     CONV_9_inc,
                     CONV_9_bias,
                     CONV_9_out,
                     reps
                  );
#ifdef DEBUG
    cout << "conv_9_out " << conv_9_out.size() << endl;
    // hls::stream<ap_uint<4>> res(S"res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
//第十层 池化层
hls::stream<ap_uint <CONV_9_OUT_BIT * CONV_9_OFM_CH>>  avg_pool_out("avg_pool_out");
avg_pool2d<
            7,                 // kernel
            7,
			7
			1280,
			CONV_9_OUT_BIT
            >
         (
            conv_9_out,
            avg_pool_out, 
            reps
         );

//全连接层
//第十一层 卷积层
//最后一层 SoftMax 实现分类 已放弃
}
void MobileNetV2(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps) {

#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=reps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable = conv_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_1_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_3_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_4_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_5_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_6_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_7_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_8_w complete dim = 1

    do_compute(in, out, reps);

}


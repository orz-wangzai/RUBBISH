#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;

#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "function.h"
#include "stream_tools.h"




//conv3*3 逐深度卷积 输入是一个n*n*3的矩阵（比如说），那么输出应该是3个变化后的矩阵

template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			//unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned W_BIT,
			unsigned M_BIT,

			unsigned S>
void conv3x3_dp(
	stream<ap_uint<IN_BIT * IN_CH> >& in,
	const ap_uint<W_BIT> weights[IN_CH][9],
	stream<ap_uint<OUT_BIT>> out[IN_CH], //输出是IN_CH个变化后的矩阵
	const unsigned reps = 1
	)
{
#pragma HLS DATAFLOW

	const unsigned INTER_ROW = IN_ROW + 2;
	const unsigned INTER_COL = IN_COL + 2;
	// 暂时认为输入 输出维度不变

	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;


	// padding
	stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);
	cout << "the size after padding " << padding_out.size() << endl;
	//现在把padding_out转换成多个子数据流的矩阵
    stream<ap_uint<IN_BIT> > sub_in[IN_CH];
    int count = 0;
	for(int i = 0 ;i<IN_ROW + 2;i++){
        	for(int j = 0;j< IN_COL + 2;j++){
        		ap_uint<IN_BIT*IN_CH> temp;
        		temp = padding_out.read();

        		//cout<< bitset<IN_CH * IN_BIT> (temp)<<endl;

        		for(int k = 0;k<IN_CH;k++){
        			ap_uint<IN_BIT> temp_sub;

        			temp_sub = temp(IN_BIT*IN_CH-1,IN_CH*IN_BIT-IN_BIT);

        			//cout<< bitset<IN_BIT> (temp_sub)<<endl;

        			sub_in[k].write(temp_sub);

        			temp = temp << IN_BIT;
        			//cout<< bitset<IN_CH * IN_BIT> (temp)<<endl;
        		}
        	}
        }


	// 滑动窗口 这下懂了 这个地方可以改成循环
	stream<ap_uint<IN_BIT> > swu_out("swu_out");
	ap_uint<W_BIT> weights_DP[1][9];
	for(int i = 0;i < IN_CH;i++)
	{
			//滑动窗口
			for(int j = 0; j < 9; j++)
			{
				weights_DP[0][j] = weights[i][j];
			}
			SWU<3,S,INTER_ROW, INTER_COL,1,IN_BIT> (sub_in[i],swu_out, reps);
			//矩阵向量计算
			if(S == 1){
				matrix_vector_unit<3*3, 1, IN_BIT, W_BIT, M_BIT, 1, 1, OUT_ROW*OUT_COL> (swu_out, weights_DP,out[i], reps);

			}
			if(S == 2){
				matrix_vector_unit<3*3, 1, IN_BIT, W_BIT, M_BIT, 1, 1, OUT_ROW/2*OUT_COL/2> (swu_out, weights_DP,out[i], reps);
			}

	}
	//cout << "the size after depth_wise = " << out[0].size()<<endl;

}
//conv3x3 深度卷积模块 带有激活层和批量归一化函数
template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			unsigned W_BIT,
			unsigned M_BIT,

			unsigned S>
void conv3x3_dp_bn_act(
	stream<ap_uint<IN_BIT * IN_CH> >& in,
	const ap_uint<W_BIT> weights[IN_CH][9],
	const ap_int<INC_BIT> inc[1][IN_CH],
	const ap_int <BIAS_BIT> bias[1][IN_CH],
	stream<ap_uint<OUT_BIT>> out[IN_CH], //输出是IN_CH个变化后的矩阵
	const unsigned reps = 1
	)
{
#pragma HLS DATAFLOW

	const unsigned INTER_ROW = IN_ROW + 2;
	const unsigned INTER_COL = IN_COL + 2;
	// 暂时认为输入 输出维度不变

	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;


	// padding
	stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);
	cout << "the size after padding " << padding_out.size() << endl;
	//现在把padding_out转换成多个子数据流的矩阵
    stream<ap_uint<IN_BIT> > sub_in[IN_CH];
    int count = 0;
	for(int i = 0 ;i<IN_ROW + 2;i++){
        	for(int j = 0;j< IN_COL + 2;j++){
        		ap_uint<IN_BIT*IN_CH> temp;
        		temp = padding_out.read();

        		//cout<< bitset<IN_CH * IN_BIT> (temp)<<endl;

        		for(int k = 0;k<IN_CH;k++){
        			ap_uint<IN_BIT> temp_sub;

        			temp_sub = temp(IN_BIT*IN_CH-1,IN_CH*IN_BIT-IN_BIT);

        			//cout<< bitset<IN_BIT> (temp_sub)<<endl;

        			sub_in[k].write(temp_sub);

        			temp = temp << IN_BIT;
        			//cout<< bitset<IN_CH * IN_BIT> (temp)<<endl;
        		}
        	}
        }


	// 滑动窗口 这下懂了 这个地方可以改成循环
	stream<ap_uint<IN_BIT> > swu_out("swu_out");
	ap_uint<W_BIT> weights_DP[1][9];
	ap_int<INC_BIT> inc_1[1][1];
	ap_int<BIAS_BIT> bias_1[1][1];
	for(int i = 0;i < IN_CH;i++)
	{
			//变换输入
			for(int j = 0; j < 9; j++)
			{
				weights_DP[0][j] = weights[i][j];
			}
			//滑动窗口
			inc_1[0][0] = inc[0][i];
			//cout << "inc is " << inc_1[0][1] <<endl;
			bias_1[0][0] = bias[0][i];
			//cout << "bias is "<<bias_1[0][1]<<endl;
			SWU<3,S,INTER_ROW, INTER_COL,1,IN_BIT> (sub_in[i],swu_out, reps);
			//矩阵向量计算
			if(S == 1){
				//matrix_vector_unit<3*3, 1, IN_BIT, W_BIT, M_BIT, 1, 1, OUT_ROW*OUT_COL> (swu_out, weights_DP,out[i], reps);
				matrix_vector_act_unit_no<3*3, 1, IN_BIT, OUT_BIT,W_BIT, M_BIT,INC_BIT,BIAS_BIT, 1, 1, OUT_ROW*OUT_COL> (swu_out,weights_DP,inc_1,bias_1,out[i],reps);
				//cout << "fuck"<<out[i].read()<<endl;
			}
			if(S == 2){
				//matrix_vector_unit<3*3, 1, IN_BIT, W_BIT, M_BIT, 1, 1, OUT_ROW/2*OUT_COL/2> (swu_out, weights_DP,out[i], reps);
				matrix_vector_act_unit_no<3*3, 1, IN_BIT, OUT_BIT,W_BIT, M_BIT,INC_BIT,BIAS_BIT, 1, 1, OUT_ROW/2*OUT_COL/2> (swu_out,weights_DP,inc_1,bias_1,out[i],reps);
				//cout << "conv3x3tempsize " << out[i].size() << endl;
			}

	}
	//cout << "the size after depth_wise = " << out[0].size()<<endl;

}
template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,


			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			unsigned W_BIT,
			unsigned M_BIT,
			
			unsigned SIMD,
			unsigned PE
			>
void conv1x1_bn_act(
	stream<ap_uint<IN_BIT * IN_CH> >& in,
	const ap_uint<SIMD*W_BIT> weights[PE][((IN_CH)/SIMD)*(OUT_CH/PE)],
	const ap_int<INC_BIT> inc[PE][OUT_CH/PE],
	const ap_int<BIAS_BIT> bias[PE][OUT_CH/PE],
	stream<ap_uint<OUT_BIT*OUT_CH> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;
	stream<ap_uint<SIMD*IN_BIT> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT, OUT_ROW*OUT_COL>(in, adj_out, reps);
	// 矩阵向量计算
	stream<ap_uint<PE*OUT_BIT> > mvau_out("mvau_out");
	matrix_vector_act_unit_no<IN_CH, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE,OUT_ROW*OUT_COL>
	(adj_out, weights, inc, bias, mvau_out, reps);

	StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW * OUT_COL>(mvau_out, out, reps);
}
//conv1x1_dp
template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned W_BIT,
			unsigned M_BIT,

			unsigned PE,
			unsigned S
			>
void conv1x1_dp(
	stream<ap_uint<IN_BIT>>  in[IN_CH],//输入是IN_CH个变化后的单通道矩阵
	const ap_uint<W_BIT> weights[PE][OUT_CH*IN_CH/PE],//权重理论上是1*1*IN_CH*OUT_CH
	stream<ap_uint<PE*OUT_BIT>> &out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
	const unsigned reps = 1
	)
{
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;
	ap_uint<IN_BIT * IN_CH> in_temp1;
	stream<ap_uint<IN_BIT * IN_CH>> in_temp ;
	//把IN_CH通道变回来
	for(int i = 0; i < IN_ROW; i++){
		for(int j = 0;j <IN_COL;j++){
			in_temp1 = 0;
			for(int k = 0; k <IN_CH;k++){
					//cout << bitset<IN_BIT>(in[k].read()) <<endl;
				 in_temp1 = in_temp1 << IN_BIT;
				 in_temp1(IN_BIT-1,0) = in[k].read();

				 cout << bitset<IN_CH*IN_BIT>(in_temp1)<<endl;


			}
			in_temp.write(in_temp1);
			//cout << bitset<IN_CH*IN_BIT>(in_temp.read())<<endl;
		}
	}

	stream<ap_uint<IN_BIT> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, IN_BIT, OUT_ROW*OUT_COL>(in_temp, adj_out, reps);

	matrix_vector_unit<IN_CH, OUT_CH, IN_BIT, W_BIT, M_BIT, 1, PE, OUT_ROW*OUT_COL>
	(adj_out, weights, out, reps);

}
/**
 * 卷积计算单元 同时计算bn_层与激活层
 * 在矩阵向量计算后立即计算得到激活输出
 * 只计算 1x1 的卷积 K = 1, P = 1 S = 1
 */
template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,


			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			unsigned W_BIT,
			unsigned M_BIT,

			unsigned PE
			>
void conv1x1_dp_bn_linear(
	stream<ap_uint<IN_BIT>>  in[IN_CH],//输入是IN_CH个变化后的单通道矩阵
	const ap_uint<W_BIT> weights[PE][OUT_CH*IN_CH/PE],//权重理论上是1*1*IN_CH*OUT_CH
	const ap_int<INC_BIT> inc[PE][OUT_CH/PE],
	const ap_int <BIAS_BIT> bias[PE][OUT_CH/PE],
	stream<ap_uint<OUT_CH*OUT_BIT>> &out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
	const unsigned reps = 1
	)
{
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;
	ap_uint<IN_BIT * IN_CH> in_temp1;
	stream<ap_uint<IN_BIT * IN_CH>> in_temp ;
	//cout << "IN_ROW" << IN_ROW << endl;
	//把IN_CH通道变回来
	for(int i = 0; i < IN_ROW; i++){
		for(int j = 0;j <IN_COL;j++){
			in_temp1 = 0;
			for(int k = 0; k <IN_CH;k++){
				 //cout << bitset<IN_BIT>(in[k].read()) <<endl;
				 in_temp1 = in_temp1 << IN_BIT;
				 in_temp1(IN_BIT-1,0) = in[k].read();

				 //cout << bitset<IN_CH*IN_BIT>(in_temp1)<<endl;


			}
			in_temp.write(in_temp1);
			//cout << bitset<IN_CH*IN_BIT>(in_temp.read())<<endl;
		}
	}

	stream<ap_uint<IN_BIT> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, IN_BIT, OUT_ROW*OUT_COL>(in_temp, adj_out, reps);

	//	cout << " test " << bitset<IN_BIT>(adj_out.read()) << endl;

	stream<ap_uint<PE*OUT_BIT> > mvau_out("mvau_out");
	//cout << "conv 1x1 devide"<<endl;
	matrix_vector_linear<IN_CH, OUT_CH, IN_BIT, OUT_BIT,W_BIT, M_BIT,INC_BIT,BIAS_BIT, 1, PE, OUT_ROW*OUT_COL> (adj_out,weights,inc,bias,mvau_out,reps);
	//cout << " test " << bitset<PE*OUT_BIT>(mvau_out.read()) << endl;

	//cout<<mvau_out.size()<<endl;

	StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW * OUT_COL>(mvau_out, out, reps);
	//cout<<"out read = "<< bitset<OUT_CH*OUT_BIT>(out.read())<<endl;
}

//conv1x1 带激活层
template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,


			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			unsigned W_BIT,
			unsigned M_BIT,

			unsigned PE
			>
void conv1x1_dp_bn_act(
	stream<ap_uint<IN_BIT>>  in[IN_CH],//输入是IN_CH个变化后的单通道矩阵
	const ap_uint<W_BIT> weights[PE][OUT_CH*IN_CH/PE],//权重理论上是1*1*IN_CH*OUT_CH
	const ap_int<INC_BIT> inc[PE][OUT_CH/PE],
	const ap_int <BIAS_BIT> bias[PE][OUT_CH/PE],
	stream<ap_uint<OUT_CH*OUT_BIT>> &out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
	const unsigned reps = 1
	)
{
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;
	ap_uint<IN_BIT * IN_CH> in_temp1;
	stream<ap_uint<IN_BIT * IN_CH>> in_temp ;
	//cout << "IN_ROW" << IN_ROW << endl;
	//把IN_CH通道变回来
	for(int i = 0; i < IN_ROW; i++){
		for(int j = 0;j <IN_COL;j++){
			in_temp1 = 0;
			for(int k = 0; k <IN_CH;k++){
				 //cout << bitset<IN_BIT>(in[k].read()) <<endl;
				 in_temp1 = in_temp1 << IN_BIT;
				 in_temp1(IN_BIT-1,0) = in[k].read();

				 //cout << bitset<IN_CH*IN_BIT>(in_temp1)<<endl;


			}
			in_temp.write(in_temp1);
			//cout << bitset<IN_CH*IN_BIT>(in_temp.read())<<endl;
		}
	}

	stream<ap_uint<IN_BIT> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, IN_BIT, OUT_ROW*OUT_COL>(in_temp, adj_out, reps);

	//	cout << " test " << bitset<IN_BIT>(adj_out.read()) << endl;

	stream<ap_uint<PE*OUT_BIT> > mvau_out("mvau_out");
	cout << "conv 1x1 devide"<<endl;
	matrix_vector_act_unit_no<IN_CH, OUT_CH, IN_BIT, OUT_BIT,W_BIT, M_BIT,INC_BIT,BIAS_BIT, 1, PE, OUT_ROW*OUT_COL> (adj_out,weights,inc,bias,mvau_out,reps);
	//cout << " test " << bitset<PE*OUT_BIT>(mvau_out.read()) << endl;

	//cout<<mvau_out.size()<<endl;

	StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW * OUT_COL>(mvau_out, out, reps);
	//cout<<"out read = "<< bitset<OUT_CH*OUT_BIT>(out.read())<<endl;
}




//depth_wise 深度可分离卷积模块
template<
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned W_BIT,
			unsigned M_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,

			unsigned PE,
			unsigned S
			>
void conv_dp(
			stream<ap_uint<IN_BIT * IN_CH> >& in,
		    const ap_uint<W_BIT> weights_dp3[IN_CH][9],//define weights_DP
		    const ap_uint<W_BIT> weights_dp1[PE][OUT_CH*IN_CH/PE],
		    const ap_int<INC_BIT> inc_3[1][IN_CH],// 这里是否应该全是PE
		    const ap_int<BIAS_BIT> bias_3[1][IN_CH],
		    const ap_int<INC_BIT> inc_1[PE][OUT_CH/PE],//d
		    const ap_int<BIAS_BIT> bias_1[PE][OUT_CH/PE],
			stream<ap_uint<OUT_CH*(OUT_BIT)>> &out, //输出是OUT_CH*IN_ROW*IN_COL
			const unsigned reps = 1,
			bool linear = false
			)
{

		stream<ap_uint<OUT_BIT>> conv_out[IN_CH];//存放深度卷积之后出来的那一个个小的
		stream<ap_uint<OUT_CH*(OUT_BIT+W_BIT)>> out_bc;
		static_assert((S == 1 || S == 2),"S is not allowed!");
		conv3x3_dp_bn_act<IN_ROW,IN_COL,IN_CH,IN_BIT,OUT_BIT,INC_BIT,BIAS_BIT, W_BIT, M_BIT,S> (in,weights_dp3,inc_3,bias_3,conv_out,1);
		cout << "conv3x3 size " << conv_out[0].size() << endl;
		/*
		for(int k =0;k< IN_CH;k++){
			for(int i = 0;i < IN_ROW/S;i++){
				for(int j=0;j < IN_COL/S;j ++){

					cout << "conv_out " << bitset<OUT_BIT>(conv_out[k].read())<< endl;
				}
			}
		}
		*/
		if(linear == false){
			if(S == 1){
				conv1x1_dp_bn_act<IN_ROW,IN_COL,IN_CH,OUT_BIT,OUT_CH,OUT_BIT+W_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT+W_BIT,PE>(conv_out,weights_dp1,inc_1,bias_1,out_bc,1);

			}
			if(S == 2){
				conv1x1_dp_bn_act<IN_ROW/2,IN_COL/2,IN_CH,OUT_BIT,OUT_CH,OUT_BIT+W_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT+W_BIT,PE>(conv_out,weights_dp1,inc_1,bias_1,out_bc,1);
			}
		}
	    else{
	    	if(S == 1){
	    		conv1x1_dp_bn_linear<IN_ROW,IN_COL,IN_CH,OUT_BIT,OUT_CH,OUT_BIT+W_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT+W_BIT,PE>(conv_out,weights_dp1,inc_1,bias_1,out_bc,1);

	    	}
	    	if(S == 2){
	    		conv1x1_dp_bn_linear<IN_ROW/2,IN_COL/2,IN_CH,OUT_BIT,OUT_CH,OUT_BIT+W_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT+W_BIT,PE>(conv_out,weights_dp1,inc_1,bias_1,out_bc,1);
	    	}
	    	//conv1x1_dp_bn_linear<IN_ROW,IN_COL,IN_CH,OUT_BIT,OUT_CH,OUT_BIT+W_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT+W_BIT,PE>(conv_out,weights_dp1,inc_1,bias_1,out_bc,1);
	    }


	    ap_uint<OUT_CH*OUT_BIT> out_buffer;
	    ap_uint<OUT_BIT> out_buffer_var[OUT_CH];
	    ap_uint<(OUT_BIT+W_BIT)> out_temp_buffer[OUT_CH];
	    ap_uint<OUT_CH*(OUT_BIT+W_BIT)> out_buffer_all;
	    //这里可能有点问题
	    for(int i = 0;i < IN_ROW/S;i++){
	    	for(int j = 0;j <IN_COL/S;j++){
    	    	out_buffer_all = out_bc.read();
    	    	//cout <<"OUT_BUFFER "<< bitset<OUT_CH*(OUT_BIT)>(out_buffer_all)<<endl;
	    	    for(int k = 0;k < OUT_CH;k++){
	    	    					out_temp_buffer[k] = out_buffer_all(OUT_CH*(OUT_BIT+W_BIT)-1,OUT_CH*(OUT_BIT+W_BIT)-(OUT_BIT+W_BIT));
	    	    					//cout <<"is here" <<  out_temp_buffer[k]<<endl;

	    	    					out_buffer_var[k] = out_temp_buffer[k](OUT_BIT-1,0);

	    	    					out_buffer_all = out_buffer_all << OUT_BIT+W_BIT;
	    	    					//cout <<"OUT_BUFFER after "<< bitset<OUT_CH*(OUT_BIT)>(out_buffer_all)<<endl;
	    	    					out_buffer = out_buffer << OUT_BIT;

	    	    					out_buffer(OUT_BIT-1,0) = out_buffer_var[k];
	    	    			}
	    	    //cout << out_buffer << endl;
	    	    out.write(out_buffer);
	    	 }
	    }



}




//倒置残差模块
template<
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned W_BIT,
			unsigned M_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,

			unsigned expand_ratio,//t

			unsigned PE,
			unsigned S
			>
void Inverted(
			stream<ap_uint<IN_BIT * IN_CH> >& in,
			const ap_uint<W_BIT> weights_conv3[expand_ratio*IN_CH][9],//这里思考一下输入是多少,conv1*1 展开之后的应该是
			const ap_uint<W_BIT> weights_conv1_1[PE][expand_ratio*IN_CH*IN_CH/PE],//h*w*tk
			//const ap_uint<W_BIT> weights_conv1_2[PE][OUT_CH*IN_CH/PE],//权重理论上是1*1*IN_CH*OUT_CH //h*w*OUT_CH
			const ap_uint<W_BIT> weights_conv1_3[PE][expand_ratio*IN_CH*OUT_CH/PE],//权重理论上是1*1*IN_CH*OUT_CH h*w*OUT_CH 这个应该是深度可分离卷积的权重
			const ap_int<INC_BIT> inc_conv3[1][expand_ratio*IN_CH],
			const ap_int <BIAS_BIT> bias_conv3[1][expand_ratio*IN_CH],
			const ap_int<INC_BIT> inc_conv1_1[PE][expand_ratio*IN_CH/PE], //d
			const ap_int<BIAS_BIT> bias_conv1_1[PE][expand_ratio*IN_CH/PE],
			const ap_int<INC_BIT> inc_conv1_2[PE][OUT_CH/PE], //d
			const ap_int<BIAS_BIT> bias_conv1_2[PE][OUT_CH/PE],
			stream<ap_uint<OUT_CH*OUT_BIT>> &out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
			const unsigned reps = 1
			)
		{
			static_assert((S == 1 || S == 2),"S is not allowed!");
			unsigned const hidden_dim = IN_CH * expand_ratio;

			stream<ap_uint<hidden_dim*(OUT_BIT)>> out_temp;
			stream<ap_uint<OUT_CH*OUT_BIT>> out_shortcut;
			if(IN_CH != OUT_CH){//这里不用相加
				if(expand_ratio == 1 ){
					//这里的解决方法是什么，重新配置
					stream<ap_uint<hidden_dim* IN_BIT>> in_copy_3;
					StreamingDataWidthConverter_Batch<IN_BIT*IN_CH,hidden_dim*IN_BIT,IN_ROW*IN_COL>(in,in_copy_3,reps);
					conv_dp<IN_ROW,IN_COL,hidden_dim,IN_BIT,OUT_CH,OUT_BIT,W_BIT,M_BIT,INC_BIT,BIAS_BIT,PE,S>(in_copy_3,weights_conv3,weights_conv1_3,inc_conv3,bias_conv3,inc_conv1_2,bias_conv1_2,out,1,true);
					/*
					    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
	                	nn.BatchNorm2d(hidden_dim),
	                	nn.ReLU6(inplace=True),
	                    # pw-linear
	                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
	                    nn.BatchNorm2d(oup),
					 */
				}


			else {
						conv1x1_bn_act<IN_ROW,IN_COL,IN_CH,IN_BIT,hidden_dim,OUT_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT,1,PE>(in,weights_conv1_1,inc_conv1_1,bias_conv1_1,out_temp,1);//这里尼玛是个普通卷积
						//第一次的逐点卷积模块 普通1x1卷积 输入是H*W*IN_CH 输出应该是H*W*expand_ratio*IN_CH

						conv_dp<IN_ROW,IN_COL,hidden_dim,OUT_BIT,OUT_CH,OUT_BIT,W_BIT,M_BIT,INC_BIT,BIAS_BIT,PE,S>(out_temp,weights_conv3,weights_conv1_3,inc_conv3,bias_conv3,inc_conv1_2,bias_conv1_2,out,1,true);
						//接着直接跟一个深度可分离卷积模块
						/*
					        # pw
						    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
							nn.BatchNorm2d(hidden_dim),
							nn.ReLU6(inplace=True),
							# dw
							nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
							nn.BatchNorm2d(hidden_dim),
						    nn.ReLU6(inplace=True),
							# pw-linear
							nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
							nn.BatchNorm2d(oup),
				       */


					}
			}

		   if( S == 1 && IN_CH == OUT_CH){//这下要相加咯
			   // 定义FIFO
			   stream<ap_uint<IN_BIT * IN_CH> > in_copy_1;
			   stream<ap_uint<IN_BIT * IN_CH> > in_copy_2;

			   // 在处理数据前复制数据流
			   while(!in.empty()) {
				   ap_uint<IN_BIT * IN_CH> data = in.read();
			       in_copy_1.write(data);
			       in_copy_2.write(data);  //
			   }
			   	conv1x1_bn_act<IN_ROW,IN_COL,IN_CH,IN_BIT,hidden_dim,OUT_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT,1,PE>(in_copy_1,weights_conv1_1,inc_conv1_1,bias_conv1_1,out_temp,1);//这里尼玛是个普通卷积
								//第一次的逐点卷积模块 普通1x1卷积 输入是H*W*IN_CH 输出应该是H*W*expand_ratio*IN_CH

				conv_dp<IN_ROW,IN_COL,hidden_dim,OUT_BIT,OUT_CH,OUT_BIT,W_BIT,M_BIT,INC_BIT,BIAS_BIT,PE,S>(out_temp,weights_conv3,weights_conv1_3,inc_conv3,bias_conv3,inc_conv1_2,bias_conv1_2,out_shortcut,1,true);
								//接着直接跟一个深度可分离卷积模块

			    //out == in + short_cut
			   ap_uint<OUT_CH*OUT_BIT> temp_out;
			   for(int i = 0 ;i < IN_ROW;i++){
				   for(int j = 0; j<IN_COL;j++){

						   	   temp_out = adder<OUT_BIT,OUT_CH>(in_copy_2.read(),out_shortcut.read());//adder 函数用来相加
						   	   //cout << "problem" <<endl;
						   	   out.write(temp_out);

				   }
			   }
		   }

		}


// conv3*3 卷积 带归一化层和激活函数
template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned W_BIT,
			unsigned M_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,

			unsigned SIMD,
			unsigned PE,
			unsigned S>
void conv3x3_bn(
    stream<ap_uint<IN_BIT * IN_CH> >& in,
	const ap_uint<SIMD*W_BIT> weights[PE][((IN_CH*9)/SIMD)*(OUT_CH/PE)],
	const ap_int<INC_BIT> inc[PE][OUT_CH/PE],
	const ap_int<BIAS_BIT> bias[PE][OUT_CH/PE],
	stream<ap_uint<OUT_BIT*OUT_CH> >& out,
	const unsigned reps = 1
	)
{
#pragma HLS DATAFLOW

	const unsigned INTER_ROW = IN_ROW + 2;
	const unsigned INTER_COL = IN_COL + 2;
	// 暂时认为输入 输出维度不变

	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;

	//stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
	// StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
	// padding
	stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);
	cout << "the size after padding " << padding_out.size() << endl;
	// 滑动窗口 这下懂了
	stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
	SWU<3, S, INTER_ROW, INTER_COL, IN_CH, IN_BIT> (padding_out, swu_out, reps);
	cout <<"the size of the window " <<  swu_out.size() << endl;
	// 位宽调整
	stream<ap_uint<SIMD*IN_BIT> > adj_out("adj_out");
	if(S == 1){
		StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT, 9*OUT_ROW*OUT_COL>(swu_out, adj_out, reps);
	}
	if(S == 2){
		StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT, 9*OUT_ROW/2*OUT_COL/2>(swu_out, adj_out, reps);
	}
	cout<<"the size of the adjustment " << adj_out.size() <<  endl; //swu_out*3
	for(int i = 0; i < 9; i++)
	{
		for(int j = 0 ; j < 9; j++)
		{
			cout << swu_out.read() << " ";
		}
		cout << endl;
	}

	// 矩阵向量计算
	stream<ap_uint<PE*OUT_BIT> > mvau_out("mvau_out");
	if( S == 1 )
	{
		matrix_vector_act_unit_no<IN_CH*3*3, OUT_CH, IN_BIT, OUT_BIT,W_BIT, M_BIT,INC_BIT,BIAS_BIT, SIMD, PE, OUT_ROW*OUT_COL> (adj_out, weights,inc,bias,mvau_out,reps);
	//IN_ROW*IN_COL*OUT_CH if (s == 1) //这里出来的位宽不是OUT_CH * OUT_BIT 就是OUT_BIT
	}
	if(S == 2)
	{
		matrix_vector_act_unit_no<IN_CH*3*3, OUT_CH, IN_BIT, OUT_BIT,W_BIT, M_BIT,INC_BIT,BIAS_BIT, SIMD, PE, OUT_ROW/2*OUT_COL/2> (adj_out, weights,inc,bias,mvau_out,reps);
	}
	cout << " the size of the mvau " << mvau_out.size() << endl;
	if(S == 1){
		StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW * OUT_COL>(mvau_out, out, reps);
	}
	if( S == 2)
	{
		StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW /2* OUT_COL/2>(mvau_out, out, reps);
	}
	cout << " the size of the final out " << out.size() << endl;
}





/*
// 带PE单元的1x1卷积
template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,


			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			unsigned W_BIT,
			unsigned M_BIT,

			unsigned PE
			>
void conv1x1_dp_bn_act_PE(
	stream<ap_uint<IN_BIT>>  in[IN_CH],//输入是IN_CH个变化后的单通道矩阵
	const ap_uint<W_BIT> weights[PE][OUT_CH*IN_CH/PE],//权重理论上是1*1*IN_CH*OUT_CH
	const ap_int<INC_BIT> inc[PE][OUT_CH/PE],
	const ap_int <BIAS_BIT> bias[PE][OUT_CH/PE],
	stream<ap_uint<PE*OUT_BIT>> &out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
	const unsigned reps = 1
	)
{
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;
	ap_uint<IN_BIT * IN_CH> in_temp1;
	stream<ap_uint<IN_BIT * IN_CH>> in_temp ;
	//把IN_CH通道变回来
	for(int i = 0; i < IN_ROW; i++){
		for(int j = 0;j <IN_COL;j++){
			in_temp1 = 0;
			for(int k = 0; k <IN_CH;k++){
					//cout << bitset<IN_BIT>(in[k].read()) <<endl;
				 in_temp1 = in_temp1 << IN_BIT;
				 in_temp1(IN_BIT-1,0) = in[k].read();

				 //cout << bitset<IN_CH*IN_BIT>(in_temp1)<<endl;


			}
			in_temp.write(in_temp1);
			//cout << bitset<IN_CH*IN_BIT>(in_temp.read())<<endl;
		}
	}

	stream<ap_uint<IN_BIT> > adj_out("adj_out");
	StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, IN_BIT, OUT_ROW*OUT_COL>(in_temp, adj_out, reps);
	matrix_vector_act_unit_no<IN_CH, OUT_CH, IN_BIT, OUT_BIT,W_BIT, M_BIT,INC_BIT,BIAS_BIT, 1, PE, OUT_ROW*OUT_COL> (adj_out,weights,inc,bias,out,reps);




}
//conv3*3 卷积

template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT,
			unsigned W_BIT,
			unsigned M_BIT,

			unsigned SIMD,
			unsigned PE,
			unsigned S>
void conv3x3(
	stream<ap_uint<IN_BIT * IN_CH> >& in,
	const ap_uint<SIMD*W_BIT> weights[PE][((IN_CH*9)/SIMD)*(OUT_CH/PE)],
	stream<ap_uint<OUT_BIT*OUT_CH> >& out,
	const unsigned reps = 1
	)
{
#pragma HLS DATAFLOW

	const unsigned INTER_ROW = IN_ROW + 2;
	const unsigned INTER_COL = IN_COL + 2;
	// 暂时认为输入 输出维度不变

	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;

	//stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
	// StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
	// padding
	stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);
	cout << "the size after padding " << padding_out.size() << endl;
	// 滑动窗口 这下懂了
	stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
	SWU<3, S, INTER_ROW, INTER_COL, IN_CH, IN_BIT> (padding_out, swu_out, reps);
	cout <<"the size of the window " <<  swu_out.size() << endl;

	// 位宽调整
	stream<ap_uint<SIMD*IN_BIT> > adj_out("adj_out");
	if(S == 1){
			StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT, 9*OUT_ROW*OUT_COL>(swu_out, adj_out, reps);
	}
	if(S == 2){
			StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT, 9*OUT_ROW/2*OUT_COL/2>(swu_out, adj_out, reps);
	}

	cout<<"the size of the adjustment " << adj_out.size() <<  endl; //swu_out*3
	/*for(int i = 0; i < 9; i++)
	{
		for(int j = 0 ; j < 9; j++)
		{
			cout << swu_out.read() << " ";
		}
		cout << endl;
	}
	// 矩阵向量计算
	stream<ap_uint<PE*OUT_BIT> > mvau_out("mvau_out");
	if(S==1){
		matrix_vector_unit<IN_CH*3*3, OUT_CH, IN_BIT, W_BIT, M_BIT, SIMD, PE, OUT_ROW*OUT_COL> (adj_out, weights,mvau_out, reps);//IN_ROW*IN_COL*OUT_CH if (s == 1) //这里出来的位宽不是OUT_CH * OUT_BIT 就是OUT_BIT
	}
	if(S==2){
			matrix_vector_unit<IN_CH*3*3, OUT_CH, IN_BIT, W_BIT, M_BIT, SIMD, PE, OUT_ROW/2*OUT_COL/2> (adj_out, weights,mvau_out, reps);//IN_ROW*IN_COL*OUT_CH if (s == 1) //这里出来的位宽不是OUT_CH * OUT_BIT 就是OUT_BIT
	}

	cout << " the size of the mvau " << mvau_out.size() << endl;
	if(S == 1)
	{
		StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW * OUT_COL>(mvau_out, out, reps);
	}
	if (S == 2)
	{
		StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW/2* OUT_COL/2>(mvau_out, out, reps);
	}
	cout << " the size of the final out " << out.size() << endl;
}
*/




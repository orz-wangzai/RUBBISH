#include <stdint.h>
#include <ap_int.h>
#include "stream_tools.h"
#include "conv2d.h"

#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "function.h"

#define IN_ROW      2
#define IN_COL      2
#define IN_CH       3

#define IN_BIT      4
#define W_BIT       4
#define INC_BIT     4
#define BIAS_BIT    4
#define M_BIT       4
#define OUT_BIT     4

#define OUT_CH      9
#define PE          1
#define SIMD        1

#define S           2



int main(int argc, char const *argv[])
{
    hls::stream<ap_uint<IN_CH*IN_BIT>> in("in"); //数据流
    for (int i=0; i < IN_ROW; i ++) {
        for (int j=0; j < IN_COL; j ++) {
            // data[i][j] = i * 400 + j + 1;
        	//i * IN_COL + j + 1
            in.write(273);
            //cout << bitset<IN_BIT>(in.read()) << " ";
        }
        cout << endl;
    }
    /*test padding and swu
    hls::stream<ap_uint<IN_CH*IN_BIT>> padding_out("padding_out");
    padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, 1);
    cout << padding_out.size() << endl;

    hls::stream<ap_uint<IN_CH*IN_BIT>> swu_out("swu_out");
    SWU<3, S, IN_ROW + 2, IN_COL + 2, IN_CH, IN_BIT>(padding_out, swu_out, 1);
    cout<< swu_out.size() << endl;
    for(int i=0; i < 4; i ++) {
            for (int j=0; j < 9; j ++) {
                cout << " " << swu_out.read();
            }
            cout << endl;
        }
	*/
    unsigned const expand_ratio = 2;
    ap_uint<SIMD*W_BIT> weights[PE][((IN_CH*9)/SIMD)*(OUT_CH/PE)];

    ap_uint<W_BIT> weights_dp3[IN_CH][9]; //define weights_DP

    ap_uint<W_BIT> weights_dp1_1[PE][expand_ratio*IN_CH*IN_CH/PE];//
    ap_uint<W_BIT> weights_dp1_2[PE][OUT_CH*IN_CH/PE];//h*w*tk
    ap_uint<W_BIT> weights_dp1_3[PE][expand_ratio*OUT_CH*IN_CH/PE];

    ap_int<INC_BIT> inc_conv3[1][expand_ratio*IN_CH]; //conv3x3 incs
    ap_int<BIAS_BIT> bias_3[1][expand_ratio*IN_CH];//conv3x3 bias
    ap_int<INC_BIT> inc_pw_2[PE][OUT_CH/PE]; //conv1x1 dp inc
    ap_int<BIAS_BIT> bias_pw_2[PE][OUT_CH/PE];//conv1x1 dp inc
    ap_int<INC_BIT> inc_pw_1[PE][expand_ratio*IN_CH/PE]; //conv1x1 dp inc
    ap_int<BIAS_BIT> bias_pw_1[PE][expand_ratio*IN_CH/PE];
    ap_int<INC_BIT> inc_pw_3[PE][expand_ratio*IN_CH*OUT_CH/PE]; //d
    ap_int<BIAS_BIT> bias_pw_3[PE][expand_ratio*IN_CH*OUT_CH/PE];
    for(int k = 0; k < expand_ratio*IN_CH; k++){
    	inc_conv3[0][k]= 1;
    	bias_3[0][k] = 1;
    }
    for(int k = 0; k <PE; k++){
    		for(int i = 0; i < OUT_CH/PE; i++){
    			inc_pw_2[k][i]= 1;
    			bias_pw_2[k][i] = 1;
    		}
    }
    for(int k = 0; k <PE; k++){
    		for(int i = 0; i < expand_ratio*IN_CH/PE; i++){
        			inc_pw_1[k][i]= 1;
        			bias_pw_1[k][i] = 1;
        	}
    }
    for(int k = 0; k <PE; k++){
        		for(int i = 0; i < expand_ratio*IN_CH*OUT_CH/PE; i++){
            			inc_pw_3[k][i]= 1;
            			bias_pw_3[k][i] = 1;
            	}
        }
    //weights_DP for 3x3
    for(int i = 0;i < IN_CH;i ++){
    	for (int k = 0; k < 9; k++)
			{
				weights_dp3[i][k] = 1;
				cout << weights_dp3[i][k]<< " ";
				if(k == 8 || k == 17 || k==26)
				cout << endl;
			}
    }
    cout<< endl;
    //weights_DP for 1x1
    for(int i = 0;i < PE;i ++){
            for (int k = 0; k < expand_ratio*IN_CH*IN_CH; k++)
        		{
        			weights_dp1_1[i][k] = 1;
        		}
            }
     cout<< endl;

    //weights for pw
    //测试正常卷积
    for(int i = 0;i < PE;i ++){
         for (int k = 0; k < IN_CH*OUT_CH/PE; k++)
             {
             			weights_dp1_2[i][k] = 1;
             	}
    }
    cout<< endl;
    for(int i = 0;i < PE;i ++){
             for (int k = 0; k < expand_ratio*IN_CH*OUT_CH/PE; k++)
                 {
                 			weights_dp1_3[i][k] = 1;
                 	}
        }
        cout<< endl;
    for(int i = 0;i < PE;i ++){
        for (int k = 0; k < ((IN_CH*9)/SIMD)*(OUT_CH/PE); k++)
    		{
    			weights[i][k] = 1;
    			cout << weights[i][k]<< " ";
    			if(k == 8 || k == 17 || k==26)
    				cout << endl;
    		}
        }
     cout<< endl;
      hls::stream<ap_uint<OUT_CH*(OUT_BIT)>> conv_out("conv_out");


    Inverted<IN_ROW,IN_COL,IN_CH,IN_BIT,OUT_CH,OUT_BIT,W_BIT,M_BIT,INC_BIT,BIAS_BIT,expand_ratio,PE,S>
    		(
    			in,
				weights_dp3,
    			weights_dp1_1,//给pw使用的
    			//weights_dp1_2,//单独给expand_ratio == 1 dp使用的
				weights_dp1_3,//单独给expand_ratio ！=1 dp使用的
    			inc_conv3,
    			bias_3,
    			inc_pw_1, //d
    			bias_pw_1,
				inc_pw_2,
				bias_pw_2,
				//inc_pw_3,
				//bias_pw_3,
    			conv_out, //输出是OUT_CH IN_ROW*IN_COL*IN_CH
    			1
				);

    for(int i = 0; i < IN_ROW/S; i ++) {
          	for(int j = 0; j < IN_COL/S; j++)
          	{

          				//conv_buffer = conv_out.read();
          				//cout << " add1 && add2 "<<(conv_buffer)<<endl;
          				//conv_res = adder<OUT_BIT,OUT_CH>(conv_buffer,conv_buffer);
          				//cout << "res= "<<(conv_res)<<endl;
          				//cout << "res= "<<bitset<72>(conv_res)<<endl;
          				cout << (conv_out.read())<<" ";
          	}


    }
    return 0;

}

/*测试深度卷积
hls::stream<ap_uint<OUT_BIT>> out[IN_CH];
conv3x3_dp <IN_ROW,IN_COL,IN_CH,IN_BIT,OUT_BIT,W_BIT,M_BIT,S> (in,weights,out,1);
for(int k = 0; k < IN_CH;k++){
	for(int i = 0 ; i < 4;i++)
		{
			for(int j = 0 ; j < 4; j++){
				cout << out[k].read()<<" ";
			}
		}
	cout << endl;
}
*/

//检查一下为什么OUT_BIT和M_BIT 有点差别
//conv3x3_bn <IN_ROW,IN_COL,IN_CH,IN_BIT,OUT_CH,OUT_BIT,W_BIT,M_BIT,INC_BIT,BIAS_BIT,SIMD,PE,S> (in,weights,inc,bias,conv_out,1);
//conv 3x3 正确用法 注意内部已经有padding
//conv3x3<IN_ROW,IN_COL,IN_CH,IN_BIT,OUT_CH,OUT_BIT,W_BIT,M_BIT,SIMD,PE,S> (in,weights,conv_out,1);

//测试深度可分离模块的连接
/*
stream<ap_uint<OUT_BIT>> out[IN_CH];

conv3x3_dp_bn_act<IN_ROW,IN_COL,IN_CH,IN_BIT,OUT_BIT,INC_BIT,BIAS_BIT, W_BIT, M_BIT,S> (in,weights_dp3,inc_conv3,bias_3,out,1);

conv1x1_dp_bn_act<IN_ROW,IN_COL,IN_CH,OUT_BIT,OUT_CH,OUT_BIT+W_BIT,INC_BIT,BIAS_BIT,W_BIT,M_BIT+W_BIT,PE>(out,weights_dp1,inc_pw_1,bias_pw_1,conv_out,1);

//测试深度可分离模块
    conv_dp<IN_ROW,IN_COL,IN_CH,IN_BIT,OUT_CH,OUT_BIT,W_BIT,M_BIT,INC_BIT,BIAS_BIT,PE,S>
      (in,weights_dp3,weights_dp1,inc_conv3,bias_3,inc_pw_1,bias_pw_1,conv_out,1,true);//define weights_DP

      cout << "size = "<<conv_out.size() << endl;

      ap_uint<OUT_CH*(OUT_BIT)> conv_buffer;
      ap_uint<OUT_CH*(OUT_BIT)> conv_res;
      for(int i = 0; i < IN_ROW/S; i ++) {
      	for(int j = 0; j < IN_COL/S; j++)
      	{

      				//conv_buffer = conv_out.read();
      				//cout << " add1 && add2 "<<(conv_buffer)<<endl;
      				conv_res = adder<OUT_BIT,OUT_CH>(conv_buffer,conv_buffer);
      				//cout << "res= "<<(conv_res)<<endl;
      				//cout << "res= "<<bitset<72>(conv_res)<<endl;
      				cout << (conv_out.read())<<" ";


      	}
      	cout  << endl;
      }
   }
*/


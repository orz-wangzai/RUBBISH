// 

#define CONV_0_IFM_CH 3 
#define CONV_0_IFM_ROW 224
#define CONV_0_IFM_COL 224
#define CONV_0_OFM_CH 32 //第一层输出通道为32
#define CONV_0_PE 16 
#define CONV_0_IN_BIT 8 
#define CONV_0_OUT_BIT 4 
#define CONV_0_W_BIT 4 
#define CONV_0_INC_BIT  14
#define CONV_0_BIAS_BIT 26 
#define CONV_0_SIMD     3
#define CONV_0_S        2

//bottle neck 0 t c n s 1 16 1 1
#define Bottle_neck_0_IFM_ROW 112
#define Bottek_neck_0_IFM_COL 112
#define Bottle_neck_0_IFM_IN_CH 32
#define Bottle_neck_0_IN_BIT    4

#define Bottle_neck_0_OFM_CH    16
#define Bottle_neck_0_OUT_BIT   4
 
#define Bottle_neck_0_W_BIT    4
#define Bottle_neck_0_INC_BIT 13
#define Bottle_neck_0_BIAS_BIT 21

#define Bottle_neck_0_expand_ratio 1
#define Bottle_neck_0_PE 8  //这个PE应该怎么设置啊
#define Bottle_neck_0_S 1
#define Bottle_neck_0_loop 1

//bottle neck 1 t c n s 6 24 2 2
#define Bottle_neck_1_IFM_ROW 112
#define Bottek_neck_1_IFM_COL 112
#define Bottle_neck_1_IFM_IN_CH 16
#define Bottle_neck_1_IN_BIT    4

#define Bottle_neck_1_OFM_CH    24
#define Bottle_neck_1_OUT_BIT   4
 
#define Bottle_neck_1_W_BIT    4
#define Bottle_neck_1_INC_BIT 13
#define Bottle_neck_1_BIAS_BIT 21

#define Bottle_neck_1_expand_ratio 6
#define Bottle_neck_1_PE 8 
#define Bottle_neck_1_loop 2 //这个PE应该怎么设置啊
#define Bottle_neck_1_S 2

//bottle neck 2 t c n s 6 32 3 2
#define Bottle_neck_2_IFM_ROW 56
#define Bottek_neck_2_IFM_COL 56
#define Bottle_neck_2_IFM_IN_CH 24
#define Bottle_neck_2_IN_BIT    4

#define Bottle_neck_2_OFM_CH    32
#define Bottle_neck_2_OUT_BIT   4
 
#define Bottle_neck_2_W_BIT    4
#define Bottle_neck_2_INC_BIT 13
#define Bottle_neck_2_BIAS_BIT 21

#define Bottle_neck_2_expand_ratio 6
#define Bottle_neck_2_PE 8 
#define Bottle_neck_2_loop 3 //这个PE应该怎么设置啊
#define Bottle_neck_2_S 2


//bottle neck 3 t c n s 6 64 4 2
#define Bottle_neck_3_IFM_ROW 28
#define Bottek_neck_3_IFM_COL 28
#define Bottle_neck_3_IFM_IN_CH 32
#define Bottle_neck_3_IN_BIT    4

#define Bottle_neck_3_OFM_CH    64
#define Bottle_neck_3_OUT_BIT   4
 
#define Bottle_neck_3_W_BIT    4
#define Bottle_neck_3_INC_BIT 13
#define Bottle_neck_3_BIAS_BIT 21

#define Bottle_neck_3_expand_ratio 6
#define Bottle_neck_3_PE 8 
#define Bottle_neck_3_loop 4 //这个PE应该怎么设置啊
#define Bottle_neck_3_S 2

//bottle neck 4 t c n s 6 96 3 1
#define Bottle_neck_4_IFM_ROW 14
#define Bottek_neck_4_IFM_COL 14
#define Bottle_neck_4_IFM_IN_CH 64
#define Bottle_neck_4_IN_BIT    4

#define Bottle_neck_4_OFM_CH    96
#define Bottle_neck_4_OUT_BIT   4
 
#define Bottle_neck_4_W_BIT    4
#define Bottle_neck_4_INC_BIT 13
#define Bottle_neck_4_BIAS_BIT 21

#define Bottle_neck_4_expand_ratio 6
#define Bottle_neck_4_PE 8 
#define Bottle_neck_4_loop 3 //这个PE应该怎么设置啊
#define Bottle_neck_4_S 1

//bottle neck 5 t c n s 6 160 3 2
#define Bottle_neck_5_IFM_ROW 14
#define Bottek_neck_5_IFM_COL 14
#define Bottle_neck_5_IFM_IN_CH 96
#define Bottle_neck_5_IN_BIT    4

#define Bottle_neck_5_OFM_CH    160
#define Bottle_neck_5_OUT_BIT   4
 
#define Bottle_neck_5_W_BIT    4
#define Bottle_neck_5_INC_BIT 13
#define Bottle_neck_5_BIAS_BIT 21

#define Bottle_neck_5_expand_ratio 6
#define Bottle_neck_5_PE 8 
#define Bottle_neck_5_loop 3 //这个PE应该怎么设置啊
#define Bottle_neck_5_S 2

//bottle neck 6 t c n s 6 320 1 1
#define Bottle_neck_6_IFM_ROW 7
#define Bottek_neck_6_IFM_COL 7
#define Bottle_neck_6_IFM_IN_CH 160
#define Bottle_neck_6_IN_BIT    4

#define Bottle_neck_6_OFM_CH    320
#define Bottle_neck_6_OUT_BIT   4
 
#define Bottle_neck_6_W_BIT    4
#define Bottle_neck_6_INC_BIT 13
#define Bottle_neck_6_BIAS_BIT 21

#define Bottle_neck_6_expand_ratio 6
#define Bottle_neck_6_PE 8 
#define Bottle_neck_6_loop 1 //这个PE应该怎么设置啊
#define Bottle_neck_6_S 1


//conv1x1 pw 参数
#define CONV_9_IFM_ROW 7
#define CONV_9_IFM_COL 7
#define CONV_9_IFM_CH  320               
#define CONV_9_IN_BIT  4

#define CONV_9_OFM_CH  1280
#define CONV_9_OUT_BIT 4                 
#define CONV_9_INC_BIT 13
#define CONV_9_BIAS_BIT 21  
#define CONV_9_W_BIT 4

#define CONV_9_PE 8
    
//还有最后一层全连接层
                  
                  
                  

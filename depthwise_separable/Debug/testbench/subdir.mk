################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../conv_test.cpp 

OBJS += \
./testbench/conv_test.o 

CPP_DEPS += \
./testbench/conv_test.d 


# Each subdirectory must supply rules for building sources it contributes
testbench/conv_test.o: C:/Users/31392/Desktop/bachelor_thesis/My_code/conv_1/depthwise_separable/conv_test.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -DAESL_TB -D__llvm__ -D__llvm__ -IE:/Vivado/xilinx/Vivado/2018.3/include/etc -IE:/Vivado/xilinx/Vivado/2018.3/include/ap_sysc -IE:/Vivado/xilinx/Vivado/2018.3/include -IE:/Vivado/xilinx/Vivado/2018.3/win64/tools/auto_cc/include -IE:/Vivado/xilinx/Vivado/2018.3/win64/tools/systemc/include -IC:/Users/31392/Desktop/bachelor_thesis/My_code/conv_1 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



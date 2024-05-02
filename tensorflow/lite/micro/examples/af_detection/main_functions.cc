/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <iostream>
#include <fcntl.h>      // ÎÄŒþ¿ØÖÆ¶šÒå
#include <termios.h>    // PPSIX ÖÕ¶Ë¿ØÖÆ¶šÒå
#include <unistd.h>     // Unix ±ê×Œº¯Êý¶šÒå
#include <cstring>      // ×Ö·ûŽ®¹ŠÄÜ
#include <errno.h>      // ŽíÎóºÅ¶šÒå
#include <cstdlib>
#include <sys/time.h>
#include <string>
#include <sstream>
#include "tensorflow/lite/micro/examples/af_detection/main_functions.h"
// #include "tensorflow/lite/micro/examples/af_detection/detection_responder.h"
// #include "tensorflow/lite/micro/examples/af_detection/image_provider.h"
// #include "tensorflow/lite/micro/examples/af_detection/model_settings.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/af_detect_model_data.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
const char *write_buffer = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
int fd ;

}  // namespace
// return usec
long long get_timestamp(void)//获取时间戳函数
{
    long long tmp;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    tmp = tv.tv_sec;
    tmp = tmp * 1000 * 1000;
    tmp = tmp + tv.tv_usec;

    return tmp;
}
//usart
int set_interface_attribs(int fd, int speed) {
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return -1;
    }

    // ÉèÖÃ²šÌØÂÊ
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    // ÉèÖÃCSIZEÎªCS8£¬ŒŽ8Î»ÊýŸÝ³€¶È
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    // ÉèÖÃCLOCALºÍCREAD£¬Ê¹ÄÜœÓÊÕÆ÷ºÍ±ŸµØÄ£Êœ
    tty.c_cflag |= (CLOCAL | CREAD);

    // ÉèÖÃPARENBÎª0£¬ŒŽÎÞÐ£ÑéÎ»
    tty.c_cflag &= ~PARENB;

    // ÉèÖÃCSTOPBÎª0£¬ŒŽ1Î»Í£Ö¹Î»
    tty.c_cflag &= ~CSTOPB;

    // ÉèÖÃCRTSCTSÎª0£¬ŒŽ²»Ê¹ÓÃÓ²ŒþÁ÷¿ØÖÆ
    tty.c_cflag &= ~CRTSCTS;

    // ÉèÖÃICANONÎª0£¬ŒŽ·Ç¹æ·¶Ä£Êœ£¬ÕâÑùreadŸÍ²»»áÊÜÐÐ»º³åµÄÓ°Ïì
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // ÉèÖÃOPOSTÎª0£¬ŒŽœûÓÃÊä³öŽŠÀí
    tty.c_oflag &= ~OPOST;

    // ÉèÖÃICANONÎª0£¬ŒŽ·Ç¹æ·¶Ä£Êœ
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // ÉèÖÃVMINÎª1£¬VMAXÎª0£¬ÕâÑùreadŸÍ»áÒ»Ö±×èÈû£¬Ö±µœÓÐÊýŸÝ¿É¶Á
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return -1;
    }

    return 0;
}

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_af_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.

  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  //usart
  // fd = open("/dev/ttyS7", O_RDWR | O_NOCTTY);
  // if (fd == -1) {
  //     perror("open_port: Unable to open /dev/ttyS7");
  //}
  //if (set_interface_attribs(fd, B115200) == -1) {
  //     close(fd);
  // }
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  // if (kTfLiteOk !=
  //    GetImage(kNumCols, kNumRows, kNumChannels, input->data.f)) {
  //  MicroPrintf("Image capture failed.");
  // }
  //usart
  fd = open("/dev/ttyS7", O_RDWR | O_NOCTTY);
  if (fd == -1) {
       perror("open_port: Unable to open /dev/ttyS7");
  }
  if (set_interface_attribs(fd, B115200) == -1) {
        close(fd);
   }
  sleep(1);
  //get data from usart
  write_buffer = "begin";
  if (write(fd, write_buffer, strlen(write_buffer)) < 0) {
        perror("write");
        close(fd);
  }
  std::cout<<"send begin"<<std::endl;
  char read_buffer[9];
  float* test_data = input->data.f; 
  std::stringstream ss;
  for(int i=0; i <1250; i++){
	int flag = 0;
	while(true){
		memset(read_buffer, 0, sizeof(read_buffer));
		int n = read(fd, read_buffer, 8);
		read_buffer[n]='\0';
        	if (n < 0) {
            		perror("read");
            		close(fd);
        	}
		ss << read_buffer;
        	// std::cout << "Read " << n << " bytes: " << read_buffer << std::endl;
		for(int j=0; j<n; j++){
			if(read_buffer[j]==' '){
				flag = 1;
				ss >> test_data[i];
                                // std::cout<<"fdata"<<i<<": "<<test_data[i]<< std::endl;
				break;
			}
		}
		if(flag){
			break;
		}
	}
	MicroPrintf("fdata %d: %f",i,(double)test_data[i]);
	// std::cout<<"fdata: "<<test_data[i]<< std::endl;
  }
  MicroPrintf("data_get_over");
  long long start_time = get_timestamp();
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }
  long long end_time = get_timestamp();
  long long cost_time = end_time - start_time;
  MicroPrintf("start_time:%lld, end_time:%lld, cost_time:%lld \n", start_time,end_time, cost_time);
  // usart return ok
  write_buffer = "ok";
  if (write(fd, write_buffer, strlen(write_buffer)) < 0) {
        perror("write");
        close(fd);
  }
  // recvice 200 status
  memset(read_buffer, 0, sizeof(read_buffer));
  int n = read(fd, read_buffer, sizeof(read_buffer));
  if (n < 0) {
	    perror("read");
            close(fd);
  }
  // int status = atoi(read_buffer);
  // return ans and time
  TfLiteTensor* output = interpreter->output(0);
  // Process the inference results.
  double ch0_score = output->data.f[0];
  double ch1_score = output->data.f[1];
  char result[20];
  // RespondToDetection(person_score, no_person_score);
  MicroPrintf("score: ch0: %f , ch1:%f \n",ch0_score, ch1_score);
  // ss << cost_time;
  ss.clear();
  if(ch0_score>ch1_score){
	  ss <<"0,";
	  //strcpy(result,"0,");
	  //strcat(result,ltoa((long)cost_time)); 
  }
  else{
	  ss <<"1,";
	  //strcpy(result,"1,");
          //strcat(result,ltoa((long)cost_time));
  }
  ss << cost_time;
  ss >> result;
  std::cout<<"result" << result<<std::endl;
  if (write(fd, result, strlen(result)) < 0) {
        perror("write");
        close(fd);
  }
  close(fd);
  MicroPrintf("loop end");
}

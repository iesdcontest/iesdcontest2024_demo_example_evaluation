import argparse
import serial
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os


def txt_to_numpy(filename, row):
  file = open(filename)
  lines = file.readlines()
  datamat = np.arange(row, dtype=np.float64)
  row_count = 0
  for line in lines:
    line = line.strip().split(' ')
    datamat[row_count] = line[0]
    row_count += 1

  return datamat


def main():
  t = time.strftime('%Y-%m-%d_%H-%M-%S')
  if not os.path.exists('./log/'):
    os.makedirs('./log/')
  List = []
  resultList = []
  labelList = []
  timeList = []
  port = args.com  # set port number
  ser = serial.Serial(port=port, baudrate=args.baudrate, timeout=1)  # open the serial
  print(ser)
  f = open('./testset_index.txt', 'r')
  for line in f:
    List.append(line)

  for idx in tqdm(range(0, 2)):
    labelList.append(List[idx].split(',')[0])
    # load data from txt files and reshape to (1, 1, 1250, 1)
    testX = txt_to_numpy(args.path_data + List[idx].split(',')[1].strip(), 1250).reshape(1, 1, 1250, 1)
    # receive messages from serial port, the length is the number of bytes remaining in the input buffer
    for i in range(0, testX.shape[0]):
      # don't continue running the code until a "begin" is received, otherwise receive iteratively
      nus = 0
      while ser.in_waiting < 5:
        time.sleep(1)
        # print("w1")
        nus = nus + 1
        if nus > 15:
          recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
          print(recv)
          exit(1)
        pass
      # when receiving the code "begin", send the test data cyclically
      recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
      print(recv)
      # clear the input buffer
      ser.reset_input_buffer()
      # if True:
      if recv.strip() == 'begin':
        for j in range(0, testX.shape[1]):
          for k in range(0, testX.shape[2]):
            for l in range(0, testX.shape[3]):
              time.sleep(0.003)
              send_str = str(testX[i][j][k][l]) + ' '
              ser.write(send_str.encode(encoding='utf8'))

        # don't continue running the code until a "ok" is received
        while ser.in_waiting < 2:
          time.sleep(0.01)
          pass
        time.sleep(0.01)
        recv = ser.read(size=ser.in_waiting).decode(encoding='utf8')
        print(recv)
        ser.reset_input_buffer()
        if recv.strip() == 'ok':
          time.sleep(0.02)
          # send status 200 to the board
          send_str = '200 '
          ser.write(send_str.encode(encoding='utf8'))
          time.sleep(0.01)
        # receive results from the board, which is a string separated by commas
        while ser.in_waiting < 4:
          pass
        recv = ser.read(ser.in_waiting).decode(encoding='utf8')
        print(recv)
        ser.reset_input_buffer()
        # the format of recv is ['<result>','<dutation>']
        result = recv.split(',')[0]
        inference_latency = recv.split(',')[1]
        if result == '0':
          resultList.append('0')
        else:
          resultList.append('1')
        # inference latency in ms
        timeList.append(float(inference_latency) / 1000)

  total_time = sum(timeList)
  avg_time = np.mean(timeList)
  print(avg_time)
  C_labelList = np.array(labelList).astype(int)
  C_resultList = np.array(resultList).astype(int)
  C = confusion_matrix(C_labelList, C_resultList, labels=[0, 1])
  print(C)
  print(labelList)
  print(resultList)

  acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
  precision = C[1][1] / (C[1][1] + C[0][1])
  sensitivity = C[1][1] / (C[1][1] + C[1][0])
  FP_rate = C[0][1] / (C[0][1] + C[0][0])
  PPV = C[1][1] / (C[1][1] + C[1][0])
  NPV = C[0][0] / (C[0][0] + C[0][1])
  F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
  F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)

  print("\nacc: {},\nprecision: {},\nsensitivity: {},\nFP_rate: {},\nPPV: {},\nNPV: {},\nF1_score: {}, "
        "\nF_beta_score: {},\ntotal_time: {}ms,\n average_time: {}ms".format(acc, precision, sensitivity, FP_rate, PPV,
                                                                             NPV, F1_score, F_beta_score,
                                                                             total_time, avg_time))

  f = open('./log/2pack_log_{}.txt'.format(t), 'a')
  f.write("Accuracy: {}\n".format(acc))
  f.write("Precision: {}\n".format(precision))
  f.write("Sensitivity: {}\n".format(sensitivity))
  f.write("FP_rate: {}\n".format(FP_rate))
  f.write("PPV: {}\n".format(PPV))
  f.write("NPV: {}\n".format(NPV))
  f.write("F1_Score: {}\n".format(F1_score))
  f.write("F_beta_Score: {}\n".format(F_beta_score))
  f.write("Total_Time: {}ms\n".format(total_time))
  f.write("Average_Time: {}ms\n\n".format(avg_time))
  f.write(str(C) + "\n\n")
  f.close()


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  # 选择好数据串口的COM端口
  argparser.add_argument('--com', type=str, default='com7')
  argparser.add_argument('--baudrate', type=int, default=115200)
  argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
  args = argparser.parse_args()
  main()

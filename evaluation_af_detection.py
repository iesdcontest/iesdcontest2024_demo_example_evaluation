import argparse
import serial
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import struct


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


def float_to_hex(f):
    # return hex(struct.unpack('<I', struct.pack('<f', f))[0])
    return struct.pack('>f', float(f)).hex()


def hex_to_float(h):
    i = int(h, 16)
    return struct.unpack('<f', struct.pack('<I', i))[0]


def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])


def hex_to_double(h):
    i = int(h, 16)
    return struct.unpack('<d', struct.pack('<Q', i))[0]


def print_hex(bytes):
    l = [hex(int(i)) for i in bytes]
    print(" ".join(l))


import binascii


def main():
    t = time.strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    List = []
    resultList = []
    labelList = []
    timeList = []
    port = args.com  # set port number
    ser = serial.Serial(port=port, baudrate=args.baudrate, timeout=10)  # open the serial , timeout=1
    ser.set_buffer_size(rx_size=12800, tx_size=12800)
    print(ser)

    f = open('./testset_index.txt', 'r')
    for line in f:
        List.append(line)
    ofp = open(file='log/res_{}.txt'.format(t), mode='w')  # make a new log file
    for idx in tqdm(range(0, len(List))):
    # for idx in tqdm(range(0, 5)):
        labelList.append(List[idx].split(',')[0])
        testX = txt_to_numpy(args.path_data + List[idx].split(',')[1].strip(), 1250).reshape(1, 1, 1250, 1)
        testZ = np.arange(1250 + 1, dtype=np.int64)
        s = 1
        for i in range(0, testX.shape[0]):
            for j in range(0, testX.shape[1]):
                for k in range(0, testX.shape[2]):
                    for l in range(0, testX.shape[3]):
                        testZ[s] = int(float_to_hex(testX[i][j][k][0]), base=16)
                        s += 1
                        # print(k,":",testX[i][j][k][0])
        testW = np.asanyarray(testZ, dtype="uint32")
        ser.flushOutput()
        datalen = 1250
        testW[0] = (datalen << 16) | 0x55aa
        result = ser.write(testW)
        # ser.in_waiting()
        while ser.in_waiting < 5:
            pass
            time.sleep(0.01)
        recv = ser.read(8)
        ser.reset_input_buffer()

        # the format of recv is ['<result>','<dutation>']
        result = recv[3]
        if result == 0:
            resultList.append('0')
        else:
            resultList.append('1')

        # tm_start = recv[4] | (recv[5] << 8) | (recv[6] << 16) | (recv[7] << 24)
        # tm_end = recv[8] | (recv[9] << 8) | (recv[10] << 16) | (recv[11] << 24)
        # s_tm_start = str(hex(tm_start))
        # s_tm_end = str(hex(tm_end))
        # f_tm_start = hex_to_float(s_tm_start)
        # f_tm_end = hex_to_float(s_tm_end)
        # print("f_tm_start",f_tm_start,"f_tm_end",f_tm_end)
        tm_cost = recv[4] | (recv[5] << 8) | (recv[6] << 16) | (recv[7] << 24)
        s_tm_cost = str(hex(tm_cost))
        f_tm_cost = hex_to_float(s_tm_cost)
        print("f_tm_cost", f_tm_cost)
        # f_tm = f_tm_cost / 1000.0

        # inference latency in ms
        timeList.append(f_tm_cost)
        ofp.write(str(result) + ' ' + str(f_tm_cost) + ' \r')
    ofp.close()
    C_labelList = np.array(labelList).astype(int)
    C_resultList = np.array(resultList).astype(int)
    C = confusion_matrix(C_labelList, C_resultList, labels=[0, 1])
    print(C)

    total_time = sum(timeList)
    avg_time = np.mean(timeList)
    acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
    precision = C[1][1] / (C[1][1] + C[0][1])
    sensitivity = C[1][1] / (C[1][1] + C[1][0])
    FP_rate = C[0][1] / (C[0][1] + C[0][0])
    PPV = C[1][1] / (C[1][1] + C[1][0])
    NPV = C[0][0] / (C[0][0] + C[0][1])
    F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)

    print("\nacc: {},\nprecision: {},\nsensitivity: {},\nFP_rate: {},\nPPV: {},\nNPV: {},\nF1_score: {}, "
          "\nF_beta_score: {},\ntotal_time: {}ms,\n average_time: {}ms".format(acc, precision, sensitivity, FP_rate,
                                                                               PPV,
                                                                               NPV, F1_score, F_beta_score,
                                                                               total_time, avg_time))

    f = open('./log/log_{}.txt'.format(t), 'a')
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

    return 0


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--com', type=str, default='com7')
    argparser.add_argument('--baudrate', type=int, default=115200)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    args = argparser.parse_args()
    main()

/*
 * @Author: yehongcen 
 * @Date: 2022-11-20 15:32:26 
 * @Last Modified by:   yehongcen 
 * @Last Modified time: 2022-11-20 15:32:26 
 */
#ifndef NNET_DATA_H
#define NNET_DATA_H

#include "nnet.h"

const GRULayer gru1;
const GRULayer gru2;
const GRULayer gru3;
const DenseLayer fc1;
const DenseLayer fc2;
const DenseLayer fc3;
const DenseLayer fc4;

typedef struct RNNState {
  float gru1_state[64];
  float gru2_state[64];
  float gru3_state[64];
  short seqnum;
} RNNState;

#endif
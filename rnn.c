/*
 * @Author: yehongcen 
 * @Date: 2022-11-20 15:29:41 
 * @Last Modified by:   yehongcen 
 * @Last Modified time: 2022-11-20 15:29:41 
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "nnet.h"
#include "nnet_data.h"
#include <stdio.h>
#define GRUNUM 64

void test_rnn(RNNState *st, float *input, float *output)
{
  float fc1_state[32], fc2_state[64], fc3_state[32], fc4_state[10];
  compute_dense(&fc1, fc1_state, input);
  compute_dense(&fc2, fc2_state, fc1_state);
  compute_gru(&gru1, st->gru1_state, fc2_state);
  compute_gru(&gru2, st->gru2_state, st->gru1_state);
  compute_gru(&gru3, st->gru3_state, st->gru2_state);
  compute_dense(&fc3, fc3_state, st->gru3_state);
  compute_dense(&fc4, fc4_state, fc3_state);
  for (int i = 0; i < 10; i++)
  {
    output[i] = fc4_state[i];
  }
  st->seqnum++;
  if(st->seqnum == 8)
  {
    st->seqnum = 0;
    for(int i = 0;i < GRUNUM;i++)
    {
      st->gru1_state[i] = 0;
      st->gru2_state[i] = 0;
      st->gru3_state[i] = 0;
    }
  }
}

int rnn_init(RNNState *st)
{
  for (int i = 0; i < GRUNUM; i++)
  {
    st->gru1_state[i] = 0;
    st->gru2_state[i] = 0;
    st->gru3_state[i] = 0;
  }
  st->seqnum = 0;
  return 0;
}



int main()
{
  FILE *ft, *fs;
  RNNState rnn_st;
  rnn_init(&rnn_st);
  ft = fopen("bone975.txt", "rb");
  fs = fopen("test975.txt", "wb");
  if ( (ft==NULL) ) {printf("Error opening filebone!\n");  exit(0);}
  if ( (fs==NULL) ) {printf("Error opening filetest!\n");  exit(0);}
  float bone[10];
  float out[10];
  double tmp[10];

  for (int i = 0; i < 973; i++)
  {
    for (int j = 0; j < 10; j++)
    {
      fscanf(ft, "%le", &(tmp[j]));
      bone[j] = (float)tmp[j];
    }
    
    test_rnn(&rnn_st, bone, out);
    for (int k = 0; k < 10; k++)
    {
      fprintf(fs, "%lf ", out[k]);
    }
    fprintf(fs,"\n");
  }
  // 0.2610, 0.5253, 0.7210, 0.9499, 0.0000, 1.4978, 1.8588, 2.0757,2.5317, 2.8552
  fclose(fs);
  fclose(ft);
  return 0;
}

/* Copyright (c) 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

void test_rnn(RNNState *st,float *input,float *output)
{
  float fc1_state[32],fc2_state[64],fc3_state[32],fc4_state[10];
  compute_dense(&fc1,fc1_state,input);
  compute_dense(&fc2,fc2_state,fc1_state);
  compute_gru(&gru1,st->gru1_state,fc2_state);
  compute_gru(&gru2,st->gru2_state,st->gru1_state);
  compute_gru(&gru3,st->gru3_state,st->gru2_state);
  compute_dense(&fc3,fc3_state,st->gru3_state);
  compute_dense(&fc4,fc4_state,fc3_state);
  for(int i = 0;i < 10;i++)
  {
    output[i] = fc4_state[i];
  }
}

int rnn_init(RNNState *st)
{
  for(int i = 0;i < GRUNUM;i++)
  {
    st->gru1_state[i] = 0;
    st->gru2_state[i] = 0;
    st->gru3_state[i] = 0;
  }
  return 0;
}

int main()
{
  RNNState rnn_st;
  rnn_init(&rnn_st);
  float bone[10] = {0.21985416516793296,0.39853800022659486,0.66326318957104513,0.93804336219668305,1.0926980060584051,1.5113391609445703,1.8404661702146814,2.1760780071360513,2.5377066115782325,2.8023390241110175};
  float out[10];
  //0.2610, 0.5253, 0.7210, 0.9499, 0.0000, 1.4978, 1.8588, 2.0757,2.5317, 2.8552
  test_rnn(&rnn_st,bone,out);
  for(int i = 0;i < 10;i++)
  {
    printf("%lf\n",out[i]);
  }
  return 0;
}

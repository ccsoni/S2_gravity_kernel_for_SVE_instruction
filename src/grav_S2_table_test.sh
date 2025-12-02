#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=eap-small"
#PJM -L "elapse=05:00"

export OMP_NUM_THREADS=1

./grav_S2_table_test

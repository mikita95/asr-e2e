#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:./asr/

/home/nikita/anaconda3/bin/python \
                ./asr/utils/data/examples/dataprocessor.py \
                --writer_config=./asr/resources/ctc/configs/data_config.ini \
                --record_path=./asr/resources/ctc/data/mfcc_18_ctc/val.tfrecord \
                --data_dir=/home/nikita/Development/validation

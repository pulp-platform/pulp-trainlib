#!/bin/bash

# priority	                              timeout       out_xml           proc  in_yaml
nice -n10 python sw/bwruntest.py --report-junit -t 1800 --yaml -o ne16_tests.xml -p 32 ./basic.yml

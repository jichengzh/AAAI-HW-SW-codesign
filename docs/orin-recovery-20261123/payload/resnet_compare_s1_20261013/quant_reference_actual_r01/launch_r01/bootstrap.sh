set +e
/home/jichengzhi/miniforge3/bin/python3.12 /home/jichengzhi/resnet_compare_s1_20261013/quant_reference_actual_r01/supervisor_reference_r02.py supervise --job /home/jichengzhi/resnet_compare_s1_20261013/quant_reference_actual_r01/reference_job_r02.json --expected-job-sha256 467fcfba89d6df6b2b5c47570efde5d530af3c126146c021afaa1292ee02637f --evidence /home/jichengzhi/resnet_compare_s1_20261013/quant_reference_actual_r01/execution_r01
code=$?
printf '{"supervisor_exit_code":%d}\n' "$code" > /home/jichengzhi/resnet_compare_s1_20261013/quant_reference_actual_r01/launch_r01/supervisor_exit.json
exit "$code"

#!/bin/sh
PID=$$
echo $PID
MESSAGE="[$USER] Processus $PID uses a GPU. If you need a GPU, please kill it with : kill -TERM -$PID."
curl -X POST -H 'Content-type: application/json' --data  "{\"text\":\"$MESSAGE\"}"  https://hooks.slack.com/services/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

PYTHONPATH="/home/sequel/ncarrara/phd_code"
N=0
while [ "0"="0" ]
do
  seed=$(($(date +%s%N)/1000000))
  seed=${seed#"15480"}
  echo $seed
  python main.py config/camera_ready_3.json $seed 1
done
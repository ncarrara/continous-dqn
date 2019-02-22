#!/bin/bash
echo $BASHPID
#for i in $(seq $1 $2)
for i in 1000 1001
do
     echo "Seed $i"
     python egreedy/main_egreedy.py config/slot_filling/camera_ready.json $i 1
     case $? in
        139)
            echo "erreur 139 on seed $i"
            ;;
        134)
            echo "erreur 134 on seed $i"
     esac
done
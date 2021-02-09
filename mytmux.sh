session="CCWSN_session" 
node_dir="/tmp/CCWSN_RL/"
node=""

[[ -d $node_dir ]] && echo -e "\e[31m" $node_dir ": directory exists\e[0m" && exit  

for i in $(cat ./config_files/nodes.txt); do
	mkdir -p ${node_dir}node_$i; 
	cp ./code/* ${node_dir}node_$i; 
	cp ./config_files/config_ccwsn.txt ${node_dir}node_$i;
	echo $i >> ${node_dir}node_$i/config_ccwsn.txt;
done

tmux new-session -d -s ${session} bash

for i in $(cat ./config_files/nodes.txt); do
	tmux split-window -h -t ${session} -c ${node_dir}node_$i bash; 
	tmux send-keys 'PS1='$i'": \n$ "; clear ' Enter;
	tmux select-layout tiled; 
done

tmux kill-pane -t ${session}:0.0 
tmux select-layout tiled 

node="" 
for _line in $(cat ./config_files/data.txt); do
	if [[ ${_line:0:1} = "[" ]] 
	then 
		node=${_line}
	else
		echo ${_line} >> ${node_dir}node_${node}/data.txt 
	fi
done

node="" 
for _line in $(cat ./config_files/interest.txt); do
	if [[ ${_line:0:1} = "[" ]]
	then 
		node=${_line}
	else
		echo ${_line} >> ${node_dir}node_${node}/interest.txt 
	fi
done 


#for _pane in $(tmux list-panes -F '#P' -t ${session}:0); do
#	tmux send-keys -t ${session}:0.${_pane} 'echo ' ${_pane} Enter; 
#done

for _pane in $(tmux list-panes -F '#P' -t ${session}:0); do
	tmux send-keys -t ${session}:0.${_pane} 'docker run -it --rm -v $PWD:/home pytorch/pytorch python3 /home/CCWSN.py' Enter; 
done

tmux attach -t ${session}


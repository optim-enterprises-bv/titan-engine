# build v2 then gate1 MTP=2 and MTP off
source $HOME/titan-engine/m4/pc/lib.sh
build || exit 1
bash $P/s_g1m2.sh
bash $P/s_g1m0.sh

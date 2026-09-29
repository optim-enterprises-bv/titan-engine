# deploy the window-2 binary (unit files already point at bin/mistralrs-titan-pc)
cp -a $HOME/titan-engine/target-pc-oxide/release/mistralrs $HOME/titan-engine/bin/mistralrs-titan-pc && ls -la $HOME/titan-engine/bin/mistralrs-titan-pc
touch $HOME/titan-engine/m4/pc/q/end

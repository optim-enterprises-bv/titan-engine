# deploy: bin/mistralrs-titan-pc, both unit files point at it, daemon-reload (the window's trap starts it)
set -u
E=$HOME/titan-engine
cp -a $E/target-pc-oxide/release/mistralrs $E/bin/mistralrs-titan-pc && ls -la $E/bin/mistralrs-titan-pc
for u in $E/deploy/titan-mistral.service $HOME/.config/systemd/user/titan-mistral.service; do
  sed -i 's#bin/mistralrs-titan-integ serve#bin/mistralrs-titan-pc serve#; s#^Description=\(.*\), web UI#Description=\1, hybrid prefix cache, web UI#' $u
  grep -E "^(Description|ExecStart)" $u
done
systemctl --user daemon-reload
touch $E/m4/pc/q/end

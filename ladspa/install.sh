#!/bin/bash
#
#sudo apt install pulseaudio pulseaudio-utils pavucontrol
#

#cp config-ov.ini ../models/config.ini
#
wget https://huggingface.co/Intel/deepfilternet-openvino/resolve/main/deepfilternet3.zip
unzip deepfilternet3.zip -d models_tmp/
python3 reshape-ir-ov.py 
rm -R models_tmp/ && true
rm deepfilternet3.zip
##wget https://huggingface.co/Intel/deepfilternet-openvino/resolve/main/deepfilternet2.zip

cp config-ov.ini ../models/config.ini
cp default.pa ~/.config/pulse/
systemctl --user restart pulseaudio.service
sleep 5
systemctl --user status pulseaudio.service

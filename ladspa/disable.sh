#!/bin/bash
#

systemctl --user stop pulseaudio.service pulseaudio.socket
systemctl --user mask pulseaudio.service
systemctl --user status pulseaudio.service

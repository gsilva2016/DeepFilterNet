#!/bin/bash
#
systemctl --user unmask pulseaudio.service
systemctl --user start pulseaudio.service pulseaudio.socket
systemctl --user status pulseaudio.service

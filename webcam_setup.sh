#!/bin/bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-ctrls  # X -- for your device number
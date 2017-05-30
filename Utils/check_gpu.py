#!/usr/local/bin/python
try:
	import cntk
except:
	print("You do not have CNTK")
	exit()
print(cntk.device.all_devices())

if cntk.try_set_default_device(cntk.device.gpu(0)):
	print("You have GPU Support in CNTK")
else:
	print("You DO NOT have GPU Support in CNTK")

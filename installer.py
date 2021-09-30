import os
from platform import system
import subprocess

if system == "Linux":
    subprocess.run(args=["./installer.sh"])
elif system == "Windows":

else:
    print("We only support Linux(Ubuntu) and Windows.")
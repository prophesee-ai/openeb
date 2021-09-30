from platform import system
import subprocess
import os

def is_on_FAT():
    import psutil
    drive = os.getcwd()[:2]
    dps = psutil.disk_partitions()
    for i in [0, len(dps) - 1]:
        if (dps[i].device[:2] == drive):
            if (dps[i].fstype.find("FAT") != -1):
                print ("The current disk was detected as FAT.\nOpenEB cannot install on a FAT/exFAT drive. Please move and restart the script on a non-FAT partition.")
                return(true)
    return(false)
                
def has_psutil():
    import importlib.util
    return (importlib.util.find_spec('psutil') != "")

if system == "Linux":
    subprocess.run(args=["./installer.sh"])
elif system == "Windows":
    if (has_psutil()):
        if (is_on_FAT()):
            exit()
    else:
        print("psutil not found: no check if disk is (ex)FAT or not (FAT drives will not work).")
    
else:
    print("We only support Ubuntu and Windows.")
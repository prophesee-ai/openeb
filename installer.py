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

#run the shell installer for Ubuntu
if system == "Linux":
    subprocess.run(args=["installer.sh"])
#run lots of subprocesses for Windows
elif system == "Windows":
    #check and exit if we are on a FAT/exFAT drive
    if (has_psutil()):
        if (is_on_FAT()):
            exit()
    os.chdir("..")
    #check for, and install a local vcpkg
    if not os.isdir("vcpkg"):
        subprocess.run(args=["git", "clone", "https://github.com/microsoft/vcpkg.git"])
    os.chdir("vcpkg")
    subprocess.run(args=["git", "checkout", "08c951fef9de63cde1c6b94245a63db826be2e32"])
    subprocess.run(args=["bootstrap-vcpkg.bat"])
    subprocess.run(args=["vcpkg.exe", "--triplet", "x64-windows", "install", "libusb", "eigen3", "boost", "opencv", "glfw3", "glew", "gtest", "dirent", "pybind11"])
    else:
        print("psutil not found: will not check if disk is (ex)FAT or not (FAT drives will not work).")
    
    
else:
    print("We only support Ubuntu and Windows.")
@echo off
echo ==== NeuroWorkshop Machine Setup ====
echo.
echo This script will prepare your VM for NeuroWorkshop
echo.
echo  - copying notebooks
cd ..\notebooks
mkdir c:\dsvm\notebooks\NeuroWorkshop
copy *.ipynb c:\dsvm\notebooks\NeuroWorkshop
echo  - mounting Azure drive L:
net use L: \\ailearn.file.core.windows.net\data /u:AZURE\ailearn jVEpt3hqkNzNK7cyQ9llgHBbG8irXLuQi6cHAX5HQipqN1kxitwAD9HJlA3j6KcdKy/nPK4o+RbPQNOS/EHgYA==
echo  - copying cats challenge files
echo    WARNING: This will take quite some time!
pushd
c:
cd \
mkdir Cats_Dogs
cd Cats_Dogs
copy L:\Cats_Dogs\*.* .
popd
echo  - VM Setup Done, you are ready to rock!

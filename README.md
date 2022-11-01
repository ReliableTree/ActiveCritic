## Prerequisites 
ActiveCritic requires mujoco version 2.1. Follow the instructions here: https://github.com/openai/mujoco-py .

Install active critic from the root folder of the project with:
pip install -e .
.

Install metaworld. 
Follow the instructions from https://github.com/rlworkgroup/metaworld:
in the root folder, open a terminal and input:

pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld .
You might need to add some path variables, which will be promted, when you try to run metaworld for the first time.

After you installed the package, run the tests by entering the following lines into a console from the root folder of the project:
cd tests/
python test_complete.py

If you get the error:
"
 /tmp/pip-install-rsxccpmh/mujoco-py/mujoco_py/gl/osmesashim.c:1:23: fatal error: GL/osmesa.h: No such file or directory
    compilation terminated.
    error: command 'gcc' failed with exit status 1
"

you can try to run:
$ sudo apt-get install libosmesa6-dev
.

If you get  :
[Errno 2] No such file or directory: 'patchelf'
you can run:
sudo apt-get install patchelf
.

If you get an import error with mujoco_py and GLIBCXX_3.4.20 on Ubuntu 22.02, you can try to find the folder of your anaconda environment -> lib. 
For example:
"~/miniconda/envs/ac/lib".
Open a terminal and type:
mv -vf libstdc++.so.6 libstdc++.so.6.old
To backup you current libstdc.
Then type 
 ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./libstdc++.so.6
 To make a symbolic link to the correct version provided by ubuntu.
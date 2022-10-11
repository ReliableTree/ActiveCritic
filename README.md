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
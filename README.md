# Parallel simulation of Retinal Bipolar Neurons
Members:
- Abhiruchi Kotamkar
- Anupsa Swain
- Shobhit Maheshwari
- Shreyas Grampurohit

## Instructions to run
For the python model, simply go to `python_files` and run the script `runner.py`
For the c++ model, compile with the following command: `g++ -fopenmp -O3 src/* -I ./include -o brain_executable`
The executable can be run as `./brain_executable [num_neurons] [num_threads] [num_milliseconds_simualtion]`

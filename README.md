# chaos_detection_ANN

A neural network that has been trained to detect temporal correlation and distinguish chaotic from stochastic signals.

The 'auto_ANN_Omega' file depicts the fully automatic code with the necessary libraries.

- The main file 'chaos_detection_ANN.py' contains all the information.

- The files: 
- 'W1.dat', 'W2.dat', 'B1.dat', 'B2.dat' are the weights of the ANN
- 'colorednoise.py' is the library to generate the flicker noise (colored noise)

Instructions for running the code:

python chaos_detection_ANN.py serie.dat 

'serie.dat' is the time series to be analyzed.
The code compares the time-series with one flicker-noise time-series with the same correlation coefficient (predicted by the ANN)
and the same length.
For small time-series length<1000 points) we suggest the command:

python chaos_detection_ANN.py serie.dat 10 

In this case, the code compares the time-series with 10 flicker-noise time-series.




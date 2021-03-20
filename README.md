# chaos_detection_ANN

A neural network that has been trained to detect temporal correlation and distinguish chaotic from stochastic signals.

The <code>'/auto_ANN_Omega/'</code> directory depicts the fully automatic code with the necessary libraries.

The main file <code>'chaos_detection_ANN.py'</code> contains all the information.

The files: 
<code>W1.dat'</code>, <code>'W2.dat'</code>, <code>'B1.dat'</code>, <code>'B2.dat'</code> are the weights of the ANN.
<code>'colorednoise.py'</code> is the library to generate the flicker noise (colored noise).

Instructions for running the code:

<code>python chaos_detection_ANN.py serie.dat</code> 

<code>'serie.dat'</code> is the time series to be analyzed.
The code compares the time-series with <code>1</code> flicker-noise time-series with the same correlation coefficient (predicted by the ANN)
and the same length.
For small time-series length<1000 points we suggest the command:

<code>python chaos_detection_ANN.py serie.dat 10 </code>

In this case, the code compares the time-series with <code>10</code> flicker-noise time-series.

The <code>'tests'</code> directory presents an autorun of the <code>Figure 3</code> for a practical use, with fewer points (<code>101</code> initial conditions insted of <code>1000</code>) and less precision (length <code>2^16</code> instead of <code>2^20</code>).

Instructions for running the code:
<code>python autorun3.py </code>

After a few minutes the figure <code>'test_fig3.png'</code> is generated.






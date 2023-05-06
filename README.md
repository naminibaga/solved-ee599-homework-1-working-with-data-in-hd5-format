Download Link: https://assignmentchef.com/product/solved-ee599-homework-1-working-with-data-in-hd5-format
<br>
Problem 1: Working with data in hd5 format

In this problem you will create an hd5 file containing a numpy array of binary random sequences that you generate yourself. The hd5 format is useful because it allows one to store multiple data objects in a single file with object names – e.g., you can store ‘regressor’ and ‘label’ datasets where the former is a nupy array of floats and the latter is a numpy integer array. Also, hd5 allows for fast, non-sequential access to data in the dataset without reading the entire dataset – e.g., one can access x[ idxs ] where idxs = [ 2, 234, 512] efficiently. The access property is useful when pulling a batch of data from a large dataset in the process of training. Here are the steps to follow:

<ol>

 <li>Obtain and run the template python file provided – this is in DEBUG mode by default.</li>

 <li>Experiment with the error trapping assert statements to see what they are doing, using the shape method on numpy arrays, etc.</li>

 <li>Make the DEBUG flag False and enter 25 binary sequences, each of length 20. <strong>It is important that you do this by hand and without the aid of a compute random number generator or, for example, a coin.</strong></li>

 <li>Make sure that your hd5 file has been written properly and can be read and checked to be correct.</li>

 <li>Name your hd5 file with your full name and submit it through the Canvas website with this assignment.</li>

</ol>

<h1>Problem 2: A simple feedforward (MLP) neural network</h1>

An MLP has two input nodes, one hidden layer, and two outputs. The activation for the hidden layer is ReLu. The output layer is linear (<em>i.e., </em>identity activation). The two sets of weights and biases are given by

<strong>W</strong>

<strong>W</strong>

What is the output activation when the input is <strong>x </strong>= [ + 1 − 1 ]<sup>t</sup>?

<strong>Note: </strong>the ReLu activation is <em>h</em>(<em>x</em>) = <em>x </em>for <em>x &gt; </em>0 and <em>h</em>(<em>x</em>) = 0 for <em>x </em>≤ 0.

1

<a href="https://www.ankitcodinghub.com/wp-content/uploads/2020/07/HW1.zip">HW1</a>Problem 3: Convolutions and correlations

Convolutions and correlations are covered in a typical signals and systems class in your undergraduate degree. These are used in Convolutional Neural Networks (CNNs), so this is a brief review problem. For a one-dimensional, discrete time (index) signals, convolution is defined as

and a correlation is defined as

For two-dimensional signals (e.g., images and image filters) convolution is

and correlation is

<ol>

 <li>Show that <em>y</em>[<em>n</em>] = <em>x</em>[<em>n</em>] ∗ <em>h</em>[<em>n</em>] = <em>x</em>[<em>n</em>] <em>? h</em>[−<em>n</em>] – i.e., that correlation is convolution with the reflected (reversed) signal. State and show the relationship between convolution and correlation in the 2D case.</li>

 <li>Find <em>y</em>[<em>n</em>] = <em>x</em>[<em>n</em>]∗<em>h</em>[<em>n</em>] for the signals shown below. Plot the result in python using a stem-plot.</li>

 <li>Find <em>r</em>[<em>i</em>][<em>j</em>] = <em>x</em>[<em>i</em>][<em>j</em>] <em>? h</em>[<em>i</em>][<em>j</em>] for the signals shown below. Plot a heat-map of the result using <a href="https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.matshow.html">pyplot.matshow</a><a href="https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.matshow.html">.</a></li>

</ol>

<table width="576">

 <tbody>

  <tr>

   <td width="30">…</td>

   <td width="21">…</td>

   <td width="102">…</td>

   <td width="102">…</td>

   <td width="90">…</td>

   <td width="90">…</td>

   <td width="90">…</td>

   <td width="21">…</td>

   <td width="30"></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="102">0</td>

   <td width="102">0</td>

   <td width="90">0</td>

   <td width="90">0</td>

   <td width="90">0</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="102"><em>x</em>[−2][1] = 1</td>

   <td width="102"><em>x</em>[−1][1] = 1</td>

   <td width="90"><em>x</em>[0][1] = 1</td>

   <td width="90"><em>x</em>[1][1] = 1</td>

   <td width="90"><em>x</em>[2][1] = 1</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="102"><em>x</em>[−2][0] = 1</td>

   <td width="102"><em>x</em>[−1][0] = 1</td>

   <td width="90"><em>x</em>[0][0] = 1</td>

   <td width="90"><em>x</em>[1][0] = 1</td>

   <td width="90"><em>x</em>[2][0] = 1</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="102"><em>x</em>[−2][−1] = 1</td>

   <td width="102"><em>x</em>[−1][−1] = 1</td>

   <td width="90"><em>x</em>[0][−1] = 1</td>

   <td width="90"><em>x</em>[1][−1] = 1</td>

   <td width="90"><em>x</em>[2][−1] = 1</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="102">0</td>

   <td width="102">0</td>

   <td width="90">0</td>

   <td width="90">0</td>

   <td width="90">0</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"></td>

   <td width="21">…</td>

   <td width="102">…</td>

   <td width="102">…</td>

   <td width="90">…</td>

   <td width="90">…</td>

   <td width="90">…</td>

   <td width="21">…</td>

   <td width="30">…</td>

  </tr>

 </tbody>

</table>

and

<table width="428">

 <tbody>

  <tr>

   <td width="30">…</td>

   <td width="21">…</td>

   <td width="116">…</td>

   <td width="105">…</td>

   <td width="105">…</td>

   <td width="21">…</td>

   <td width="30"></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="116">0</td>

   <td width="105">0</td>

   <td width="105">0</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="116"><em>h</em>[−1][1] = 1<em>/</em>4</td>

   <td width="105"><em>h</em>[0][1] = 1<em>/</em>2</td>

   <td width="105"><em>h</em>[1][1] = 1<em>/</em>4</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="116"><em>h</em>[−1][0] = 1<em>/</em>2</td>

   <td width="105"><em>h</em>[0][0] = 1</td>

   <td width="105"><em>h</em>[1][0] = 1<em>/</em>2</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="116"><em>h</em>[−1][−1] = 1<em>/</em>4</td>

   <td width="105"><em>h</em>[0][−1] = 1<em>/</em>2</td>

   <td width="105"><em>h</em>[1][−1] = 1<em>/</em>4</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"><em>…</em></td>

   <td width="21">0</td>

   <td width="116">0</td>

   <td width="105">0</td>

   <td width="105">0</td>

   <td width="21">0</td>

   <td width="30"><em>…</em></td>

  </tr>

  <tr>

   <td width="30"></td>

   <td width="21">…</td>

   <td width="116">…</td>

   <td width="105">…</td>

   <td width="105">…</td>

   <td width="21">…</td>

   <td width="30">…</td>

  </tr>

 </tbody>

</table>

<strong>Note: </strong>In both cases, all values at indices not shown are assumed to be zero and you only need to show the output over the region where it is non-zero.

<h1>Problem 4: Using Filters in Python</h1>

This is a basic problem to build proficiency in Python. There are filter design and analysis tools in Python that are very similar to those in Matlab. In this problem you will design and plot the frequency response for a few ARMA filters. For your reference and review, Fig. 1.

<ol>

 <li>Many methods in deep learning use a first order (<em>L </em>= 1) AR filter with unit gain at DC – <em>i.e., </em>the gain of the frequency response at zero frequency is 1. Also by “AR” it is implied that <em>b</em>[<em>i</em>] = 0 for all <em>i &gt; </em>0. This type of filter has a single parameter <em>α </em>which is the pole location in the <em>z</em>-plane.</li>

</ol>

<ul>

 <li>Specify the difference equation and <em>Z</em>-transform for this first order AR filter – <em>e., </em>specify <em>b</em>[0], and <em>a</em>[1] in terms of <em>α</em>. Note that, in the standard form in Fig. 1 <em>a</em>[0] = 1 by default, but, just as in matlab, you need to enter for <em>a</em>[0] in the Python routine.</li>

</ul>

<ol>

 <li><strong>Note: </strong>enough information has been given above to fully specify this simple filter. If you do not see that, this is a brief review of signals and systems and you can get help from the TAs.</li>

</ol>

<ul>

 <li>Plot the magnitude of the frequency response for <em>α </em>= 0<em>.</em>9, <em>α </em>= 0<em>.</em>5, <em>α </em>= 0<em>.</em>1, <em>α </em>= −0<em>.</em> Use linear normalized frequency <em>ν</em>, which is unique on [−1<em>/</em>2<em>,</em>+1<em>/</em>2] and has units of cycles per sample. <strong>Hint: </strong>use scipy.signal.freqz.</li>

 <li>If the time constant is defined as the number of samples required for the impulse response to decay to 20% of its value at <em>n </em>= 0, give the time constant for <em>α </em>= 0<em>.</em>9, <em>α </em>= 0<em>.</em>5, <em>α </em>= 0<em>.</em></li>

</ul>

Python has most of the matlab functionality for filter design. These sub-problems are aimed at getting you experience using those tools.

<ul>

 <li>Design an <em>L </em>= 4 Butterworth filter with bandwidth of <em>ν</em><sub>0 </sub>= 0<em>.</em>25 – here normalized linear frequency is denoted by <em>ν </em>in cycles/sample – i.e., the frequency response is periodic in <em>ν </em>with period one. Provide the AR and MA coefficients and plot the magnitude of the frequency response.</li>

</ul>

<em>v</em>[<em>n</em>] = <em>x</em>[<em>n</em>] (<em>a</em>[1]<em>v</em>[<em>n </em>1] + <em>a</em>[2]<em>v</em>[<em>n </em>2] + ···<em>a</em>[<em>L</em>]<em>v</em>[<em>n L</em>]) <em>y</em>[<em>n</em>] = <em>b</em>[0]<em>v</em>[<em>n</em>] + <em>b</em>[1]<em>v</em>[<em>n </em>1] + <em>b</em>[2]<em>v</em>[<em>n </em>2] + ··· + <em>b</em>[<em>L</em>]<em>v</em>[<em>n L</em>]

state[<em>n</em>] = (<em>v</em>[<em>n         </em>1]<em>,v</em>[<em>n           </em>1]<em>,…v</em>[<em>n          L</em>])

implements this difference equation:

Frequency response:

<em>                             z</em>=<em>e</em><em>j</em>2<em>⇡⌫</em>

Figure 1: The general format and notation for an ARMA filter. The arrays of AR coefficients <em>a</em>[<em>n</em>] and MA coefficients <em>b</em>[<em>n</em>] define the filter. The order of teh filter is the number of delay elements in thsi diagram, <em>L</em>.

<ul>

 <li>Create a numpy array of length 300 comprising iid realizations of a standard normal distribution. Filter this sequence using the Butterworth filter and plot the input and output signals on the same plot. <strong>Hint: </strong>Use signal.lfilter.</li>

</ul>
#  Project report
  
  
- [Project report](#project-report)
  - [Simulation](#simulation)
    - [Problem analysis](#problem-analysis)
    - [Program design](#program-design)
      - [Random mode](#random-mode)
        - [inter-arrival time](#inter-arrival-time)
        - [service time](#service-time)
      - [Trace mode](#trace-mode)
      - [Build requests](#build-requests)
      - [Simulation](#simulation-1)
        - [UML](#uml)
        - [Simulation flow](#simulation-flow)
  - [Design problems](#design-problems)
    - [Basic ideas](#basic-ideas)
      - [Generate requests](#generate-requests)
      - [Replication experiments](#replication-experiments)
      - [Transient removal](#transient-removal)
      - [Confidence interval](#confidence-interval)
    - [Version 1](#version-1)
    - [Version 2](#version-2)
    - [Compare algorithm](#compare-algorithm)
  
##  Simulation
  
  
###  Problem analysis
  
  
According to the project's description, we know:
  
- The server farm consists of 3 servers, each has an infinite queue, and a dispatcher with negligible process time.
- The dispatcher can distribute requests according to servers' status.
- Server 1 and server 2 have the same process speed 1, and server 3 has process speed f.
- All of the servers are FCFS (first come first serve).
  
We need to build a simulation program based on these rules. And it must have the following functions:
  
- Do simulation based on input requests.
- Do simulation based on parameters, which would be used to generate request distribution.
- Output simulation results.
  
###  Program design
  
  
We have trace mode and random mode to generate different requests. And here is the program flow chart.
  

![](/assets/77967c8b30c20f08b7f4c9acc8ebce130.png?0.5229833603202279)  
  
####  Random mode
  
  
In random mode, we need to generate the requests' distribution based on the input parameters.
  
#####  inter-arrival time
  
  
According to part 5.1.1 we know that the inter-arrival times of the requests <img src="https://latex.codecogs.com/gif.latex?a_k%20=%20a_{1k}a_{2k}"/>
  
And <img src="https://latex.codecogs.com/gif.latex?a_{1k}"/> is exponentially distributed with a mean arrival rate <img src="https://latex.codecogs.com/gif.latex?&#x5C;lambda"/>. Which means the PDF of <img src="https://latex.codecogs.com/gif.latex?a_{1k}"/> is
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?f(t)%20=%20&#x5C;begin{cases}%20%20%20&#x5C;lambda%20e^{-&#x5C;lambda%20t}%20&amp;&#x5C;text{if%20}%20t%20&#x5C;ge%200%20&#x5C;&#x5C;%20%20%200%20&amp;&#x5C;text{if%20}%20t%20&#x5C;lt%200&#x5C;end{cases}"/></p>  
  
  
Then the CDF of <img src="https://latex.codecogs.com/gif.latex?a_{1k}"/> is 
<p align="center"><img src="https://latex.codecogs.com/gif.latex?F(t)%20=%20&#x5C;int_{0}^{t}%20f(t)%20dt%20=%201%20-%20e^{-&#x5C;lambda%20t}"/></p>  
  
  
So based on lecture, we can use an uniformly distributed <img src="https://latex.codecogs.com/gif.latex?u(t)"/> in <img src="https://latex.codecogs.com/gif.latex?(1,0)"/> to generate the exponentially distribution.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}D(u(t))%20&amp;=%20F^{-1}(u(t))%20&#x5C;&#x5C;&amp;=%20-%20&#x5C;log%20&#x5C;frac{1-u(t)}{&#x5C;lambda}&#x5C;end{aligned}"/></p>  
  
  
And here is the python code to generate <img src="https://latex.codecogs.com/gif.latex?a_{1k}"/> with <img src="https://latex.codecogs.com/gif.latex?&#x5C;lambda%20=%202"/> and 10000 point with uniformly distributed <img src="https://latex.codecogs.com/gif.latex?u(t)"/>
  
```python
import numpy as np 
  
n = 10000
lamb = 2
  
u = np.random.random((n,))
exp_distribution = -np.log(1-u)/lamb
```
  
Then We use a simple code to plot the distribution I generated, to get an intuitive view.
  
```python
def plot_expected(x, expected_func, name = None):
    nb = 50 # Number of bins in histogram 
    freq, bin_edges = np.histogram(x, bins = nb, range=(0,np.max(x)))
  
    # Lower and upper limits of the bins
    bin_lower = bin_edges[:-1]
    bin_upper = bin_edges[1:]
    # expected number in each bin
    y_expected = n*(expected_func(bin_lower)-expected_func(bin_upper))
  
    bin_center = (bin_lower+bin_upper)/2
    bin_width = bin_edges[1]-bin_edges[0]
  
    plt.bar(bin_center,freq,width=bin_width)
    plt.plot(bin_center,y_expected,'r--',label = 'Expected',Linewidth =3)
    plt.legend()
  
    if name is not None:
        plt.savefig(name)
```
  
These code can be found in `distribution.ipynb`. So the distribution of <img src="https://latex.codecogs.com/gif.latex?a_{1k}"/> is
  
```python
plot_expectd(exp, lambda x: np.exp(-lamb * x), name='exp.png')
```
![](../exp.png?0.8352778772647482 )  
  
Then we aims to generate distribution of <img src="https://latex.codecogs.com/gif.latex?a_{2k}"/>. We notice that <img src="https://latex.codecogs.com/gif.latex?a_{2k}"/> is an uniformly distribution in the interval <img src="https://latex.codecogs.com/gif.latex?[a_{2l},a_{2u}]"/>. We can write a linear transformation to generate this distribution.
  
```python
a2k = (a2u - a2l) * u + a2l
```
  
So we can build the inter-arrival time distribution using <img src="https://latex.codecogs.com/gif.latex?a_k%20=%20a_{1k}%20a_{2k}"/>.
  
#####  service time
  
  
According to the project description, we know the PDF of service time distribution is
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?g(t)%20=%20&#x5C;begin{cases}%20%20%20%200%20&amp;&#x5C;text{if%20}%200%20&#x5C;le%20t%20&#x5C;le%20&#x5C;alpha&#x5C;&#x5C;%20%20%20%20&#x5C;frac{&#x5C;gamma}{t^&#x5C;beta}%20&amp;&#x5C;text{if%20}%20&#x5C;alpha%20&#x5C;le%20t&#x5C;&#x5C;&#x5C;end{cases}"/></p>  
  
  
and <img src="https://latex.codecogs.com/gif.latex?&#x5C;gamma%20=%20&#x5C;frac{&#x5C;beta%20-%201}{&#x5C;alpha%20^{1-&#x5C;beta}}"/>.
Then the CDF of service time distribution is
<p align="center"><img src="https://latex.codecogs.com/gif.latex?G(t)%20=%20&#x5C;int_{0}^{t}%20g(t)%20dt%20=%20&#x5C;begin{cases}0%20&amp;&#x5C;text{if%20}%200%20&#x5C;le%20t%20&#x5C;le%20&#x5C;alpha&#x5C;&#x5C;&#x5C;frac{&#x5C;gamma%20x%20^{1-&#x5C;beta}}{1-&#x5C;beta}%20&amp;&#x5C;text{if%20}%20&#x5C;alpha%20&#x5C;le%20t&#x5C;&#x5C;&#x5C;end{cases}"/></p>  
  
  
So based on lecture, we can use an uniformly distributed <img src="https://latex.codecogs.com/gif.latex?u(t)"/> in <img src="https://latex.codecogs.com/gif.latex?(1,0)"/> to generate the service time distribution.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{aligned}D(u(t))%20&amp;=%20G^{-1}(u(t))%20&#x5C;&#x5C;&amp;=%20(&#x5C;frac{(&#x5C;beta%20-%201)}{&#x5C;gamma}%20u(t))^{&#x5C;frac{1}{1-&#x5C;beta}}&#x5C;end{aligned}"/></p>  
  
  
And here is the python code to generate <img src="https://latex.codecogs.com/gif.latex?a_{1k}"/> with <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha%20=%200.150&#x5C;text{%20and%20}&#x5C;beta%20=%203.600"/> and 10000 point with uniformly distributed <img src="https://latex.codecogs.com/gif.latex?u(t)"/>
  
```python
n = 1000
alpha = 0.150
beta = 3.600
gama = (beta - 1)/(alpha ** (1-beta))
  
u = np.random.random((n,))
service_time = ( (beta - 1)/gama * u )**(1/(1 - beta))
```
  
These code can be found in `distribution.ipynb`. And I also use `plot_expected()`to plot the distribution I generated, to get an intuitive view of service time.
  
```python
plot_expectd(t, lambda x: gama / (x ** beta), name='service_time.png')
```
![](../service_time.png?0.5540886758737014 )  
  
####  Trace mode
  
  
We could read inter-arrival time and service time from the given files. Then we can uses these data to build the requests.
  
####  Build requests
  
  
We can use the inter-arrival time and service time data to build our request. Which consist of the following components.
  
- interarrival_time
- service_time
- arrival_time
  
And we use a python class to represent `Request`. The python code is in `main_ex.py`, named as `Request`.
  
```python
class Request:
  
    def __init__(self, interarrival_time, service_time, arrival_time):
        self.interarrival_time = interarrival_time
        self.service_time = service_time
        self.arrival_time = arrival_time
  
    def setArrivalTime(self, arrival_time):
        self.arrival_time = arrival_time
  
    def __str__(self):
        return str({
            'interarrival_time': self.interarrival_time,
            'service_time': self.service_time,
            'arrival_time': self.arrival_time
        })
  
    def __repr__(self):
        return self.__str__()
```
  
In build requests' part, we need to use inter-arrival times to calculate the arrival times. Here is the flow char.
  

![](/assets/77967c8b30c20f08b7f4c9acc8ebce131.png?0.7740524905047996)  
  
And here is the python code.
  
```python
request_list = []
last_arrival = 0
for i in range(len(interarrival_time_list)):
    request_list.append(Request(
         interarrival_time_list[i],
         service_time_list[i], 
         last_arrival + interarrival_time_list[i]))
    last_arrival += interarrival_time_list[i]
```
  
####  Simulation
  
  
#####  UML
  
  
After we built the requests, we need use computer program to make simulation. We need 3 class to represent the states in out simulation.
  
- Response
- ServerProcess
- Server
  
The UML is showing as follow.
  

![](/assets/77967c8b30c20f08b7f4c9acc8ebce132.png?0.42826807889160934)  
  
The `Response` class is used to make a response to a request, which includes the following contents.
  
- arrival_time: the arrival time of this request.
- departure_time: the response time of this request.
- service_time: the service time needed to process this request.
- server_name: which server did this process.
- response_time: the response time of this request.
  
And the python code is in `main_ex.py`, named as `Response`.
  
The `ServerProcess` class is used to represent the internal state of servers. Which includes
  
- arrival_time: the arrival time of this request.
- departure_time: the response time of this request.
- service_time: the service time needed to process this request.
  
And the python code is in `main_ex.py`, named as `ServerProcess`.
  
  
The `Server` class is the abstraction of server. It has the following attributes.
  
- process: the on processing status, which is a ServerProcess.
- request_buffer: the requests queue of this server.
- server_name: the server name.
- mu: the processing speed of this server, typically server1 and server2 is 1, server3 is <img src="https://latex.codecogs.com/gif.latex?&#x5C;lambda"/>.
  
We also add some functions to this class, which would be very helpful when making a simulation.
  
- is_finish_request: see if this server finish its request.
- departure_request: departure the request.
- get_next_departure_time: get the departure time of this server.
- get_jobs: get the number of requests in this server (numbers in queue + server)
  
And the python code is in `main_ex.py`, named as `Server`.
  
#####  Simulation flow
  
  
Then we could design our simulation flow as follow.
  

![](/assets/77967c8b30c20f08b7f4c9acc8ebce133.png?0.31751191934952905)  
  
There are 2 sub routines in the simulation process.
  
- process requests
- process server
  
For process request routine, we have.
  

![](/assets/77967c8b30c20f08b7f4c9acc8ebce134.png?0.8116069565340487)  
  
For process servers routine, we have.
  

![](/assets/77967c8b30c20f08b7f4c9acc8ebce135.png?0.11268828384960794)  
  
And the python code of this simulation program is in `main_ex.py`, named as `simulation()`.
  
##  Design problems
  
  
###  Basic ideas
  
  
Here is the basic ideas to determine the suitable value of d for load balance algorithms.
  
For each of the d values, do the following steps. 
  
1. Generate requests: for CRN method, we want use the same requests for different d value to reduced the confidence interval.
2. Replication experiments: for a value of d, repeat simulations n times using different requests.
3. Transient removal: get the mean response time of steady state from each simulation.
4. Calculate confidence interval.
  
####  Generate requests
  
  
We need to build a set of requests for the CRN method and Independent replication. And this can be found in `design problem.ipynb`.
  
```python
# build requests
times = 10
  
requests = []
for i in range(times):
    requests.append(build_request_list_random(lamb, a2l,a2u,alpha,beta, n = 50000))
```
  
####  Replication experiments
  
  
And we  also can write code to make the replication experiments. And this can be found in `design problem.ipynb`.
  
```python
# independent replication experiments
def get_ravg(d,times=3, version = 1, time_end=5000):
    ravg_list = []
    for i in range(times):
        req = copy.deepcopy(requests[i])
        rr = simulation(req, version, d, f, time_end) 
        response_time = [r.response_time for r in rr]
        ravg_list.append(sum(response_time) / len(response_time))
  
    return np.array(ravg_list)
```
  
Here we plot the confidence interval of different replication numbers (d = 1, using version 1).
  
![](experiment.png?0.7091135892630955 )  
  
  
From this plot, we choose to use **3** replica experiments. Because it is good enough, and further experiments can not improve the confidence interval
  
  
  
####  Transient removal
  
  
Then, let's take a look at the means response time of first k request. Using the following python code, which can be found in `design problem.ipynb` to plot. 
  
```python
# means response time of first k request.
d = 1
time_end = 5000
version = 1
  
request_list = build_request_list_random(lamb, a2l,a2u,alpha,beta)
rr = simulation(request_list, version, d, f, time_end, 
                generate_request=lambda now: 
                    build_request_list_random(lamb, a2l,a2u,alpha,beta, now=now))
response_time = [r.response_time for r in rr]
  
avg = 0
avg_list = []
for i in range(1, len(response_time)):
    avg_list.append(sum(response_time[:i]) / i)
  
plt.plot(avg_list)
plt.savefig('first_k.png')
```
  
![](first_k.png?0.5783657638229291 )  
  
From this figure, we found that remove first 10000 response would be a good choice to calculate the means response time of steady state.
  
####  Confidence interval
  
  
From the lecture we can write code to calculate the confidence interval. And this can be found in `design problem.ipynb`.
  
```python
# compute error value
def get_error_value(ravg):
    p = 0.95 
    num_tests = ravg.shape[0]
    mf = t.ppf(1-(1-p)/2,num_tests-1)/np.sqrt(num_tests)
    return np.std(ravg,ddof=1) * mf
```
  
###  Version 1
  
  
For version 1 algorithm, we found that the core statement is line 4, `else if ns == 0 or ns <= n3 - d` , which means the value from d-1 (not included) to d have the same behaviors. So we use <img src="https://latex.codecogs.com/gif.latex?d_i%20=%20i%20&#x5C;text{%20for%20}%20i%20&#x5C;in%20[0,%20&#x5C;inf]%20&#x5C;cap%20N"/> to find the best <img src="https://latex.codecogs.com/gif.latex?d_i"/>.
  
1. for each <img src="https://latex.codecogs.com/gif.latex?d_i"/>, do **3** replica experiments.
2. for each experiments, remove the first **10000** response, then calculate the mean response time for this experiment.
3. compute the confidence interval with <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha%20=%200.95"/>, for this 5 mean response time.
4. plot these confidence interval.
  
Here is the first 10 di's confidence interval of mean response time.
  
![](v1.png?0.8458549944013183 )  
  
From this figure, we can find that the best value of d for **Version 1** algorithm is **1.0**. The code used here is showing below, and can also find in `design problem.ipynb`.
  
```python
max_d = 10
dec = 1
  
x_ticks = [str(i/dec) for i in range(max_d * dec)]
x1 = []
y1 = []
error = []
for i in range(0,max_d * dec):
    d = i / dec
    ravg = get_ravg(d, times = 3)
    x1.append(i)
    y1.append(np.mean(ravg))
    error.append(get_error_value(ravg))
  
plt.errorbar(x=x1, y=y1, yerr=error, color="black", capsize=3,
             linestyle="None",
             marker="s", markersize=7, mfc="black", mec="black")
  
  
plt.xticks(x1, x_ticks, rotation=90)
  
plt.tight_layout()
plt.show()
```
  
###  Version 2
  
  
For version 2 algorithm, we found that the core statement is line 4, `else if ns == 0 or ns <= n3/f - d` . We use <img src="https://latex.codecogs.com/gif.latex?d_i%20=%20&#x5C;frac{i}{10}%20&#x5C;text{%20for%20}%20i%20&#x5C;in%20[0,%20&#x5C;inf]%20&#x5C;cap%20N"/> to find the best <img src="https://latex.codecogs.com/gif.latex?d_i"/>.
  
1. for each <img src="https://latex.codecogs.com/gif.latex?d_i"/>, do **3** replica experiments.
2. for each experiments, remove the first **10000** response, then calculate the mean response time for this experiment.
3. compute the confidence interval with <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha%20=%200.95"/>, for this 5 mean response time.
4. plot these confidence interval.
  
Here is the first 20 di's confidence interval of mean response time.
  
![](v2.png?0.8409674893186427 )  
  
From this figure, we can find that the best value of d for **Version 2** algorithm is **from 0 to 0.3**. And we choose to use <img src="https://latex.codecogs.com/gif.latex?d_2=0.3"/> as our d value. The code used here is showing below, and can also find in `design problem.ipynb`.
  
```python
max_d = 2
dec = 10
  
x_ticks = [str(i/dec) for i in range(max_d * dec)]
x1 = []
y1 = []
error = []
for i in range(0,max_d * dec):
    print(i)
    d = i / dec
    ravg = get_ravg(d, version = 2, times = 3)
    x1.append(i)
    y1.append(np.mean(ravg))
    error.append(get_error_value(ravg))
  
plt.errorbar(x=x1, y=y1, yerr=error, color="black", capsize=3,
             linestyle="None",
             marker="s", markersize=7, mfc="black", mec="black")
  
  
plt.xticks(x1, x_ticks, rotation=90)
  
plt.tight_layout()
plt.savefig('v2.png')
```
  
###  Compare algorithm
  
  
Based on the work on version 1 and version 2. we know the best d values.
  
- For version 1: <img src="https://latex.codecogs.com/gif.latex?d_1%20=%201"/>
- For version 2: <img src="https://latex.codecogs.com/gif.latex?d_2%20="/>
  
To compare these 2 systems' performance, we did the following jobs.
  
1. do **3** replica experiments.
2. for each experiments, remove the first **10000** response, then calculate the mean response time for this experiment.
3. compute the confidence interval with <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha%20=%200.95"/>, for this 5 mean response time.
4. plot these confidence interval.
  
Here is the confidence interval of these systems.
  
![](compare.png?0.8909839297845448 )  
  
From this figure, we can find that CIs overlap and mean of a system is in the CI of the other. So they are **not different**.
  
The code used here is showing below, and can also find in `design problem.ipynb`.
  
```python
# compare
d1 = 1
d2 = 0.3
  
x_ticks = ['version 1', 'version 2']
x1 = [1, 2]
y1 = []
error = []
  
print('v1')
ravg = get_ravg(d1, version = 1, times = 3)
y1.append(np.mean(ravg))
error.append(get_error_value(ravg))
  
print('v2')
ravg = get_ravg(d2, version = 2, times = 3)
y1.append(np.mean(ravg))
error.append(get_error_value(ravg))
  
plt.errorbar(x=x1, y=y1, yerr=error, color="black", capsize=3,
             linestyle="None",
             marker="s", markersize=7, mfc="black", mec="black")
  
plt.xticks(x1, x_ticks, rotation=90)
  
plt.tight_layout()
plt.savefig('compare.png')
```
  
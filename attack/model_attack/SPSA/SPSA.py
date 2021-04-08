"""
The codes are in https://github.com/TheBugger228/SPSA
"""
import numpy as np
import matplotlib.pyplot as plt

# For information about the algorithm, visit:
# https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF

# Consatants to be used for the gradient descent
constats = {"alpha": 0.302, "gamma": 0.101, "a": 0.9283185307179586, "c": 0.1, "A": False}
# constats = {"alpha": 0.602, "gamma": 0.101, "a": 0.6283185307179586, "c": 0.1, "A": False}
# Main minimising function
def SPSA(f, theta, n_iter, extra_params = False, theta_min = None, theta_max = None,
	report=False, constats=constats, return_progress=False, threshold=None):
	
	# Parameters:
	# 	f: Function to be minimised (func)
	# 	theta: Initial function parameters (np.array)
	# 	n_iter: Number of iterations (int)
	# 	extra_params: Extra parameters taken by f (np.array)
	# 	theta_min: Minimum value of theta (np.array)
	# 	theta_max: Maximum value of theta (np.array)
	# 	report: Print progress. If False, nothing is printed. If int, every
	# 		report iterations you will get the iteration number, function 
	# 		value and parameter values (bool / int)
	# 	constats: Constants needed for the gradient descent (dict)
	#		default is {"alpha": 0.602, "gamma": 0.101, "a": 0.6283185307179586, "c": 0.1, "A": False}
	# 	return_progress: Return array with all the function values at every return_progress iteration (bool / int)

	# Returns:
	# 	theta: Optimum parameters values to minimise f (np.array)
	# 	f(theta): Minimum value found (float)
	# 	If return_progress == True:
	# 		progress: Array with all the function values at each return_progress iteration (np.array)

	# Get value of p from paramters
	p = len(theta)

	# Get constants from dictionary
	alpha = constats["alpha"]
	gamma = constats["gamma"]
	a = constats["a"]
	c = constats["c"]
	A = constats["A"]

	if A == False:
		A = n_iter / 10

	if return_progress:
		if extra_params is False:
			progress = np.array([[0, f(theta)]])
		else:
			progress = np.array([[0, f(theta)]])  # progress = np.array([[0, f(*theta, *extra_params)]])

	# Carry out the iterations
	for k in range(1, n_iter + 1):
		ak = a / (k + A)**alpha
		ck = c / k**gamma

		delta = 2 * np.round(np.random.rand(p, )) - 1

		theta_plus = theta + ck * delta
		theta_minus = theta - ck * delta

		if extra_params is False:
			y_plus = f(theta_plus)
			y_minus = f(theta_minus)
		else:
			y_plus = f(theta_plus)
			y_minus = f(theta_minus)

		# Get derivative
		g_hat = (y_plus - y_minus) / (2 * ck * delta)

		# Gradient descent step
		theta = theta - ak * g_hat

		# Make sure theta is within the boundaries
		if theta_min is not None:
			index_min = np.where(theta < theta_min)
			theta[index_min] = theta_min[index_min]

		if theta_max is not None:
			index_max = np.where(theta > theta_max)
			theta[index_max] = theta_max[index_max]

		# Track progress
		if return_progress:
			if k % return_progress == 0:
				if extra_params is False:
					progress = np.concatenate((progress, np.array([[k, f(theta)]])))
				else:
					progress = np.concatenate((progress, np.array([[k, f(theta)]])))

		# Report progress
		if report:
			if k % report == 0:
				if extra_params is False:
					print(f"Iteration: {k}\tArguments: {theta}\tFunction value: {f(theta)}")
				else:
					print(f"Iteration: {k}\tArguments: {theta}\tFunction value: {f(theta)}")

		# Check if f is lower than the threshold
		if threshold is not None:
			if extra_params is False:
				val = f(theta)
			else:
				val = f(theta)
			if val <= threshold:
				if not return_progress:
					if extra_params is False:
						return theta, f(theta)
					else:
						return theta, f(theta)
				else:
					if extra_params is False:
						return theta, f(theta), progress
					else:
						return theta, f(theta), progress

	# Return optimum value
	if not return_progress:
		if extra_params is False:
			return theta, f(theta)
		else:
			return theta, f(theta)
	else:
		if extra_params is False:
			return theta, f(theta), progress
		else:
			return theta, f(theta), progress

def moving_aver(arr, n, axis = None):
	cum = np.cumsum(arr, axis=axis)
	cum[n:] = cum[n:] - cum[: -n]
	return cum[n - 1:] / n

# Plot the progress of SPSA (I know, very useful comment :) )
def plot_progress(progress, title = False, xlabel = False, ylabel = False, moving_average = False, save = False):
	# Parameters:
	# 	progress: Third output from SPSA (np.array)
	# 	title: Graph title (str)
	# 	xlabel: Label for the x axis. Use r"$$" for latex formatting (str)
	# 	ylabel: Label for the y axis. Use r"$$" for latex formatting (str)
	#	moving_average: If not False, plot the moving average with the specified n (bool / int)
	# 	save: If not False, save the graph with the name given (bool / str)
	if moving_average is not False:
		progress = moving_aver(progress, moving_average, axis = 0)
	plt.plot(progress[:, 0], progress[:, 1], color="#e100ff")
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	if title:
		plt.title(title)
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = 8
	fig_size[1] = 6
	plt.rcParams["figure.figsize"] = fig_size
	plt.grid(b = True, which = "major", linestyle = "-")
	plt.minorticks_on()
	plt.grid(b = True, which = "minor", color = "#999999", linestyle = "-", alpha = 0.2)
	if save:
		plt.savefig(save)
	plt.show()




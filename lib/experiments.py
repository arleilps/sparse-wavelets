import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.static import *
from mpl_toolkits.mplot3d import axes3d
import numpy
from matplotlib.mlab import griddata
from matplotlib import cm
from lib.syn import *
import time
import sys

def L2(F, F_approx):
	"""
		Sum of squared errors
	"""
	e = 0
	for i in range(F.shape[0]):
		e = e + ((F[i]-F_approx[i])**2).sum()

	return float(e)

def size_time_experiment(sizes, balance, sparsity, energy, noise, num):
	"""
		Size x time experiment using synthetic data.
		Input:
			* sizes: many
			* balance
			* sparsity
			* energy
			* noise
			* num: number of repetitions
		Output:
			* res_time: time results 
	"""
	res_time = []

	for s in range(len(sizes)):
		res_t = []
		for i in range(num):
			(G,F,cut) = synthetic_graph(sizes[s], 3*sizes[s], sparsity, energy, balance, noise)

			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1

			k = int(len(G.edges())*sparsity)
			start_time = time.time()
			c = one_d_search(G, F, k, ind)
			time_slow = time.time()-start_time

			start_time = time.time()
			c = fast_search(G, F, k, 5, ind)
			time_5 = time.time()-start_time
			
			start_time = time.time()
			c = fast_search(G, F, k, 20, ind)
			time_20 = time.time()-start_time
			
			start_time = time.time()
			c = fast_search(G, F, k, 50, ind)
			time_50 = time.time()-start_time

			res_t.append([time_slow, time_5, time_20, time_50])

		r = numpy.mean(numpy.array(res_t), axis=0)
		res_time.append(r)

	return numpy.array(res_time)

def sparsity_acc_experiment(sparsity, size, balance, energy, noise, num):
	"""
		Sparsity x accuracy experiments using synthetic data.
		Input:
			* sparsity: many
			* size
			* balance
			* energy
			* noise
			* num: number of repetitions
		Output:
			* res: accuracy results
	"""
	res = []

	for s in range(len(sparsity)):
		res_a = []
		for i in range(num):
			(G,F,k) = synthetic_graph(size, 3*size, sparsity[s], energy, balance, noise)
			
			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1

			c = one_d_search(G, F, k, ind)
			acc_slow = c["energy"]
			
			c = fast_search(G, F, k, 5, ind)
			acc_5 = c["energy"]
			
			c = fast_search(G, F, k, 20, ind)
			acc_20 = c["energy"]
			
			c = fast_search(G, F, k, 50, ind)
			acc_50 = c["energy"]

			res_a.append([acc_slow, acc_5, acc_20, acc_50])
		
		r = numpy.mean(numpy.array(res_a), axis=0)
		res.append(r)

	return numpy.array(res)

def noise_acc_experiment(noise, size, sparsity, energy, balance, num):
	"""
		Noise x accuracy experiments using synthetic data. 
		Input:
			* noise: many
			* size
			* sparsity
			* energy
			* balance
			* num: number of repetitions
		Output:
			* res: accuracy results
	"""
	res = []

	for s in range(len(noise)):
		res_a = []
		for i in range(num):
			(G,F,k) = synthetic_graph(size, 3*size, sparsity, energy, balance, noise[s])
			
			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1
			
			c = one_d_search(G, F, k, ind)
			acc_slow = c["energy"]
			
			L = networkx.laplacian_matrix(G)
			c = fast_search(G, F, k, 5, ind)
			acc_5 = c["energy"]
			
			c = fast_search(G, F, k, 20, ind)
			acc_20 = c["energy"]
			
			c = fast_search(G, F, k, 50, ind)
			acc_50 = c["energy"]

			res_a.append([acc_slow, acc_5, acc_20, acc_50])
		
		r = numpy.mean(numpy.array(res_a), axis=0)
		res.append(r)

	return numpy.array(res)

def energy_acc_experiment(energy, size, sparsity, noise, balance, num):
	"""
		Energy x accuracy experiments using synthetic data. 
		Input:
			* energy: many
			* size
			* sparsity
			* noise
			* balance
			* num: number of repetitions
		Output:
			* res: accuracy results
	"""
	res = []

	for s in range(len(energy)):
		res_a = []
		for i in range(num):
			(G,F,k) = synthetic_graph(size, 3*size, sparsity, energy[s], balance, noise)
			
			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1
			
			c = one_d_search(G, F, k, ind)
			acc_slow = c["energy"]
			
			c = fast_search(G, F, k, 5, ind)
			acc_5 = c["energy"]
			
			c = fast_search(G, F, k, 20, ind)
			acc_20 = c["energy"]
			
			c = fast_search(G, F, k, 50, ind)
			acc_50 = c["energy"]

			res_a.append([acc_slow, acc_5, acc_20, acc_50])
		
		r = numpy.mean(numpy.array(res_a), axis=0)
		res.append(r)

	return numpy.array(res)

def plot_size_time_experiment(results, sizes, output_file_name):
	"""
		Plots size x time experiment.
		Input:
			* results: time results
			* sizes: graph sizes
			* output_file_name: output file name	
		Output:
			* None
	"""
	plt.clf()
	
	ax = plt.subplot(111)
	
	ncol=2
	ax.plot(sizes, results[:,0], marker="x", color="cyan", label="SWT", markersize=15)
	ax.plot(sizes, results[:,1], marker="o", color="orangered", label="FSWT-5", markersize=15)
	ax.plot(sizes, results[:,2], marker="o", color="darkgreen", label="FSWT-20", markersize=15)
	ax.plot(sizes, results[:,3], marker="o", color="k", label="FSWT-50", markersize=15)
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel('time (sec.)', fontsize=30)
	ax.set_xlabel('#vertices', fontsize=30)
	ax.tick_params(labelsize=23)
	#plt.rcParams['xtick.labelsize'] = 80 
	#plt.rcParams['ytick.labelsize'] = 80
	ax.set_xlim([180,1020])
	ax.set_ylim([0.01,50000])
	ax.set_yscale('log')

	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_sparsity_acc_experiment(results, sparsity, output_file_name):
	"""
		Plots sparsity x accuracy experiment.
		Input:
			* results: accuracy results
			* sparsity: sparsity values
			* output_file_name: output file name
		Output:
			* None
	"""
	plt.clf()

	ncol=2
	ax = plt.subplot(111)
	width = 0.04       # the width of the bars)
	ax.bar(numpy.array(sparsity)-2*width, results[:,0], width, color='cyan', label="SWT", hatch="/")
	ax.bar(numpy.array(sparsity)-width, results[:,1], width, color='orangered', label="FSWT-5", hatch="\\")
	ax.bar(numpy.array(sparsity), results[:,2], width, color='darkgreen', label="FSWT-20", hatch="-")
	ax.bar(numpy.array(sparsity)+width, results[:,3], width, color='k', label="FSWT-50", hatch="*")
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ energy', fontsize=30)
	ax.set_xlabel('sparsity', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_xlim([0.1,0.9])
	ax.set_ylim([0,150])
	
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_noise_acc_experiment(results, noise, output_file_name):
	"""
		Plots noise x accuracy experiment.
		Input:
			* results: accuracy results
			* noise: noise values
			* output_file_name: output file name
		Output:
			* None
	"""
	plt.clf()
	
	ax = plt.subplot(111)
	ncol=2
	width = 0.04       # the width of the bars)
	rects1 = ax.bar(numpy.array(noise)-2*width, results[:,0], width, color='cyan', label="SWT")
	rects2 = ax.bar(numpy.array(noise)-width, results[:,1], width, color='orangered', label="FSWT-5")
	rects3 = ax.bar(numpy.array(noise), results[:,2], width, color='darkgreen', label="FSWT-20")
	rects4 = ax.bar(numpy.array(noise)+width, results[:,3], width, color='k', label="FSWT-50")

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ energy', fontsize=30)
	ax.set_xlabel('noise', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_xlim([0.1,0.9])
	ax.set_ylim([0,150])
	
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_energy_acc_experiment(results, energy, output_file_name):
	"""
		Plots energy x accuracy experiment.
		Input:
			* results: accuracy results
			* energy: energy values
			* output_file_name: output file name
		Output:
			* None
	"""
	plt.clf()
	ncol=2
	ind = numpy.array(list(range(4)))
	ax = plt.subplot(111)
	width = 0.2      # the width of the bars)
	rects1 = ax.bar(ind-width, results[:,0], width, color='cyan', label="SWT", log=True)
	rects2 = ax.bar(ind, results[:,1], width, color='orangered', label="FSWT-5", log=True)
	rects3 = ax.bar(ind+width, results[:,2], width, color='darkgreen', label="FSWT-20", log=True)
	rects4 = ax.bar(ind+2*width, results[:,3], width, color='k', label="FSWT-50", log=True)

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper left', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ energy', fontsize=30)
	ax.set_xlabel(r'L$_2$ energy (data)', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_yscale('log')
	ax.set_xticks(ind + width)
	ax.set_xticklabels((r'10$\mathregular{^1}$', r'10$\mathregular{^2}$', r'10$\mathregular{^3}$', r'10$\mathregular{^4}$'))
	ax.set_ylim(0.1,1000000)
	ax.set_xlim(-0.3,3.7)
	
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_compression_experiments(results, comp_ratios, output_file_name, max_y):
	"""
		Plots compression size x accuracy experiment.
		Input:
			* results: accuracy results
			* comp_ratios: compression ratios
			* output_file_name: output file name
			* max_y: maximum y-axis
		Output:
			* None
	"""
	plt.clf()
	
	for alg in results:
		for i in range(1,results[alg].shape[0]):
			if results[alg][i] > results[alg][i-1]:
				results[alg][i] = results[alg][i-1]	

	ax = plt.subplot(111)
	ncol = 3
	
	ax.semilogy(comp_ratios, results["FSWT"], marker="o", color="r", label="FSWT", markersize=15)
	ax.semilogy(comp_ratios, results["FT"], marker="*", color="c", label="FT", markersize=15)
	
	if "SWT" in results:
		ax.semilogy(comp_ratios, results["SWT"], marker="x", color="b", label="SWT", markersize=15)
	
	ax.semilogy(comp_ratios, results["GWT"], marker="s", color="g", label="GWT", markersize=15)
	
	if "HWT" in results:
		ax.semilogy(comp_ratios, results["HWT"], marker="v", color="y", label="HWT", markersize=15)
	
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ error', fontsize=30)
	ax.set_xlabel('size', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_xlim(0.,0.65)
	ax.set_ylim(0., max_y)

	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def compression_experiment_static(G, F, algs, comp_ratios, num):
	"""
		Runs compression experiment static.
		Input:
			* G: graph
			* F: graph signal
			* algs: compression algorithms/transforms
			* comp_ratios: compression ratios
			* num: number of repetitions
		Output:
			* results: compression results
			* times: compression times
	"""
	results = {}
	times = {}

	for alg in algs:
		results[alg.name()] = []
		times[alg.name()] = []
			
		for r in range(len(comp_ratios)):
			T = []
			R = []
			for i in range(num):
				start_time = time.time()
				alg.set_graph(G)
				tr = alg.transform(F)
				size = int(F.size * comp_ratios[r])
				appx_tr = alg.drop_frequency(tr, size)
				appx_F = alg.inverse(appx_tr)
				t = time.time()-start_time
				T.append(t)
				R.append(L2(F, appx_F))
			T = numpy.array(T)
			R = numpy.array(R)
			times[alg.name()].append(numpy.mean(T))
			results[alg.name()].append(numpy.mean(R))

		results[alg.name()] = numpy.array(results[alg.name()]) 

		times[alg.name()] = numpy.array(times[alg.name()]) 
              
	return results, times

	

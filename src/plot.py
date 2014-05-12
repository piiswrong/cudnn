import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def curve_plot(logs):

	for f, legend in logs:
	    mat = np.loadtxt(f)
	    x = mat[:,1]
	    y = np.sqrt(mat[:,0])
	    plt.plot(x, y, label=legend)


	plt.legend( loc = 'upper right' )
	plt.xlabel('Epoches')
	plt.ylabel('Error')
	#matplotlib.rcParams.update({'font.size': 32})

	plt.savefig('a.png')

def hist_plot(logs):
	y = np.fromfile("../data/RankLabel.bin", dtype = np.float32)
	i = 0
	for f, legend in logs:
		yp = np.loadtxt(f)
		print y.shape, yp.shape
		yp = abs(yp-y)
		i+=1
		plt.subplot(len(logs), 1, i)
		plt.hist(yp, label = legend)
		plt.legend( loc = 'upper right' )
	plt.xlabel('Error')
	plt.ylabel('Freq')
	#matplotlib.rcParams.update({'font.size': 32})

	plt.savefig('a.png')


logs = [ ('1.log', 'layers=1') , ('2.log', 'layers=2') , ('3.log', 'layers=3') ]
#logs = [ ('3.log', 'layers=3') ]
hist_plot(logs)


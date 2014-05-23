import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def curve_plot(logs, output):

	for f, legend in logs:
	    mat = np.loadtxt(f)
	    x = mat[:,1]
	    y = np.log10(np.sqrt(mat[:,0]))
	    plt.plot(x, y, label=legend)


	plt.legend( loc = 'upper right' )
	plt.xlabel('Epoches')
	plt.ylabel('Error')
	#matplotlib.rcParams.update({'font.size': 32})

	plt.savefig(output)

def hist_plot(logs, output):
	y = np.fromfile("../data/RankLabel.bin", dtype = np.float32)
	i = 0
	for f, legend in logs:
		yp = np.loadtxt(f)
		print y.shape, yp.shape
		yp = abs(yp-y)/abs(y)
		i+=1
		plt.subplot(len(logs), 1, i)
		plt.hist(yp, label = legend)
		plt.legend( loc = 'upper right' )
	plt.xlabel('Error')
	plt.ylabel('Freq')
	#matplotlib.rcParams.update({'font.size': 32})

	plt.savefig(output)


#logs = [ ('1.log', 'layers=1') , ('2.log', 'layers=2') , ('3.log', 'layers=3') ]
#logs = [ ('3.log', 'layers=3') ]
logs = [("2-2.log", "2-2")]
"""for l in xrange(1,3):
    for r in xrange(1,4):
        logs.append( ('%d-%d.log'%(l,r), 'generate=%d approx=%d'%(l,r)) )
print logs
"""
#curve_plot(logs, "a.png")
hist_plot(logs, "b.png")


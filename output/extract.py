import csv
import scipy.io as sio

mat = sio.loadmat('imagelabels.mat')
labels = mat['labels']

size = labels.shape[1]
print "Size {}".format(size)
labelsDict = {i:0 for i in range(1,size+1)}

for key in labelsDict:
	labelsDict[key] = labels[0][key-1]
	if labelsDict[key] == 64:
		print key, labelsDict[key]

"""
with open('labels.csv', 'w') as csvfile:
	fields = ['id','category']
	writer = csv.DictWriter(csvfile, fieldnames=fields)

	for key in labelsDict:
		writer.writerow([key, labelsDict[key]])
"""
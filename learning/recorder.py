from config import *
def get_step():
	filepath = os.path.join(train_path,'training_loss.txt') 
	if not tf.gfile.Exists(filepath):
		return 0
	file = open(filepath,'r')
	record = file.readlines()
	file.close()
	return int(record[-1].split(',')[0])

def record_loss(step,mloss,rloss,loss,task):
	if task == 0:
		filepath = os.path.join(train_path,'training_loss.txt') 
	elif task == 1:
		filepath = os.path.join(test_path,'testing_loss.txt')
	else:
		filepath = os.path.join(best_path,'minmum_loss.txt')

	file = open(filepath,'a')
	file.write('%10d,%f,%f,%f\n'%(step,mloss,rloss,loss))
	file.close()

def get_min_loss(index = 2):
	filepath = os.path.join(best_path,'minmum_loss.txt')
	if not tf.gfile.Exists(filepath):
		return 2147483647.0
	file = open(filepath,'r')
	lines = file.readlines()
	file.close()
	return float(lines[-1].split(',')[index+1])

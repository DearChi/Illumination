from PIL import Image
from struct import *
import matplotlib.image as plimg
import xml.sax,re,os
#implement xml handler
class MovieHandler( xml.sax.ContentHandler ):
   def __init__(self):
      self.CurrentData = ""
      self.str = ""
   def startElement(self, tag, attributes):
      self.CurrentData = tag
   def endElement(self, tag):
      self.CurrentData = ""
   def characters(self, content):
      if self.CurrentData == "data":
         self.str += content

#generate sh input from xml files
def xml2shinput(filename):
	parser = xml.sax.make_parser()

	parser.setFeature(xml.sax.handler.feature_namespaces, 0)

	Handler = MovieHandler()
	parser.setContentHandler( Handler )

	parser.parse(filename)

	data_str = Handler.str
	flist = []
	aList = re.findall('([-+]?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',data_str) 
	for ss in aList:
	    flist.append(float(ss[0]+ss[2]))
	return flist

class BatchFileWriter(object):
	def __init__(self, prefix, postfix, max_size=500, binary = True):
		self.hanlder = None
		self.prefix = prefix
		self.postfix = postfix
		self.max_size = max_size
		self.method = "wb" if binary else "w"

	def write(self,string):
		if self.hanlder == None:
			self.size = 0
			self.index = 0
			self.hanlder = open(self.prefix+'_%03d.'%self.index + self.postfix, self.method)
			self.index += 1

		if self.size > 0 and self.size + len(string)/1024/1024 > self.max_size:
			self.size = 0
			self.hanlder.close()
			self.hanlder = open(self.prefix+'_%03d.'%self.index + self.postfix, self.method)
			self.index += 1

		self.hanlder.write(string)
		self.size += len(string)/1024/1024

	def close(self):
		self.hanlder.close()

'''-----------------------------------------------------------------------'''

image_size = 256

data_path = '/home/dearchi/workspace/cgi/simulate_video/output'
bin_data_path = '/data/illumination/baseline/test/'

train_set = [1,2,3,4,
	6,7,
	9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,
	29,30,31,32,33,34,
	36,37,38,39,40,41,
	43,44,45,46,47,48,49,50,51,52,53,54,55,56,
	58,59,
	61,
	63,64,65,66,67,
	69,70,71,72,73,74,75,76,
	78,79,80,81,82,83,
	86,87,88,89,90,
	92,93,94,95,96,97,98
]
test_set = [16,35,42,60,62,68,77,84,85,91]
num_images_per_envmap = 60
image_length = image_size * image_size * 3
#the max size of per binary file (MB)

max_size = 1000

def gen_data(prefix, image_set):

	writer = BatchFileWriter(os.path.join(bin_data_path, prefix), 'bin',max_size = max_size)

	count = 0
	for index in image_set:
		print(">converting : ", index)
		for sample in range(num_images_per_envmap):
			print('sample',sample)

			#write sh input file
			shfile = os.path.join(data_path,'%02d'%index,'%02d.xml'%sample)
			shlist = xml2shinput(shfile)

			#write image input file
			imagefile  = os.path.join(data_path,'%02d'%index,'%02d_000.png'%(sample))
			image      = Image.open(imagefile)
			imdata     = plimg.pil_to_array(image).reshape(image_length).tolist()

			for d in range(image_length):
				imdata[d] = float(imdata[d])

			writer.write(
				pack('%df27f'%image_length,*imdata, *shlist)
			)
			print(shlist)
			count += 1

	writer.close()
	return writer.index, count

if not os.path.exists(bin_data_path):
	os.makedirs(bin_data_path)

info = open(os.path.join(bin_data_path,'data.info'),'w')

a,b = gen_data('train',train_set)
c,d = gen_data('test',test_set)

info.write(str(a)+'\n'+str(b)+'\n'+str(c)+'\n'+str(d)+'\n')

info.close()
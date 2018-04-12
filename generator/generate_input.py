from PIL import Image
from struct import *
import matplotlib.image as plimg
import xml.sax,re,os

class MovieHandler(xml.sax.ContentHandler):
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

def getSHFromXML(filename):
	parser = xml.sax.make_parser()
	parser.setFeature(xml.sax.handler.feature_namespaces, 0)

	Handler = MovieHandler()

	parser.setContentHandler( Handler )
	parser.parse(filename)

	string = Handler.str
	rawSHList = re.findall('([-+]?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',string) 
	return [float(sh[0]+sh[2]) for sh in rawSHList]

class BatchFileWriter(object):
	def __init__(self, data_path, prefix, postfix, file_size=1000, binary = True):
		self.hanlder = None
		self.data_path = data_path
		self.prefix = prefix
		self.postfix = postfix
		self.max_size = file_size
		self.method = "wb" if binary else "w"

	def write(self,data):
		if not os.path.exists(self.data_path):
			os.makedirs(self.data_path)

		if self.hanlder == None:
			self.size = 0
			self.index = 0
			self.hanlder = open(os.path.join(self.data_path,self.prefix+'_%03d.'%self.index + self.postfix), self.method)
			self.index += 1

		elif self.size > 0 and self.size + len(data)/1024/1024 > self.max_size:
			self.size = 0
			self.hanlder.close()
			self.hanlder = open(os.path.join(self.data_path,self.prefix+'_%03d.'%self.index + self.postfix), self.method)
			self.index += 1

		self.hanlder.write(data)
		self.size += len(data)/1024/1024

	def numberOfFiles(self):
		return self.index

	def close(self):
		self.hanlder.close()

'''-----------------------------------------------------------------------'''
def getSHArray(data_path, index,sample):
	xml = os.path.join(data_path,'sh','%03d'%index,'%03d.xml'%sample)
	return getSHFromXML(xml)

def getImage(data_path,index,sample,label):
	png = os.path.join(data_path,'im','%03d'%index,'%03d_%d.png'%(sample,label))
	image = Image.open(png)
	return image

def getImageArray(image,out_width,out_height):
	image = image.resize((out_width,out_height))
	image = plimg.pil_to_array(image).reshape(out_width*out_height*3).tolist()
	return [float(data) for data in image]


def main():
	setes = [
		{'prefix':'train','indices':[_ + 1 for _ in range(30)],'samples':list(range(246))},
		{'prefix':'test' ,'indices':[1,2,3,4,5],'samples':list(range(246,256))}
	]

	data_in_path = '/document/rawdata/fandb-fov60-256x256/data'
	data_out_path = '/document/data/illumination/experiment-10x32/'

	max_file_size = 500 #MB

	image_width = 256
	image_height = 256

	for group in setes:
		numberOfExamples = 0
		writer = BatchFileWriter(
			data_path = data_out_path,
			prefix=group["prefix"],
			postfix="bin",
			file_size=max_file_size)

		for index in group["indices"]:
			print(">converting : ", index)
			for sample in group["samples"]:
				sh = getSHArray(data_in_path, index,sample)

				front = getImage(data_in_path, index, sample, 0)
				bacck = getImage(data_in_path, index, sample, 1)

				front = getImageArray(front, image_width, image_height)
				bacck = getImageArray(bacck, image_width, image_height)

				data_length = len(front) + len(bacck) + len(sh)
				writer.write(
					pack('%df'%(data_length), *front, *bacck, *sh)
				)
				numberOfExamples += 1
		writer.close()

		#write data info
		info = open(os.path.join(data_out_path,'data.info'),'a')
		info.write('%d\n%d\n'%(writer.numberOfFiles(), numberOfExamples))
		info.close()

if __name__ == '__main__':
	main()
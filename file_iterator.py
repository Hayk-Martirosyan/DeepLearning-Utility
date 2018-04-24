import re
import os

class FileIterator:
    def __init__(self, inputFiles, recurseSubDirectories, pattern, listDirectories=False):
        self.inputFiles = inputFiles
        self.recurseSubDirectories = recurseSubDirectories
        self.listDirectories = listDirectories
        self.filter = re.compile(pattern) if pattern !=None  else None

    
    def list(self):
        for filename in self.inputFiles.split(' '):
            if os.path.isdir(filename):
                for (basename, fullname) in self.listDir(filename, self.filter, self.recurseSubDirectories):
                        yield (basename, fullname)
            else:
                if self.filter==None or self.filter.match(filename):
                    yield (os.path.basename(filename), filename)



    def listDir(self, inputFile, filter, recurseSubDirectories):

        for filename in os.listdir(inputFile):
            fullname = os.path.join(inputFile, filename)

            if os.path.isdir(fullname):
                if self.listDirectories:
                    if filter==None or filter.match(fullname):
                        yield (filename, fullname)
                if recurseSubDirectories:
                    for (basename, fullname) in self.listDir(fullname, filter, True):
                        yield (basename, fullname)
                #else skip
            else:
                if filter==None or filter.match(fullname):
                    yield (filename, fullname)

import os, sys
from glob import glob

dir = sys.argv[1]

def rename(dir):
    # for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
    #     title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    #     os.rename(pathAndFilename,
    #               os.path.join(dir, titlePattern % title + ext))
    length = len(dir)+4
    globbed = glob(dir)
    newName = ".fits"
    number = ""
    front = ""
    for file in globbed:
        print(len(file))
        print(length)
        print(file)
        if (len(file) == length+1):
            front = file[:-6]
            apple = file[-6:-5]
        if (len(file) == length+2):
            front = file[:-7]
            apple = file[-7:-5]
        if (len(file) == length+3):
            front = file[:-8]
            apple = file[-8:-5]
        newName = apple + newName
        for i in range(4-len(apple)):
            newName = "0"+newName
        print('rewriting' + " "+ newName)
        os.rename(file, front+newName)
        newName = ".fits"

rename(dir)

#rename(r'c:\temp\xx', r'*.doc', r'new(%s)')
#The above example will convert all *.doc files in c:\temp\xx dir to new(%s).doc, where %s is the previous base name of the file (without extension).


##############################################################################################################################3
#import shutil
# badtrackedframes = [118,125,141,172,204,236,263,292,324,353,381,414,458,475,550]
# toodim =  [285,446,447,457,459,465,522,566]
# not4or5int = [551,552,553,554,555,556,557]
#
# shitframes = badtrackedframes + toodim + not4or5int
#
# sourceBase = "/home/sko/astro120/lab3/Cordata/"
# basepath = "/home/sko/astro120/lab3/badCorData/"
# endName = ".fits"
# reason ="badTrack/"
# for fileNum in badtrackedframes:
#     source = sourceBase+str(fileNum)+endName
#     destination = basepath+reason+str(fileNum)+endName
#     shutil.move(source, destination)

# reason = "differentIntegrationTime/"
# for fileNum in not4or5int:
#     source = sourceBase+str(fileNum)+endName
#     destination = basepath+reason+str(fileNum)+endName
#     shutil.move(source, destination)

# reason = "dumb/"
# for fileNum in toodim:
#     source = sourceBase+str(fileNum)+endName
#     destination = basepath+reason+str(fileNum)+endName
#     shutil.move(source, destination)
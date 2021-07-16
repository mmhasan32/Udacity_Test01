import os
import tarfile
import urllib.request

print("downloading the Enron dataset (this may take a while)")
print("to check on progress, you can cd up one level, then execute <ls -lthr>")
print("Enron dataset should be last item on the list, along with its current size")
print("download will complete at about 423 MB")

url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
urllib.request.urlretrieve(url, filename="../../../../enron_mail_20150507.tgz")
print("download complete!")

print()
print("unzipping Enron dataset (this may take a while)")
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tgz", "r:gz")
tfile.extractall(".")
print("you're ready to go!")

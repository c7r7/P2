# file=open("file.txt")
import regex as re
file="Charan Kumar Raju !!!!somalaraju@@@@@"
# sp_c=re.sub(r'[\(\)\[\]\-\_\+\=\!\@\$\%\^\&\*\<\>\/\?]','',file).lower()
sp_c=re.sub(r'[^a-z0-9\s]','',file)
# sp_c=re.sub(r'')
# updated_file=sp_c.lower()
sp_cc=sp_c.lower().split()
print(sp_cc)


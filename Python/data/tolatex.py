data = ""
with open("metrics.txt", "r") as f:
    data= f.read()


data = data.split("\n")
newdata=""
for i,line in enumerate(data):
    if(i<2 or i>57): continue
    line=line.split()
    newline=" "
    for c in range(0, len(line)-2):
        newline+= line[c] + " & "
    newline+=line[len(line)-2] + "\\\\\n \hline\n"
    newdata+=newline
print(newdata)





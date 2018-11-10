import numpy as np

##source ="abcdef"
##target = "acfghf"

def del_cost(a,b):
      if a == b:
            return 0
      else:
            return 1
def ins_cost(a,b):
      if a == b:
            return 0
      else:
            return 1
def sub_cost(a, b):
      if a == b:
            return 0
      else:
            return 1

def EditD(source,target):
      n = len(source)
      m = len(target)
      D = []
      for  i  in range(n+1):
            D.append([])
            for j in range(m+1):
                  D[i].append( 0)
      # Initialization: the zeroth row and column is the distance from the empty string
      D[0][0] = 0
      for  i  in range(1,n+1):
            D[i][0] = D[i-1][0] + 1
      for j in range(1,m+1):
            D[0][j] = D[0][j-1] +1
            
      for  i in range(1,n+1):
            for j in range(1,m+1):
                  D[i][j] =  min([D[i-1][j]+del_cost(source[i-1],target[j-1]), D[i-1][j-1]+sub_cost(source[i-1], target[j-1]), 
                                  D[i][j-1]+ins_cost(source[i-1],target[j-1])])

      print(D[n][m])
     # Termination
      #return D[n,m]
      

#EditD("intention","execution")
#EditD("nimo","camo")
EditD("execution","")

import pandas as pd
from PIL import Image
pd.read_csv('file2.csv', error_bad_lines=False)
dataset=pd.read_csv('file2.csv')
#dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
dataset
dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
dataset
ex_rand=[]
ex_opt=[]

##OPTIMIZED
def do_analysis():
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    l=dataset.shape[1]
    fig1=plt.subplots()
    beta=0.2
    alpha=0.2
    outcome=[1 for i in range(l)]
    w=np.ones(l-2)
    w=w.tolist()
    probability=np.zeros(3).tolist()
    expert=[[0 for i in range(l)] for i in range(l-2)]

    for i in range(l):
        if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])>0):
            outcome[i]=-1
        if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])<0):
            outcome[i]=1
        if((dataset.iloc[1-2,i]-dataset.iloc[l-3,i])==0):
            outcome[i]=0

        
    for i in range(0,l-2):
        for j in range(l):
             if(dataset.iloc[i+1,j]-dataset.iloc[i,j]>0):
                 expert[i][j]=-1
             elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]<0):
                 expert[i][j]=1
             elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]==0):
                 expert[i][j]=0

    expertdf=pd.DataFrame(expert)
##expertdf = expertdf.drop([1,0],axis=0)
    expertdf

    pred=np.ones(l).tolist()

## reduce weights for |S-1|
    for i in range(l):
        w_minus=0
        w_zero=0
        w_one=0
        for j in range(l-3):
            if(expert[j][i] != outcome[j]):
                  w[j]=w[j]-w[j]*beta
      
        
    w1=list.copy(w)


    idx=0;
    for i in range(l):
         w_minus=0
         w_zero=0
         w_one=0
         for j in range(l-2):
             if(expert[j][i] != outcome[j]):
                w[j]=w[j]-w[j]*beta
       
         for j in range(l-2):
             if(expertdf.iloc[j,i]==-1):
                  w_minus=w_minus+w[j]
             elif(expertdf.iloc[j,i]==1):
                   w_one=w_one+w[j]
             elif(expertdf.iloc[j,i]==0):
                   w_zero=w_zero+w[j]
             p_one=w_one/(w_one+w_minus+w_zero)
             p_minus=w_minus/(w_one+w_minus+w_zero)
             p_zero=w_zero/(w_one+w_minus+w_zero)
          
             probability[0]=p_one
             probability[1]=p_minus
             probability[2]=p_zero
    
         
         
         y=np.random.choice(3, 1, p=probability)
         if(y==0):
            pred[i] = 1
         elif(y==1):
            pred[i] = -1
         elif(y==0): 
            pred[i] = 0
    colors = ['gold', 'yellowgreen', 'lightcoral']
    plt.pie(probability, autopct='%1.0f%%', shadow=True, colors=colors)
    plt.axis('equal')
    name="file"+str(idx)+'.png'
    plt.savefig(name)
    Image.open("file"+str(idx)+'.png').show()
    idx=idx+1

  
    import difflib
    sm=difflib.SequenceMatcher(None,pred,outcome)
    x1=sm.ratio()
    df_opt=[]
    companies=["IBM", "Microsoft", "Disney", "Johnson&Johnson","Boeing","Intel" ]
    for i in range(len(pred)):
        df_opt.append((companies[i],pred[i],outcome[i]))
    data_opt=pd.DataFrame(df_opt, columns=('Company', 'Predicted', 'Actual'))
    print(data_opt)
    print("Accuracy:" + str(x1))
    d1=pred
    
    


##RANDOMIZED 


    import matplotlib.pyplot as plty
    import numpy as np
    fig2=plty.subplots()
    l=dataset.shape[1]
    p=np.zeros(l-2).tolist()
    beta=0.2
    def all_prob(w):
         prob=np.zeros(l-2).tolist()
         c=0
         for i in w:
              prob[c]=i/total(w)
              p[c]=prob[c]
              c+=1
    def total(w):
         sum=0
         for j in w:
             sum=sum+j
         return sum
    outcome=[1 for i in range(l)]
    w=np.ones(l-2)
    w=w.tolist()
    probability=np.zeros(3).tolist()
    expert=[[0 for i in range(l)] for i in range(l-2)]

    for i in range(l):
        
        if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])>0):
              outcome[i]=-1
        if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])<0):
               outcome[i]=1
        if((dataset.iloc[1-2,i]-dataset.iloc[l-3,i])==0):
               outcome[i]=0

        
    for i in range(0,l-2):
         for j in range(l):
             if(dataset.iloc[i+1,j]-dataset.iloc[i,j]>0):
                  expert[i][j]=-1
             elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]<0):
                  expert[i][j]=1
             elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]==0):
                   expert[i][j]=0

    expertdf=pd.DataFrame(expert)

    expertdf
    x=np.zeros(l).tolist()
    idx=0
    for i in range(l):
         for j in range(l-2):
             if(expert[j][i] != outcome[j]):
                 w[j]=w[j]-w[j]*beta
         
         
    
         all_prob(w)
   
         y=np.random.choice(l-2, 1, p=p)
         x[i]=expert[y[0]][i]
    colors=['yellowgreen', 'lightcoral','pink','brown', 'yellow']
    plty.pie(w, autopct='%1.1f%%', shadow=True, colors=colors)
    plty.axis('equal')
    name="ran"+str(idx)+'.png'
    plty.savefig(name)
    Image.open("ran"+str(idx)+'.png').show()
    idx=idx+1
    #print(x)
    sm=difflib.SequenceMatcher(None,x,outcome)
    x2=sm.ratio()
    ##Create the dataframe
    companies=["IBM", "Microsoft", "Disney", "Johnson&Johnson","Boeing","Intel" ]
    df_rand=[]
    for i in range(len(x)):
        df_rand.append((companies[i],x[i],outcome[i]))
    data_rand=pd.DataFrame(df_rand, columns=('Company', 'Predicted', 'Actual'))
    print(data_rand)
    print("Accuracy:" + str(x2))
    d2=x
    y=[1,2,3,4,5,6]
    import matplotlib.pyplot as one
    fig = one.figure()
    ax1 = fig.add_subplot(111)
    y=[1,2,3,4,5,6]
    ax1.scatter(x=y, y=d1, color='blue')
    ax1.scatter(x=y, y=outcome, color='red')
    one.legend(loc='upper left');
    one.savefig("graph.png")
    Image.open("graph.png").show()
    
    import matplotlib.pyplot as two
    fig = two.figure()
    ax1 = fig.add_subplot(111)
    y=[1,2,3,4,5,6]
    ax1.scatter(x=y, y=d2, color='blue')
    ax1.scatter(x=y, y=outcome, color='red')
    two.legend(loc='upper left');
    two.savefig("graph.png")
    Image.open("graph.png").show()

do_analysis()

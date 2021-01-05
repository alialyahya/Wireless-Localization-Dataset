# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:21:40 2020

@author: mrhaboon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:23:47 2020

@author: mrhaboon
"""
import csv
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

par=3
fold=2
poly_count=1

train1=[]
test1=[]

with open('D_Train1.csv') as csv_file:
    reader=csv.reader(csv_file)
    j=0
    for i in reader:
        if j==0:
            j+=1
            continue
        else:
            train1.append([float(x) for x in i])
    train1=np.array(train1)

j=0    
with open('D_Test1.csv') as csv_file:
    reader=csv.reader(csv_file)
    for i in reader:
        if j==0:
            j+=1
            continue
        else:
            test1.append([float(x) for x in i])
    test1=np.array(test1)
    
#%%

scaler=preprocessing.StandardScaler().fit(train1)
std_train=scaler.transform(train1)
std_test=scaler.transform(test1)


normer=preprocessing.MinMaxScaler()       
norm_train=normer.fit_transform(train1)
norm_test=normer.fit_transform(test1)



class find_best_svm:
    
    def __init__(self,train,test,dim='none'):
        self.train=train
        self.train_data=self.train[:,1:]
        self.train_labels=self.train[:,0]
        self.test=test
        self.test_data=self.test[:,1:]
        self.test_labels=self.test[:,0]
        
        if dim=='none':
            self.best=self.model_search(self.train_data)
            params=self.best
                        
            print(params)
            ckernel=params[1][0]
#            print(ckernel)
#            print(params)
            if ckernel=='linear':
                self.final_model=SVC(kernel='linear')
            
            elif ckernel=='poly':
                deg=params[1][1]
                g=params[1][2]
                r=params[1][3]
                self.final_model=SVC(kernel='poly',degree=deg,gamma=g,coef0=r)
                
            elif ckernel=='rbf':
                g=params[1][1]
                c=params[1][2]
                self.final_model=SVC(C=c,kernel='rbf',gamma=g)
                
            elif ckernel=='sigmoid':
                g=params[1][1]
                r=params[1][2]
                self.final_model=SVC(kernel='sigmoid',gamma=g,coef0=r)
                
            self.final_model.fit(self.train_data,self.train_labels)
            self.performance(self.train_data,self.test_data)
            
        elif dim=='perm':
            self.gen_perm()
            self.models={}
            best=(((0,0),0),0)
            for i in self.perm:
                tmp_train=self.dim_transform(self.train_data,i)
                self.models[i]=self.model_search(tmp_train)
            for i in self.models:
                if self.models[i][0][0]>best[0][0][0]:
                    best=(self.models[i],i)
            
            self.best=best
            params=self.best[0]
            ckernel=params[1][0]
#            print(ckernel)
            if ckernel=='linear':
                self.final_model=SVC(kernel='linear')
            
            elif ckernel=='poly':
                deg=params[1][1]
                g=params[1][2]
                r=params[1][3]
                self.final_model=SVC(kernel='poly',degree=deg,gamma=g,coef0=r)
                
            elif ckernel=='rbf':
                g=params[1][1]
                c=params[1][2]
                self.final_model=SVC(C=c,kernel='rbf',gamma=g)
                
            elif ckernel=='sigmoid':
                g=params[1][1]
                r=params[1][2]
                self.final_model=SVC(kernel='sigmoid',gamma=g,coef0=r)
                                  
                    
                    
                    

            new_train=self.dim_transform(self.train_data,best[1])
            new_test=self.dim_transform(self.test_data,best[1])
            self.final_model.fit(new_train,self.train_labels)
            self.performance(new_train,new_test)
            
        elif dim=='PCA':
            best=(((0,0),0),0)
            for i in range(7):
                tmp=PCA(n_components=i+1)
                tmp.fit(self.train_data)
                tmp_train=tmp.transform(self.test_data)
                current_stat=self.model_search(tmp_train)
                if current_stat[0][0]>best[0][0][0]:
                    best=(current_stat,i+1)
            
            self.best=best
            params=self.best[0]
            ckernel=params[1][0]
#            print(ckernel)
            if ckernel=='linear':
                self.final_model=SVC(kernel='linear')
            
            elif ckernel=='poly':
                deg=params[1][1]
                g=params[1][2]
                r=params[1][3]
                self.final_model=SVC(kernel='poly',degree=deg,gamma=g,coef0=r)
                
            elif ckernel=='rbf':
                g=params[1][1]
                c=params[1][2]
                self.final_model=SVC(C=c,kernel='rbf',gamma=g)
                
            elif ckernel=='sigmoid':
                g=params[1][1]
                r=params[1][2]
                self.final_model=SVC(kernel='sigmoid',gamma=g,coef0=r)
                                  
                    
                    
                    

            tran=PCA(n_components=best[1])
            tran.fit(self.train_data)
            new_train=tran.transform(self.train_data)
            new_test=tran.transform(self.test_data)
            self.final_model.fit(new_train,self.train_labels)
            self.performance(new_train,new_test)
                
                
        elif dim=='Fisher':
            best=(((0,0),0),0)
            for i in range(3):
                tran=LinearDiscriminantAnalysis(n_components=i+1)
                tran.fit(self.train_data,self.train_labels)
                new_train=tran.transform(self.train_data)
                current_stat=self.model_search(new_train)
                if current_stat[0][0]>best[0][0][0]:
                    best=(current_stat,i+1)
                    
            self.best=best
            params=self.best[0]
            ckernel=params[1][0]
#            print(ckernel)
            if ckernel=='linear':
                self.final_model=SVC(kernel='linear')
            
            elif ckernel=='poly':
                deg=params[1][1]
                g=params[1][2]
                r=params[1][3]
                self.final_model=SVC(kernel='poly',degree=deg,gamma=g,coef0=r)
                
            elif ckernel=='rbf':
                g=params[1][1]
                c=params[1][2]
                self.final_model=SVC(C=c,kernel='rbf',gamma=g)
                
            elif ckernel=='sigmoid':
                g=params[1][1]
                r=params[1][2]
                self.final_model=SVC(kernel='sigmoid',gamma=g,coef0=r)
                                  
                    
                    
                    


            tran=LinearDiscriminantAnalysis(n_components=best[1])
            tran.fit(self.train_data,self.train_labels)
            new_train=tran.transform(self.train_data)
            new_test=tran.transform(self.test_data)
            self.final_model.fit(new_train,self.train_labels)
            self.performance(new_train,new_test)
                
        elif dim=='poly':
            best=(((0,0),0),0)
            for i in range(poly_count):
                poly=PolynomialFeatures(i+1)
                new_train=poly.fit_transform(self.train_data)
                current_stat=self.model_search(new_train)
                if current_stat[0][0]>best[0][0][0]:
                    best=(current_stat,i+1)
                    
            self.best=best
            params=self.best[0]
            ckernel=params[1][0]
#            print(ckernel)
            if ckernel=='linear':
                self.final_model=SVC(kernel='linear')
            
            elif ckernel=='poly':
                deg=params[1][1]
                g=params[1][2]
                r=params[1][3]
                self.final_model=SVC(kernel='poly',degree=deg,gamma=g,coef0=r)
                
            elif ckernel=='rbf':
                g=params[1][1]
                c=params[1][2]
                self.final_model=SVC(C=c,kernel='rbf',gamma=g)
                
            elif ckernel=='sigmoid':
                g=params[1][1]
                r=params[1][2]
                self.final_model=SVC(kernel='sigmoid',gamma=g,coef0=r)
                                  
                    
                    
                    


            poly=PolynomialFeatures(best[1])
            new_train=poly.fit_transform(self.train_data)
            new_test=poly.fit_transform(self.test_data)
            self.final_model.fit(new_train,self.train_labels)
            self.performance(new_train,new_test)
                
    def performance(self,train,test):
        self.train_conf=confusion_matrix(self.train_labels,self.final_model.predict(train))
        self.test_conf=confusion_matrix(self.test_labels,self.final_model.predict(test))
        self.cross_acc=self.best
        self.test_acc=self.final_model.score(test,self.test_labels)
#        self.train_acc=
#        self.test_acc=
    
    def gen_perm(self):
        self.perm=[]
        tmp=[]
        data_length=len(self.train_data[0])
        for i in range(data_length):
            tmp.append(i+1)
        for i in range(data_length):
            for i in combinations(tmp,i+1):
                self.perm.append(i)
      
    def dim_transform(self,data,dim):
        result=[]
        for i in data:
            tmp_data=[]
            for j in dim:
                tmp_data.append(i[j-1])
            result.append(tmp_data)
        return result
     
    def gen_modelsig(self,data,labels,g,r):
        scores=[]
        for i in range(fold):
            test_model=SVC(kernel='sigmoid',gamma=g,coef0=r)
            scores.extend(list(cross_val_score(test_model,data,labels,cv=5)))
        scores=np.array(scores)
        return (scores.mean(),scores.std()**2)
    
    def gen_modelpoly(self,data,labels,deg,g,r):
        scores=[]
        for i in range(fold):
            print(i)
            test_model=SVC(kernel='poly',degree=deg,gamma=g,coef0=r)
            scores.extend(list(cross_val_score(test_model,data,labels,cv=5)))
        print('out')
        scores=np.array(scores)
        return (scores.mean(),scores.std()**2)
    
    def gen_modelrbf(self,data,labels,g,c):
        scores=[]
        for i in range(fold):
#            print(fold)
            print(i)
            test_model=SVC(C=c,kernel='rbf',gamma=g)
            scores.extend(list(cross_val_score(test_model,data,labels,cv=5)))
        print('out')
        scores=np.array(scores)
        return (scores.mean(),scores.std()**2)
    
    def gen_modellin(self,data,labels):
        scores=[]
        for i in range(fold):
            test_model=SVC(kernel='linear')
            scores.extend(list(cross_val_score(test_model,data,labels,cv=5)))
        scores=np.array(scores)
        return (scores.mean(),scores.std()**2)

    def model_search(self,train):
        possible=['linear','poly','rbf','sigmoid']
#        possible=['rbf']
        best_model={}
        for i in possible:
            print(i)
            if i=='linear':
                best_model[i]=(self.gen_modellin(train,self.train_labels),('linear',0))
                
            elif i=='poly':
                best=((0,0),(0,0,0))
                gam_list=np.linspace(1e-3,1e3,par)
                r_list=np.linspace(1e-3,1e3,par)
                for deg in range(poly_count):
                    for g in gam_list:
                        for r in r_list:
                            curr_stat=self.gen_modelpoly(train,self.train_labels,deg+1,g,r)
                            if curr_stat[0]>best[0][0]:
                                best=(curr_stat,('poly',deg+1,g,r))
                best_model[i]=best
        
                
            elif i=='rbf':
                best=((0,0),(0,0))
                gam_list=np.linspace(1e-3,1e3,par)
                C_list=np.linspace(1e-3,1e3,par)
                for g in gam_list:
                    for C in C_list:
                        curr_stat=self.gen_modelrbf(train,self.train_labels,g,C)
                        if curr_stat[0]>best[0][0]:
                            best=(curr_stat,('rbf',g,C))
                best_model[i]=best
                
                
            elif i=='sigmoid': 
                best=((0,0),(0,0))
                gam_list=np.linspace(1e-3,1e3,par)
                r_list=np.linspace(1e-3,1e3,par)
                for g in gam_list:
                    for r in r_list:
                        curr_stat=self.gen_modelsig(train,self.train_labels,g,r)
                        if curr_stat[0]>best[0][0]:
                            best=(curr_stat,('sigmoid',g,r))
                best_model[i]=best
        overall_best=((0,0),None)
        for i in best_model:
                if best_model[i][0][0]>overall_best[0][0]:
                    overall_best=best_model[i]
        return overall_best
           
    def build_model(self,params):
        ckernel=params[1][0]
        
        if ckernel=='linear':
            return SVC(kernel='linear')
        
        elif ckernel=='poly':
            deg=params[1][1]
            g=params[1][2]
            r=params[1][3]
            return SVC(kernel='poly',degree=deg,gamma=g,coef0=r)
            
        elif ckernel=='rbf':
            g=params[1][1]
            c=params[1][2]
            return SVC(C=c,kernel='rbf',gamma=g)
            
        elif ckernel=='sigmoid':
            g=params[1][1]
            r=params[1][2]
            return SVC(kernel='sigmoid',gamma=g,coef0=r)
            
        
        
        
        
#%%
            
test=find_best_svm(train1,test1)

data_sets=[[train1,test1],[std_train,std_test],[norm_train,norm_test]]      
#dim_choice=['none','Fisher','PCA']     
dim_choice=['none']     

file1=open('SVM_results.txt','w')

i=0
for k in data_sets: 
    if i==0:
        file1.write('no std:\n')
    elif i==1:
        file1.write('std:\n')
    else:
         file1.write('norm:\n')   
    for j in dim_choice:
        print('we movin')
        file1.write('---'+j+':\n')
        tmp=find_best_svm(k[0],k[1],j)
        file1.write('Cross Validation Accuracy:'+str(tmp.cross_acc)+'\n')
        file1.write('train confusion matrix:'+str(tmp.train_conf)+'\n')
        file1.write('test set acc: '+str(tmp.test_acc)+'\n')
        file1.write('test confusion'+str(tmp.test_conf)+'\n')
        
    file1.write('-----------------------------------------------------------'+'\n')
    i+=1
            
      
    #%%
#test_model=SVC(kernel='rbf')
#scores=(cross_val_score(test_model,train1[:,1:],train1[:,0],cv=5))
##
##        
##        
        
        


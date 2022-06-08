from tkinter import Tk,StringVar,Label,Entry,OptionMenu,Button,Text,IntVar,PhotoImage,Canvas,END
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import time
import datetime
import warnings
import sqlite3
from time import strftime 
from tkinter import messagebox
warnings.simplefilter(action='ignore', category=FutureWarning)

symptoms_list=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

 

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

symptoms_count_list=[]
 
for x in range(0,len(symptoms_list)):
    symptoms_count_list.append(0)

# TRAINING DATA df -------------------------------------------------------------------------------------

df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X= df[symptoms_list]
y = df[["prognosis"]]
np.ravel(y)


# TESTING DATA tr --------------------------------------------------------------------------------

tr=pd.read_csv("Testing.csv")

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[symptoms_list]
y_test = tr[["prognosis"]]

np.ravel(y_test)
oplist=df.columns


'''' ------------------------------------------------     ALGORITHMS       ---------------------------------------------'''

def DecisionTree():

    if(Symptom1.get()=='Select Symptom'):
        t1.delete("1.0", END)
        t1.insert(END, 'Select atleast 1 Symptom')
    else:
        dec_tree_clf = tree.DecisionTreeClassifier() 
        des_tree_i_t=time.time()       
        dec_tree_clf = dec_tree_clf.fit(X,y)

        '''ACCURACY'''
     
        dec_tree_acc_score=dec_tree_clf.score(X_test,y_test)
        dec_tree_acc_score=dec_tree_acc_score*100  
        des_tree_acc.delete("1.0", END)
        des_tree_acc.insert(END, dec_tree_acc_score)

        '''TIME'''
        des_tree_pred_time= round(time.time()-des_tree_i_t, 3)
        des_tree_pred_time=str(des_tree_pred_time) + ' ' + 'milliseconds'
        des_tree_time.delete("1.0", END)
        des_tree_time.insert(END, des_tree_pred_time)

        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
  
        inputtest = [symptoms_count_list]
        predict = dec_tree_clf.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break 

        if (h=='yes'):
            t1.delete("1.0", END)
            t1.insert(END, disease[a])
        else:
            t1.delete("1.0", END)
            t1.insert(END, "Not Found")
   
def NaiveBayes():
    if(Symptom1.get()=='Select Symptom'):
        t2.delete("1.0", END)
        t2.insert(END, 'Select atleast 1 Symptom')
    else:

        NB_clf = GaussianNB()
        NB_i_t=time.time()
        NB_clf=NB_clf.fit(X,np.ravel(y))

   
        '''ACCURACY'''
        NB_acc_score=NB_clf.score(X_test,y_test)
        NB_acc_score= NB_acc_score*100
        nb_acc.delete("1.0", END)
        nb_acc.insert(END, NB_acc_score)
  
        '''TIME'''
      
        NB_pred_time= round(time.time()-NB_i_t, 3)
        NB_pred_time=str(NB_pred_time) + ' ' + 'milliseconds'
        nb_time.delete("1.0", END)
        nb_time.insert(END, NB_pred_time)       

        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
        inputtest = [symptoms_count_list]
        predict = NB_clf.predict(inputtest)
        predicted=predict[0]
        h='no'

        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            t2.delete("1.0", END)
            t2.insert(END, disease[a])
        else:
            t2.delete("1.0", END)
            t2.insert(END, "Not Found")

 
def knn():

    if(Symptom1.get()=='Select Symptom'):
        t3.delete("1.0", END)
        t3.insert(END, 'Select atleast 1 Symptom')
    else:

        knn_clf = KNeighborsClassifier(n_neighbors=7)
        knn_i_t=time.time()
        knn_clf.fit(X, np.ravel(y))

        '''ACCURACY'''

        knn_acc_score=knn_clf.score(X_test,y_test)
        knn_acc_score=knn_acc_score*100
        knn_acc.delete("1.0", END)
        knn_acc.insert(END, knn_acc_score)

        '''TIME'''

        knn_pred_time= round(time.time()-knn_i_t, 3)
        knn_pred_time=str(knn_pred_time) + ' ' + 'milliseconds'
        knn_time.delete("1.0", END)
        knn_time.insert(END, knn_pred_time)       

        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(symptoms_list)):
            for z in psymptoms:
                if(z==symptoms_list[k]):
                    symptoms_count_list[k]=1
        inputtest = [symptoms_count_list]
        predict = knn_clf.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
  
        if (h=='yes'):
            t3.delete("1.0", END)
            t3.insert(END, disease[a])
        else:
            t3.delete("1.0", END)
            t3.insert(END, "Not Found")         

        
        
        '''   DATABASE OPERATIONS   '''


def save_reports():
    
    NAME = NameEn.get()
    AGE = age.get()
    
    GENDER = Gender.get()
    
    if(GENDER==0):
        GENDER="MALE"
    else:
        GENDER="FEMALE"
   
    SYMPTOM_1 = Symptom1.get()
    SYMPTOM_2 = Symptom2.get()
    SYMPTOM_3 = Symptom3.get()
    SYMPTOM_4 = Symptom4.get()
    SYMPTOM_5 = Symptom5.get()
    
    DECISION_TREE_PREDICTION = t1.get('1.0', END)
    DECISION_TREE_ACCURACY = des_tree_acc.get('1.0', END)
    DECISION_TREE_TIME = des_tree_time.get('1.0', END)
    
    NAIVE_BAYES_PREDICTION = t2.get('1.0', END)
    NAIVE_BAYES_ACCURACY = nb_acc.get('1.0', END)
    NAIVE_BAYES_TIME = nb_time.get('1.0', END)
    
    KNN_PREDICTION = t3.get('1.0', END)
    KNN_ACCURACY = knn_acc.get('1.0', END)
    KNN_TIME = knn_time.get('1.0', END)
    
    '''VALIDATIONS'''
    
    if(NAME==''):
        messagebox.showinfo("MACHINE LEARNING DISEASE PREDICTOR.", "NAME can't be empty !!!")
    
    elif(AGE==''):
        messagebox.showinfo("MACHINE LEARNING DISEASE PREDICTOR.", "AGE can't be empty !!!")
 
    
    elif(SYMPTOM_1=='Select Symptom'):
        messagebox.showinfo("MACHINE LEARNING DISEASE PREDICTOR.", "Select atleast 1 SYMPTOM !!!")

    elif((len(DECISION_TREE_PREDICTION)==1) and (len(NAIVE_BAYES_PREDICTION)==1) and (len(KNN_PREDICTION)==1)):
        messagebox.showinfo("MACHINE LEARNING DISEASE PREDICTOR.", "Use atleast 1 ALGORITHM !!!")

    
    else:
        
        '''    NIL VALUES FOR TEXT BOXES '''
        
        
        if(len(DECISION_TREE_PREDICTION)==1):
            DECISION_TREE_PREDICTION="NIL"
        
        if(len(DECISION_TREE_ACCURACY)==1):
            DECISION_TREE_ACCURACY="NIL"

        if(len(DECISION_TREE_TIME)==1):
            DECISION_TREE_TIME="NIL"        
        
        
        
        if(len(NAIVE_BAYES_PREDICTION)==1):
            NAIVE_BAYES_PREDICTION="NIL"
        
        if(len(NAIVE_BAYES_ACCURACY)==1):
            NAIVE_BAYES_ACCURACY="NIL"

        if(len(NAIVE_BAYES_TIME)==1):
            NAIVE_BAYES_TIME="NIL"        
        
        
        
        if(len(KNN_PREDICTION)==1):
            KNN_PREDICTION="NIL"
        
        if(len(KNN_ACCURACY)==1):
            KNN_ACCURACY="NIL"

        if(len(KNN_TIME)==1):
            KNN_TIME="NIL"
        
        
        if(SYMPTOM_2=='Select Symptom'):
            SYMPTOM_2="NIl"
            
        if(SYMPTOM_3=='Select Symptom'):
            SYMPTOM_3="NIl"

        if(SYMPTOM_4=='Select Symptom'):
            SYMPTOM_4="NIl"

        if(SYMPTOM_5=='Select Symptom'):
            SYMPTOM_5="NIl"
            
        
        
        con = sqlite3.connect('DISEASE_PREDICTOR.db')
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS DISEASE_PREDICTOR (REPORT_ID INTEGER PRIMARY KEY,NAME varchar(50),AGE INT,GENDER varchar(50),SYMPTOM_1 varchar(50),SYMPTOM_2 varchar(50),SYMPTOM_3 varchar(50),SYMPTOM_4 varchar(50),SYMPTOM_5 varchar(50),DECISION_TREE_PREDICTION varchar(50),DECISION_TREE_ACCURACY varchar(50),DECISION_TREE_TIME varchar(50),NAIVE_BAYES_PREDICTION varchar(50),NAIVE_BAYES_ACCURACY varchar(50),NAIVE_BAYES_TIME varchar(50),KNN_PREDICTION varchar(50),KNN_ACCURACY varchar(50),KNN_TIME varchar(50) )") 
    
        try:
            cur.execute("""INSERT INTO DISEASE_PREDICTOR(NAME,AGE,GENDER,SYMPTOM_1,SYMPTOM_2,SYMPTOM_3,SYMPTOM_4,SYMPTOM_5,
                        DECISION_TREE_PREDICTION,DECISION_TREE_ACCURACY,DECISION_TREE_TIME,
                        NAIVE_BAYES_PREDICTION,NAIVE_BAYES_ACCURACY,NAIVE_BAYES_TIME,KNN_PREDICTION,KNN_ACCURACY,KNN_TIME ) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (NAME,AGE,GENDER,SYMPTOM_1,SYMPTOM_2,SYMPTOM_3,SYMPTOM_4,SYMPTOM_5,DECISION_TREE_PREDICTION,DECISION_TREE_ACCURACY,
                        DECISION_TREE_TIME,NAIVE_BAYES_PREDICTION,NAIVE_BAYES_ACCURACY,NAIVE_BAYES_TIME,KNN_PREDICTION,KNN_ACCURACY,KNN_TIME))
    
            messagebox.showinfo("MACHINE LEARNING DISEASE PREDICTOR.", "Record added successfully !!!")
            con.commit()
            con.close() 
            
        except sqlite3.DatabaseError as err:
                print(err)
                con.close() 
            
            
''''--------------------------------------------------GRAPHICAL USER INTERFACE-------------------------------'''


root = Tk()
#root.attributes("-fullscreen", True)
root.title('MACHINE LEARNING DISEASE PREDICTOR.')


'''   BACKGROUND IMAGE '''

img = PhotoImage(file="Background.png")
canvas = Canvas(root, height=768, width=1366)
canvas.create_image(0, 0, image=img, anchor='nw')

canvas.pack()

'''CLOCK GOES HERE '''

Time_label = canvas.create_text((1180,70), text="Time : ",  font='Arial 13 bold',fill="white") 
clock_label = Label(text="",fg="white", font='Arial 13 bold',bg="#3b6bc1") 
clock_label.place(x=1210,y=60)


def ctime(): 
	string = strftime('%I:%M:%S %p') 
	clock_label.config(text = string) 
	clock_label.after(1000, ctime) 
ctime() 


'''DATE GOES HERE '''

date_object = datetime.date.today()
Date_label = canvas.create_text((1180,30),text="Date : ", fill="white", font='Arial 13 bold') 
 

dateD_label = Label(text="", fg="white", font='Arial 13 bold',bg="#3b6bc1") 
dateD_label.place(x=1210,y=20)
dateD_label.configure(text=date_object)


'''FULL SCREEN FUNCTION CALL'''

def close_window():
    root.destroy()

# entry variables

Symptom1 = StringVar()
Symptom1.set('Select Symptom')
Symptom2 = StringVar()
Symptom2.set('Select Symptom')
Symptom3 = StringVar()
Symptom3.set('Select Symptom')
Symptom4 = StringVar()
Symptom4.set('Select Symptom')
Symptom5 = StringVar()
Symptom5.set('Select Symptom')

Name = StringVar()
Age = StringVar()
Gender = IntVar()

'''   MAIN HEAGING   '''

w2 = canvas.create_text((680,50), text="A GENERAL DISEASE PREDICTION MACHINE.", font='Arial 20 bold',fill="white")

''' NAME OF PATIENT LABEL AND TEXTBOX '''

NameLb = canvas.create_text((350,100), text="Name of the Patient", fill="white",  font='Arial 13 bold')
NameEn = Entry(root, textvariable=Name,width=43,  font='Arial 11 ')
NameEn.place(x=600,y=100)

 
''' AGE OF PATIENT LABEL AND TEXTBOX '''

AgeLb = canvas.create_text((290,140), text="Age", fill="white",  font='Arial 13 bold')
age = Entry(root, textvariable=Age,width=43,  font='Arial 11 ')
age.place(x=600,y=140)

''' GENDER '''


import tkinter.ttk as ttk
radio_style = ttk.Style()        
myColor = '#2780d1'               
radio_style.configure('Wild.TRadiobutton',background=myColor,foreground='white', font='Arial 13')         




gender_label = canvas.create_text((305,180), text="Gender", fill="white",  font='Arial 13 bold')

R1 = ttk.Radiobutton(root, text="Male", variable=Gender, value=0, style = 'Wild.TRadiobutton')

R1.place(x=600,y=180)
R2 = ttk.Radiobutton(root, text="Female", variable=Gender, value=1, style = 'Wild.TRadiobutton')
R2.place(x=800,y=180)


''' SYMPTOM LABELS AND OPTION BUTTONS '''


''' LABELS'''

S1Lb = canvas.create_text((320,220), text="Symptom 1", fill="white", font='Arial 13 bold')
S2Lb = canvas.create_text((320,260), text="Symptom 2", fill="white",  font='Arial 13 bold')
S3Lb = canvas.create_text((320,300), text="Symptom 3", fill="white",  font='Arial 13 bold')
S4Lb = canvas.create_text((320,340), text="Symptom 4", fill="white",  font='Arial 13 bold')
S5Lb = canvas.create_text((320,380), text="Symptom 5", fill="white",  font='Arial 13 bold')


'''OPTIONS'''

OPTIONS = sorted(oplist)


S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.place(x=600,y=220)
S1En.config(width=50)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.place(x=600,y=260)
S2En.config(width=50)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.place(x=600,y=300)
S3En.config(width=50)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.place(x=600,y=340)
S4En.config(width=50)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.place(x=600,y=380)
S5En.config(width=50)




'''OPTION MENU CHANGE EVENT '''

def option_menu_changed(*args):
    
    
    t1.delete('1.0', END)
    t2.delete('1.0', END)
    t3.delete('1.0', END)
    
    des_tree_acc.delete('1.0', END)
    nb_acc.delete('1.0', END)
    knn_acc.delete('1.0', END)
    
    des_tree_time.delete('1.0', END)
    nb_time.delete('1.0', END)
    knn_time.delete('1.0', END)
    
    
    
Symptom1.trace('w', option_menu_changed)
Symptom2.trace('w', option_menu_changed)
Symptom3.trace('w', option_menu_changed)
Symptom4.trace('w', option_menu_changed)
Symptom5.trace('w', option_menu_changed)




''' ALGORITHM NAME LABELS AND TEXTBOX AND BUTTONS'''

'''LABELS'''

lrLb = Label(root, text="Decision Tree", fg="white", bg="red", font='Arial 13',width=20)
lrLb.place(x=280,y=440)

destreeLb = Label(root, text="Naive Bayes", fg="white", bg="red", font='Arial 13',width=20)
destreeLb.place(x=280,y=480)


ranfLb = Label(root, text="KNeighborsClassifier", fg="white", bg="red", font='Arial 13',width=20)
ranfLb.place(x=280,y=520)

 
'''TEXTBOXES'''

t1 = Text(root, height=1, width=42,bg="orange",fg="black", font='Arial 11')
t1.place(x=600,y=440)

t2 = Text(root, height=1, width=42,bg="orange",fg="black", font='Arial 11')
t2.place(x=600,y=480)

t3 = Text(root, height=1, width=42,bg="orange",fg="black", font='Arial 11')
t3.place(x=600,y=520)

 
''' BUTTONS   '''

dst = Button(root, text="Decision Tree", command=DecisionTree,bg="green",fg="white", font='Arial 11', width=20)
dst.place(x=980,y=430)
rnf = Button(root, text="Naive Bayes", command=NaiveBayes,bg="green",fg="white", font='Arial 11', width=20)
rnf.place(x=980,y=470)

lr = Button(root, text="K N N", command=knn,bg="green",fg="white", font='Arial 11', width=20)
lr.place(x=980,y=510)

'''ALGORITHM ACCURACY SCORE AND CALCULATION TIME '''

'''ACCURACY LABELS'''

DT_acc_lb = Label(root, text="Decision Tree Accuracy", fg="white", bg="blue", font='Arial 13',width=25)
DT_acc_lb.place(x=280,y=580)

NB_acc_lb = Label(root, text="Naive Bayes Accuracy", fg="white", bg="blue", font='Arial 13',width=25)
NB_acc_lb.place(x=280,y=620)

KNN_acc_lb = Label(root, text="KNeighborsClassifier Accuracy",  bg="blue",fg="white", font='Arial 13',width=25)
KNN_acc_lb.place(x=280,y=660)

'''ACCURACY TEXTBOXES'''

des_tree_acc = Text(root, height=1, width=27,bg="orange",fg="black", font='Arial 11')
des_tree_acc.place(x=600,y=580)

nb_acc = Text(root, height=1, width=27,bg="orange",fg="black", font='Arial 11')
nb_acc.place(x=600,y=620)

knn_acc = Text(root, height=1, width=27,bg="orange",fg="black", font='Arial 11')
knn_acc.place(x=600,y=660)

'''TIME LABELS AND TEXTBOXES'''

'''TIME LABELS'''

des_tree_time_lb = Label(root, text="Time", fg="white", bg="blue", font='Arial 13',width=10)
des_tree_time_lb.place(x=850,y=580)

nb_time_lb = Label(root, text="Time", fg="white", bg="blue", font='Arial 13',width=10)
nb_time_lb.place(x=850,y=620)

knn_time_lb = Label(root, text="Time", fg="white", bg="blue", font='Arial 13',width=10)
knn_time_lb.place(x=850,y=660)

'''TIME TEXT BOX'''

des_tree_time = Text(root, height=1, width=20,fg="black", font='Arial 11',bg="orange")
des_tree_time.place(x=980,y=580)

nb_time = Text(root, height=1, width=20,fg="black", font='Arial 11',bg="orange")
nb_time.place(x=980,y=620)

knn_time = Text(root, height=1, width=20,fg="black", font='Arial 11',bg="orange")
knn_time.place(x=980,y=660)

'''SAVE'''

save_buttn = Button(root, text="SAVE",bg="red",fg="white", font='Arial 11 bold',width=10,command=save_reports)
save_buttn.place(x=580,y=720)

 
'''CLOSE BUTTON'''

lr = Button(root, text="CLOSE", command=close_window,bg="red",fg="white", font='Arial 11 bold')
lr.place(x=1280,y=720)




def view_reports():

    import subprocess
    command = ("View Reports.py")
    subprocess.Popen(command, shell=True)
    

    '''PREVIOUS REPORTS BUTTON'''
    
    
def clear_options():
    Symptom1.set('Select Symptom')
    Symptom2.set('Select Symptom')
    Symptom3.set('Select Symptom')
    Symptom4.set('Select Symptom')
    Symptom5.set('Select Symptom')


        

reports = Button(root, text="PREVIOUS REPORTS",bg="red",fg="white", font='Arial 11 bold',width=20,command=view_reports)
reports.place(x=720,y=720)


clear_options = Button(root, text="CLEAR",bg="red",fg="white", font='Arial 11 bold',width=10,command=clear_options)
clear_options.place(x=1000,y=300)


root.mainloop()


import numpy as np
import pandas as pd
import datetime
import pickle
import csv
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter as tk
from PIL import Image,ImageTk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

WINDOW = tk.Tk()
WINDOW.title("Car_Price_Predictor")
WINDOW.geometry('818x665')

bg = Image.open('images/front.png')
bg = ImageTk.PhotoImage(bg)
bg_s = tk.Label(WINDOW,image=bg).place(x = 0,y = 0)

x = ['symboling', 'fuel-type', 'aspiration', 'doors', 'engine-location',
       'wheel-base', 'length', 'width', 'height', 'curb-weight',
       'no-of-cylinder', 'engine-size', 'bore', 'stroke', 'compression-ratio',
       'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg',
       'brand_alfa-romero', 'brand_audi', 'brand_bmw', 'brand_chevrolet',
       'brand_dodge', 'brand_honda', 'brand_isuzu', 'brand_jaguar',
       'brand_mazda', 'brand_mercedes-benz', 'brand_mercury',
       'brand_mitsubishi', 'brand_nissan', 'brand_peugot', 'brand_plymouth',
       'brand_porsche', 'brand_renault', 'brand_saab', 'brand_subaru',
       'brand_toyota', 'brand_volkswagen', 'brand_volvo', 'body-style_hardtop',
       'body-style_hatchback', 'body-style_sedan', 'body-style_wagon',
       'drive-wheels_fwd', 'drive-wheels_rwd', 'engine-type_dohcv',
       'engine-type_l', 'engine-type_ohc', 'engine-type_ohcf',
       'engine-type_ohcv', 'engine-type_rotor', 'fuel-system_2bbl',
       'fuel-system_4bbl', 'fuel-system_idi', 'fuel-system_mfi',
       'fuel-system_mpfi', 'fuel-system_spdi']

class drop_down():
    def __init__(self,WINDOWS,feathers,feather_values,x_cor,y_cor):
        self.WINDOWS = WINDOWS
        self.feathers = feathers
        self.feather_values = feather_values
        self.x_cor = x_cor
        self.y_cor = y_cor
        self.str_ = tk.StringVar()

        self.feathers = ttk.Combobox(self.WINDOWS, width = 18 , textvariable = self.str_) 
        self.feathers['values'] = self.feather_values
        self.feathers.place(x = self.x_cor,y = self.y_cor)

class entry_text():
    def __init__(self,WINDOWS,feathers,x_cor,y_cor):
        self.WINDOWS = WINDOWS
        self.feathers = feathers
        self.x_cor = x_cor
        self.y_cor = y_cor

        self.feathers = tk.Entry(self.WINDOWS, borderwidth=2)
        self.feathers.place(x = self.x_cor, y = self.y_cor,height=20,width=132)

     
symboling = drop_down(WINDOW,'symboling',('-3','-2','-1','0','1','2','3'),30,110)

brand = drop_down(WINDOW,'brand',('Alfa-Romero','Audi','BMW','Chevrolet','Dodge',
                                   'Honda','Isuzu','Jaguar','Mazda','Mercedes-Benz',
                                   'Mercury','Mitsubishi','Nissan','Peugot','Plymouth',
                                   'Porsche','Renault','Sabb','Subaru','Toyota','Volkswagen',
                                   'Volvo','Others'),250,110)

door = drop_down(WINDOW,'doors',('2','4','6'),30,180)

fuel = drop_down(WINDOW,'fuel',('Gas','Diesel'),250,180)

aspiration = drop_down(WINDOW,'aspiration',('Std','Turbo'),30,250)

driven = drop_down(WINDOW,'driven_wheels',('4wd','fwd','rwd','awd'),250,250)

body = drop_down(WINDOW,'body_style',('convertible','Hardtop','Hatchback','Sedan','Wagon'),30,320)

engine_l = drop_down(WINDOW,'engine_location',('Front','Rear'),250,320)

engine_t = drop_down(WINDOW,'engine_type',('dohc','I','ohc','ohcf','ohcv','rotor'),250,390)

cylinder = drop_down(WINDOW,'no_cylinder',('2','4','6','8','10','12'),30,390)

fuel_s = drop_down(WINDOW,'fuel_system',('1bbl','2bbl','idi','mfi','mpfi','spdi','spfi'),30,460)



wheel_b = entry_text(WINDOW,'wheel_b',465,110)
length = entry_text(WINDOW,'length',465,180)
height = entry_text(WINDOW,'height',465,250)
width = entry_text(WINDOW,'width',465,320)
curb_w = entry_text(WINDOW,'curb_w',465,390)
bore = entry_text(WINDOW,'bore',465,460)
stroke = entry_text(WINDOW,'stroke',465,530)
compress_r = entry_text(WINDOW,'compress_r',30,600)
horse_p = entry_text(WINDOW,'horse_p',30,530)
p_rpm = entry_text(WINDOW,'p_rpm',250,530)
c_mpg = entry_text(WINDOW,'c_mpg',250,600)
h_mpg = entry_text(WINDOW,'h_mpg',250,460)
engine_s = entry_text(WINDOW,'engine_s',465,600)

def price_predict():
    global predicted
    
    try:
        predicted.destroy()
    except:
        pass
    
    try:
        data_file = open('data/Old_Car_Price.csv','a+',newline='')

        data = [[int(symboling.feathers.get()), brand.feathers.get().lower(), int(fuel.feathers.current()+1),
                int(aspiration.feathers.current()+1), int(door.feathers.get()), body.feathers.get().lower(),
                driven.feathers.get(), int(engine_l.feathers.current()+1), float(wheel_b.feathers.get()),
                float(length.feathers.get()), float(width.feathers.get()), float(height.feathers.get()),
                int(curb_w.feathers.get()),engine_t.feathers.get().lower(), int(cylinder.feathers.get()),
                int(engine_s.feathers.get()), fuel_s.feathers.get().lower(), float(bore.feathers.get()),
                float(stroke.feathers.get()), float(compress_r.feathers.get()), int(horse_p.feathers.get()),
                int(p_rpm.feathers.get()),int(c_mpg.feathers.get()),int(h_mpg.feathers.get())]]
        
        with data_file:
            write = csv.writer(data_file)
            write.writerows(data)

        data_collected = True
        
    except:
        pass
        
    try:
        if data_collected:   
            dataset = pd.read_csv('data/Old_Car_Price.csv')
            datas = dataset.iloc[:, :-1]
            
            datas = pd.get_dummies(datas,
                                 columns = ['brand','body-style','drive-wheels','engine-type','fuel-system'],
                                 drop_first = True)
            
            missing_cols = set(x) - set(datas.columns)
            for col in missing_cols:
                datas[col] = 0
            datas = datas[x]
            
            standardScaler = StandardScaler()
            standardScaler.fit(datas)
            datas = standardScaler.transform(datas)

            model = pickle.load(open('model/old_car_price_RF.ml','rb'))
            pred = model.predict(datas)
            
            predicted = tk.Button(text=str(int(pred[-1])),font = ('times new roman', 19) ,bg = 'black', fg = 'red')
            predicted.place(x = 670, y = 350)

            data_set = dataset.drop(dataset.index[205])
            data_set.to_csv('data/Old_Car_Price.csv', index=False)

    except:
        pass
    
predict = tk.Button(text='Predict Prize',font = ('times new roman',15),bg = 'black',fg = 'red',command=lambda:price_predict()).place(x=657,y=585)
warn = tk.Button(text=('WARNING \n Enter Valid Data \n Enter All Feathers'),font = ('times new roman', 17) ,bg = 'black', fg = 'red')
warn.place(x = 615, y = 165)

WINDOW.mainloop()

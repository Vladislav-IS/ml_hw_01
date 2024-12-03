from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import re
import pickle
import uvicorn
import datetime


# глобальные переменные
fuel_density = {'Petrol': 0.735, 'Diesel': 0.825}
g = 9.8
model = pickle.load(open('best_model.pkl', 'rb'))
app = FastAPI()


# код ниже автоматически запускает сервер
if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# здесь и далее - функции для предобработки датасета пере загрузкой в модель
def parse_torque(x):
    x = str(x).lower().replace(',', '.')
    rpm = ''
    nm = ''
    full_regex = '\d{1,}\.?\d* *@ \d{1,}.?\d*(-|\+/-|\~)?\d*.?\d*\((nm|kgm)@ ?rpm\)'
    matching = re.match(full_regex, x)
    if matching:
        rpm = x.split('@')[1].replace('(kgm', '').replace('(nm', '') + 'rpm'
        if x.find('kgm') < 0:
            nm = x.split('@')[0] + 'nm'
        else:
            nm = str(float(x.split('@')[0]) * g) + 'nm '
        return nm + rpm
    full_regex = '\d{1,}\.?\d* */ ?\d{1,}\.?\d*'
    matching = re.match(full_regex, x)
    if matching:
        nm = x.split('/')[0] + 'nm '
        rpm = x.split('/')[1] + 'rpm'
        return nm + rpm
    full_regex = '\d{1,}\.?\d* *(nm|kgm)?@ \d{1,}.?\d*(-|\+/-|\~)?\d*.?\d*'
    matching = re.match(full_regex, x)
    if matching:
        if x.find('kgm') < 0:
            nm = x.split('@')[0].replace('nm', '') + 'nm '
        else:
            nm = str(float(x.split('@')[0].replace('kgm', '')) * g) + 'nm'
        rpm = x.split('@')[1].replace('rpm', '') + 'rpm'
        return nm + rpm
    matching = re.search('\d{1,}.?\d*(-|\+/-|\~)?\d* *rpm', x)
    if matching:
        rpm = matching.group().replace(' rpm', 'rpm')
    matching = re.search('\d{1,}\.?\d* *(nm|kgm)', x)
    if matching:
        if matching.group().find('kgm') < 0:
            nm = matching.group()
        else:
            kgm = float(matching.group().replace('kgm', '').strip())
            nm = str(kgm * g) + 'nm '
    if nm != '':
        return nm + rpm
    elif rpm != '':
        return nm + rpm
    else:
        return np.nan


def get_nm(x):
    return float(str(x).split('nm')[0].strip()) if str(x).find('nm') > 0 else np.nan


def get_rpm(x):
    if str(x).find('rpm') < 0:
        return np.nan
    rpm = str(x).split('nm')[1].replace('rpm', '')
    if rpm.find('+/-') >= 0:
        return float(rpm.split('+/-')[0].replace('.', '')) / 2
    if rpm.find('-') >= 0 or rpm.find('~') >= 0:
        rpm = [float(d.replace('.', '')) for d in re.split('[-~]', rpm)]
        return (rpm[0] + rpm[1]) / 2
    return float(rpm.replace('.', ''))


def remove_units(x):
    matching = re.match('\d{1,}\.?\d*', str(x))
    return float(matching.group()) if matching else np.nan


def remove_units_transformer(df):
    df_copy = df.copy()
    for col in ['mileage', 'engine', 'max_power']:
        df_copy[col] = df_copy[col].apply(remove_units)
    return df_copy


def equalize_mileage_transformer(df):
    for fuel, density in fuel_density.items():
        new_mileage = df.loc[df['fuel'] == fuel, 'mileage'] / density
        df.loc[df['fuel'] == fuel, 'mileage'] = new_mileage
    return df


def parse_torque_transformer(df):
    parsed_torque = df['torque'].apply(parse_torque)
    df['torque'] = parsed_torque.apply(get_nm)
    df['torque_max_rpm'] = parsed_torque.apply(get_rpm)
    return df


def parse_name_transformer(df):
    df['name'] = df['name'].apply(lambda x: ' '.join(x.split()[:2]))
    return df


def change_types_transformer(df):
    df['engine'] = df['engine'].astype('Int64')
    df['seats'] = df['seats'].astype(str)
    return df


def append_cols_transformer(df):
    df['year_squared'] = df['year'] ** 2
    df['km_driven_squared'] = df['km_driven'] ** 2
    df['bhp_per_cc'] = df['max_power'] / df['engine']
    df['is_new'] = df['owner'] == 'Test Drive Car'
    return df


table_editor = Pipeline(steps=[
    ('units_remover', FunctionTransformer(remove_units_transformer)),
    ('mileage_equalizer', FunctionTransformer(equalize_mileage_transformer)),
    ('torque_parser', FunctionTransformer(parse_torque_transformer)),
    ('name_parser', FunctionTransformer(parse_name_transformer)),
    ('columns_appender', FunctionTransformer(append_cols_transformer)),
    ('types_changer', FunctionTransformer(change_types_transformer))
])

    
# определение моделей для входных и выходных данных
class Item(BaseModel):
    name: str = Field(pattern=r'.* .+')
    year: int = Field(gt=1900)
    km_driven: int = Field(ge=0)
    fuel: str = Field(min_length=1)
    seller_type: str = Field(min_length=1)
    transmission: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    mileage: str = Field(pattern=r'\d{1,}\.?\d* ?(kmpl|km/kg)')
    engine: str = Field(pattern=r'\d{1,}\.?\d* ?(cc|CC)')
    max_power: str = Field(pattern=r'\d{1,}\.?\d* ?(bhp|BHP)')
    torque: str = Field(min_length=1)
    seats: float = Field(gt=0)


class Items(BaseModel):
    objects: List[Item]


class Predictions(BaseModel):
    pred_list: List[float]


# определение функций для GET- и POST-запросов
@app.get("/")
async def root():
    time = datetime.datetime.now()
    return {'message': f'FastAPI service is ready to use, start time: {time}'}


# входные данные - одиночный объект Item
@app.post("/predict_item", response_model=Predictions)
async def predict_item(item: Item) -> float:
    data = table_editor.transform(pd.DataFrame(jsonable_encoder(item), index=[0]))
    return {'pred_list': model.predict(data)}


# входные данные - массив объектов Item
@app.post("/predict_items", response_model=Predictions)
async def predict_items(items: List[Item]) -> List[float]:
    data = table_editor.transform(pd.DataFrame(jsonable_encoder(items)))
    return {'pred_list': model.predict(data)}


# входные данные - csv-файл
@app.post("/predict_items_csv", response_class=FileResponse)
async def predict_items_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, index_col='Unnamed: 0')
    data = table_editor.transform(df)
    df['selling_price'] = model.predict(data) 
    df.to_csv(file.filename)
    response = FileResponse(file.filename, media_type='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
    return response
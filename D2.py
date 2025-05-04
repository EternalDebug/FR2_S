# -*- coding: windows-1251 -*-
from re import I
from uu import decode
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import torch
from torch import nn
from sys import argv
from fastapi import FastAPI
import uvicorn
from ast import Await, literal_eval
import llama_cpp
import os.path
from llama_cpp import Llama
import asyncio
import NeuralNetworks
from navec import Navec
import math

app = FastAPI()

print(llama_cpp.llama_supports_gpu_offload())

#Инициализация линейных моделей

df = pd.read_csv("data3.csv", sep='\t') # чтение данных

train, test = train_test_split(df, test_size=0.3, random_state=42) #разбиение на выборки

vectorizer = TfidfVectorizer()
vectorizer.fit(df['title'])

model_lin = RidgeCV()
model_lin.fit(vectorizer.transform(train['title']), train['score'])

def GetLinearSent(text): #Получить сентимент от линейной модели
    res = model_lin.predict(vectorizer.transform([text]))
    return res[0]

preprocessor = ColumnTransformer(
    transformers=[
        ('text', vectorizer, 'title'),  # Обработка текстового признака
        ('num', 'passthrough', ['score'])  # Прямое использование численного признака
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Предобработка признаков
    ('model', RidgeCV())  # Обучающая модель
])

pipeline.fit(train[['title', 'score']], train['percent'])

def GetLinearPerc(text): # получить процент от линейной модели

    sent = GetLinearSent(text)
    new_data = pd.DataFrame({
    'title': [text],
    'score': [sent]
    })

    # Получаем предсказание
    prediction = pipeline.predict(new_data)
    return prediction[0]
    

# КОНЕЦ ИНИЦИАЛИЗАЦИИ ЛИНЕЙНЫХ МОДЕЛЕЙ
# Нейронные сети

# Выбираем девайс
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f'Device: {device}')

path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

#path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

model_sent = NeuralNetworks.WordsGRUHybridSent(300, 1).to(device)
model_sent.load_state_dict(torch.load("model_GRU_HybridData.tar", weights_only=True))
model_sent.eval()

model_perc = NeuralNetworks.WordsGRUHybridPerc(300, 1).to(device)
model_perc.load_state_dict(torch.load("model_GRU_EXP_bidir_finance_2.tar", weights_only=True))
model_perc.eval()

#Конец инициализации

#Большая генеративная нейросеть LLama - 3.1

inst = "Ты финансовый аналитик. Оцениваешь сентимент новости и прогнозируешь изменение курса акций упомянутой в новости компании. Ответы на русском и коротко. Сентимент - число от 0 до 1, от 0 до 0.5 - негативная оценка, от 0.5 до 1 - позитивная. Изменение курса акций - число от -30% до + 30%. Сентимент тем больше, чем лучше для компании новость."

introStr = { "role": "system", "content": inst }

# Сделаем простую историю общения. Инициализируем ее системным сообщением.
history = [ introStr ]

mode1 = "ответь числами (сентимент и изменение курса) в формате: 0.45: -1.8"
mode2 = "Дай комментарий, объясняющий твои оценки новости (без указания самих оценок). Если новости всё ещё нет - так и скажи"

llm = Llama(
    model_path="C:/Meerkator/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    chat_format="llama-3",
    #use_mlock=True,
    n_gpu_layers= -1, # для использования GPU
    #n_threads= 10,
    # seed=1337, # установить конкретный seed
    n_ctx=4096, # установить размер контекста
    use_cuda = True,
)

async def SayToAI(mess_text, history1, mode = ""):
    # добавляем в историю сообщение пользователя
    history1.append({"role": "user", "content": mode + mess_text})
        # получаем ответ от модели
    out = llm.create_chat_completion(history1[-8:])
        # из объекта получаем текст сообщения
    reply = out["choices"][0]["message"]["content"]
        # сохраняем ответ ИИ в историю
    history1.append({"role": "assistant", "content": reply})
        
    history1 = history1[-8:]
    history1[0] = introStr
    
    return reply

def sameSent(x, y):
    x = (x * 2) - 1
    y = (y * 2) - 1
    if (x*y > 0):
        return True
    else:
        return False

def msquare(x,y):
    sub = x - y
    return math.sqrt((sub * sub) / 2.0)

@app.get("/")
def read_root():
    return {"Hello": "World"}

from urllib.parse import unquote_plus


@app.get("/str/{string}")
async def read_item(string: str):
    string = unquote_plus(string)
    print(string)
    #print(string2)
    res_sent = 0.5
    res_perc = 0
    ai_sent = 0
    ai_perc = 0
    comment = ""
    history = [ introStr ] #обнуляем историю

    ai = await SayToAI(string, history, mode1)
    comment = await SayToAI("", history, mode2) #экономим токены. Берем текст из памяти

    print(ai)
    print(comment)
    
    com = comment.split(" ")
    string2 = string
    print(len(com))
    if (len(com) > 4): #Если комментарий слишком короткий - новость не пришла
        string2 = comment + " " + string # увеличиваем точность анализа нейронок за счет анализа комментария

    lin_sent = GetLinearSent(string2) 
    lin_sent = (lin_sent + 1) / 2.0
    n_sent = NeuralNetworks.GetNeuroSent(model_sent, navec, device, string2)
    lin_perc = GetLinearPerc(string2)
    n_perc = NeuralNetworks.GetNeuroPerc(model_perc, n_sent, navec, device, string2)
    
    if (len(com) < 5):
        lin_sent = 0.5
        lin_perc = 0.0
        n_sent = 0.5
        n_perc = 0.0
        return {"status": "OK",
            "lin_sent": lin_sent,
            "lin_perc": lin_perc,
            "neuro_sent": n_sent,
            "neuro_perc": n_perc,
            "ai_sent": ai_sent,
            "ai_perc": ai_perc,
            "res_sent": res_sent,
            "res_perc": res_perc,
            "comment": comment}

    try:
        ai2 = []
        if ':' in ai:
            ai2 = ai.split(':')
        elif ';' in ai:
            ai2 = ai.split(';')
        ai_sent = float(ai2[0])
        ai_perc = float(ai2[1].replace("%", ""))
        ai_sent_old = ai_sent

        #Блок коррекции ошибок Лламы
        if (ai_sent < 0.5 and ai_perc > 0) or (ai_sent >= 0.5 and ai_perc < 0): #Лама косячит с сентиментом
            if (ai_perc * n_perc) > 0:
                if (ai_perc * lin_perc) > 0:
                    ai_sent = 1.0 - ai_sent
                    #ms_ainn = msquare(ai_perc, n_perc)
                    #ms_ailin = msquare(ai_perc, lin_perc)
                    #if (ms_ainn < ms_ailin):
                        #ai_sent = n_sent
                    #else:
                        #ai_sent = lin_sent
                else:
                    #ai_sent = n_sent
                    ai_sent = 1.0 - ai_sent
            else: 
                if (ai_perc * lin_perc) > 0:
                    ai_sent = 1.0 - ai_sent
                else:#лама косячит с процентом?
                    ai_perc = -ai_perc
            
      

        if sameSent(ai_sent, lin_sent) > 0:
            if sameSent(ai_sent, n_sent) > 0:
                res_sent = (ai_sent + n_sent) / 2.0 # улучшение точности нейронной сети предсказанием ламы
            else:
                res_sent = ai_sent # замена результата нейросети на более точный (случай промаха нейронки)
        else:
            if sameSent(ai_sent, n_sent) > 0: # промах линейного алгоритма
                res_sent = (n_sent + ai_sent) / 2.0
            else:
                res_sent = (lin_sent + n_sent) / 2.0 #промах ламы??
                if (ai_perc > 0 and res_sent < 0.5): #промах всех, кроме ламы!
                    res_sent = (res_sent + ai_sent) / 2.0
                    flag_LlamaCorrectCase = True

        res_perc = NeuralNetworks.GetNeuroPerc(model_perc, res_sent, navec, device, string)
            
        if (ai_perc * res_perc) > 0:
            res_perc = (ai_perc + res_perc) / 2.0
        ai_sent = ai_sent_old

    except Exception as err:
        print("ERROR: Не удалось распарсить ответ лламы!")
        print(ai)
        res_sent = n_sent 
        res_perc = n_perc

    return {"status": "OK",
            "lin_sent": lin_sent,
            "lin_perc": lin_perc,
            "neuro_sent": n_sent,
            "neuro_perc": n_perc,
            "ai_sent": ai_sent,
            "ai_perc": ai_perc,
            "res_sent": res_sent,
            "res_perc": res_perc,
            "comment": comment}

@app.get("/ai/str/{string}")
async def AI_handler(string: str):
    print(string)
    result = await SayToAI(string, history, mode2)
    return {"result":result}

@app.get("/hello/{string}")
async def AI_handler2(string: str):
    result = "world"
    return {"result":result}

#def GetRes(strn):
#    print(strn);
#    return str(model.predict(vectorizer.transform([strn])))


def main(inp):
    inp = argv[1]
    #GetRes(inp)
    
#uvicorn.run('fastapi_server', port=8000, log_level='info')
    
#main('')

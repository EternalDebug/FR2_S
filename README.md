# FR2_S
Financial Research 2. Server part. Финансовое исследование 2. Серверная часть.
Данное приложение является частью ВКР по теме "Разработка системы оценки влияния новостей на показатели курса акций с использованием различных методов анализа тональности текстов".

Интерфейс системы: https://github.com/EternalDebug/FR2_C

Для функционирования использованы наработки из предыдущего финансового исследования, а также следующие инструменты и наборы данных:
1. Датасет FiNes: https://github.com/WebOfRussia/financial-news-sentiment
2. Датасет linis-crowd за 2015: https://linis-crowd.org
3. Компактный русскоязычный эмбеддинг navec: https://natasha.github.io/navec/
4. Большая генеративная модель Llama-3.1-8b: https://hf.qhduan.com/pek111/Meta-Llama-3.1-8B-Instruct-GGUF

Вскоре выложу код, задействованный для обучения нейронных сетей

Команда для запуска серверной части в локальной сети: uvicorn D2:app --reload

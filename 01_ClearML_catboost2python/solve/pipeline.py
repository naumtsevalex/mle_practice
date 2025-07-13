from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from catboost import CatBoostClassifier
from clearml import Logger, Task

from dotenv import load_dotenv
load_dotenv(dotenv_path="./../secrets.env")

from utils import check_clearml_env, seed_everything
check_clearml_env() 


from dataclasses import asdict, dataclass
@dataclass
class CFG:
    project_name: str = "01_catboot2python"
    experiment_name: str = "jupyter_prod"

    data_path: str = "../data"
    train_name: str = "quickstart_train.csv"
    seed: int = 2024


cfg = CFG()
cfg_dict = asdict(cfg)
seed_everything(cfg.seed)


import argparse
import seaborn as sns
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100, help="Количество итераций бустинга")
    parser.add_argument("--verbose", type=int, default=0, help="Период вывода CatBoost (0 = выключено)")
    return parser.parse_args()

def run_eda(data: pd.DataFrame):
    logger = Logger.current_logger()

    # Лог классов
    class_counts = data["target_class"].value_counts()
    logger.report_table("Class balance", "EDA", table_plot=class_counts.to_frame())

    # График распределения по целям
    fig, ax = plt.subplots()
    sns.countplot(data=data, x="target_class", ax=ax)
    ax.set_title("Class Distribution")
    logger.report_matplotlib_figure(title="Target distribution", series="EDA", figure=fig, iteration=0)
    plt.close(fig)

    # Пример распределения числового признака
    num_cols = [col for col in data.columns if data[col].dtype != "object"]
    for col in num_cols[:2]:  # только первые 2 для примера
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
        logger.report_matplotlib_figure(title=f"Feature: {col}", series="EDA", figure=fig, iteration=0)
        plt.close(fig)

def preprocess_data() -> pd.DataFrame:
    url = "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/quickstart_train.csv"
    rides_info = pd.read_csv(url)
    raw_rides_info = rides_info.copy()


    cat_features = ["model", "car_type", "fuel_type"]  # Выделяем категориальные признаки
    targets = ["target_class", "target_reg"]
    features2drop = ["car_id"]  # эти фичи будут удалены

    # Отбираем итоговый набор признаков для использования моделью
    filtered_features = [
        i for i in rides_info.columns if (i not in targets and i not in features2drop)
    ]
    num_features = [i for i in filtered_features if i not in cat_features]

    print("cat_features", cat_features)
    print("num_features", len(num_features))
    print("targets", targets)

    for c in cat_features:  # Избавлеямся от NaN'ов
        rides_info[c] = rides_info[c].astype(str)

    return raw_rides_info, rides_info, filtered_features, cat_features



def main(
        iterations: int = 200,
        verbose: int = 50,
):
    logger = Logger.current_logger()
    raw_rides_info, rides_info, filtered_features, cat_features = preprocess_data()
    run_eda(rides_info)

    logger.report_table(
        title="Start val data",  # Название таблицы или метрика, получаемая на этих данных :)
        series="datasets",  # В каком разделе будут сохранены данные
        table_plot=raw_rides_info.head(),  # DataFrame
    )

    train, test = train_test_split(rides_info, test_size=0.2, random_state=cfg.seed)
    logger.report_table(
        title="Start val data",  # Название таблицы или метрика, получаемая на этих данных :)
        series="datasets",  # В каком разделе будут сохранены данные
        table_plot=test,  # DataFrame
    )

    X_train = train[filtered_features]
    y_train = train["target_class"]

    X_test = test[filtered_features]
    y_test = test["target_class"]


    cb_params = {
        "depth": 4,
        "learning_rate": 0.06,
        "loss_function": "MultiClass",
        "custom_metric": ["Recall"],
        # Главная фишка катбуста - работа с категориальными признаками
        "cat_features": cat_features,
        # Регуляризация и ускорение
        "colsample_bylevel": 0.098,
        "subsample": 0.95,
        "l2_leaf_reg": 9,
        "min_data_in_leaf": 243,
        "max_bin": 187,
        "random_strength": 1,
        # Параметры ускорения
        "task_type": "CPU",
        "thread_count": -1,
        "bootstrap_type": "Bernoulli",
        # Важное!
        "random_seed": cfg.seed,
        "early_stopping_rounds": 50,
        "iterations": iterations,
        "verbose": verbose,
    }
    task.connect(cb_params, name="CatBoost params")



    model = CatBoostClassifier(**cb_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # сохраняем модель
    model.save_model("cb_model.cbm")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.report_scalar(
        title="Accuracy",
        series="Accuracy",
        value=accuracy,
        iteration=0
    )
    cls_report = classification_report(
        y_test, y_pred, target_names=y_test.unique(), output_dict=True
    )

    cls_report = pd.DataFrame(cls_report).T
    logger.report_table(
        title="Classification report",
        series="Classification report",
        table_plot=cls_report,
        iteration=0
    )

    # Не забываем завершить таск
    task.close()





if __name__ == "__main__":
    args = parse_args()


    task = Task.init(project_name=cfg.project_name, task_name=cfg.experiment_name)
    task.add_tags(["CB_classifier"])  # Добавьте тэги обучения
    # Добавить конфиг запуска
    task.connect(
        cfg_dict,
        "Basic Config"
    )

    main(
        iterations=args.iterations,
        verbose=args.verbose
    )

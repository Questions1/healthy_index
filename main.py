
import pandas as pd
import numpy as np
import time
import functools
from datetime import timedelta
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt


def time_pass(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time_begin = time.time()
        the_result = func(*args, **kw)
        time_stop = time.time()
        time_passed = time_stop - time_begin
        minutes, seconds = divmod(time_passed, 60)
        hours, minutes = divmod(minutes, 60)
        print('%s: %s:%s:%s' % (func.__name__, int(hours), int(minutes), int(seconds)))
        return the_result

    return wrapper


class Standardize(object):
    """
    自己写的一个标准化器
    """
    def __init__(self):
        self._the_mean = pd.Series()
        self._the_sigma = pd.Series()

    @time_pass
    def fit(self, data):
        the_mean = data.drop('UNIQUENO', axis=1).apply(np.mean)
        the_sigma = data.drop('UNIQUENO', axis=1).apply(lambda x: np.sqrt(np.var(x)))

        self._the_mean = the_mean
        self._the_sigma = the_sigma + 0.00001

    @time_pass
    def transform(self, data):
        columns = data.drop('UNIQUENO', axis=1).columns
        the_result = data.copy()

        the_result[columns] = (the_result[columns] - self._the_mean)/self._the_sigma

        return the_result


@time_pass
def get_train_test(data_x, data_y, train_raito):
    """
    把数据分成训练集和测试集，因为是按时间来的，所以不能随机进行分配
    """
    bound = int(data_x.shape[0] * train_raito)
    return data_x[:bound], data_x[bound:], data_y[:bound], data_y[bound:]


@time_pass
def get_qualified(data):
    """
    首先根据数据的缺失率排除掉一些缺失率非常高的特征
    然后用自己写的"归一化"类进行归一化，并保存归一化得到信息，以备在测试集上transform
    """
    na_ratio = data.apply(lambda x: sum(pd.isnull(x))) / data.shape[0]
    sorted_na_ratio = na_ratio.sort_values(ascending=False)
    qualified_columns = sorted_na_ratio[sorted_na_ratio < 0.9].index

    qualified_train_x = data[qualified_columns]

    standardize_scaler = Standardize()  # 用我自己定义的类来进行归一化
    standardize_scaler.fit(qualified_train_x)
    the_scaled_train_x = standardize_scaler.transform(qualified_train_x)

    return the_scaled_train_x, standardize_scaler


@time_pass
def get_wave(feature, spread=100):
    """
    得到每个特征的“波动”情况，为之后的权重计算准备
    """
    feature_data = scaled_train_x[['UNIQUENO', feature]]

    def deal_id(the_id):
        """
        先选出某一个id，然后计算这个feature在这个id上的平均“波动”
        """
        id_data = feature_data[feature_data['UNIQUENO'] == the_id]
        id_label = train_y[feature_data['UNIQUENO'] == the_id]
        if all(id_data[feature].isnull()):
            return 0

        is_change = id_label['FLAG'].diff()  # 通过差分来找波动点
        is_change.iloc[0] = 0

        change_index = pd.to_datetime(is_change[is_change != 0].index)

        time_series = pd.Series(pd.to_datetime(id_data.index))
        begin_time = min(time_series)
        end_time = max(time_series)

        def cal_wave(i, p_1_element):
            """
            具体来计算每一个波动点位置的波动情况
            """
            current_index = time_series[time_series == p_1_element].index[0]

            ahead = p_1_element - timedelta(minutes=spread)
            lag = p_1_element + timedelta(minutes=spread)
            if i == 0:
                previous = begin_time
            else:
                previous = change_index[i - 1]
            if i == (len(change_index)-1):
                later = end_time
            else:
                later = change_index[i + 1]

            kaishi = max(ahead, previous, begin_time)
            jieshu = min(lag, later, end_time)

            kaishi = max(time_series[time_series <= kaishi])  # 这是波动点的前置位置
            jieshu = min(time_series[time_series >= jieshu])  # 这是波动点的后置位置

            kaishi_index = time_series[time_series == kaishi].index[0]
            jieshu_index = time_series[time_series == jieshu].index[0]

            left_mean = np.mean(id_data[feature].iloc[kaishi_index:current_index])
            right_mean = np.mean(id_data[feature].iloc[current_index:jieshu_index])

            return abs(left_mean - right_mean)

        if len(change_index) == 0:  # 如何没有发生波动，则直接返回0
            p_mean = 0
        else:
            p_mean_list = []  # 因为可能会有多个波动点，所以把每一个波动幅度存起来
            for the_i, the_p_1_element in enumerate(change_index):
                p_mean_element = cal_wave(the_i, the_p_1_element)
                p_mean_list.append(p_mean_element)
            p_mean = np.mean(pd.Series(p_mean_list))  # 某一个feature下某一个id的平均波动

        return p_mean

    all_id_means = list(map(deal_id, all_id))
    all_mean = np.mean(pd.Series(all_id_means))  # 某一个feature下所有id的平均波动

    return all_mean


@time_pass
def soft_max(iterable):
    """
    通过softmax函数得到权重
    """
    e_exponents = [np.e ** x for x in iterable]
    sum_e = np.sum(pd.Series(e_exponents))
    the_weight = [x/sum_e for x in e_exponents]

    return the_weight


@time_pass
def partial_sign():
    """
    计算偏相关系数的符号，确保计算综合指标的时候往同一个方向使劲
    """
    true_x = scaled_train_x.drop('UNIQUENO', axis=1).fillna(0)
    true_x['beta_0'] = 1
    true_y = train_y['FLAG']

    clf = linear_model.LogisticRegression()
    clf.fit(true_x, true_y)
    partial_corr = clf.coef_[0][:(scaled_train_x.shape[1] - 1)]
    the_partial_corr_sign = np.sign(partial_corr)

    return the_partial_corr_sign


def get_win_ratio():
    """
    计算每一个设备分数比多少设备更加健康
    """
    x_train = scaled_train_x.drop('UNIQUENO', axis=1).fillna(0)
    the_weight = np.array(feature_wave_df['weight']).reshape(-1, 1)
    weighted_score_train = np.dot(x_train, the_weight)

    percentile_1000 = np.percentile(weighted_score_train, np.linspace(0, 100, 1000))
    # 只取1000分位数以简化计算
    p_1000_prepared = percentile_1000.reshape(-1, 1)

    multi_compare = (p_1000_prepared > weighted_score.reshape(1, -1))
    the_win_ratio = sum(multi_compare) / len(percentile_1000) * 100

    return the_win_ratio


if __name__ == '__main__':
    his_x = pd.read_csv('./input/his_x.csv', index_col=0)
    his_y = pd.read_csv('./input/his_y.csv', index_col=0)
    his_y['FLAG'][his_y['FLAG'] > 0] = 1

    train_x, test_x, train_y, test_y = get_train_test(his_x, his_y, 0.8)

    scaled_train_x, standard_scaler = get_qualified(train_x)
    all_id = scaled_train_x['UNIQUENO'].unique()  # 所有的id
    n_samples = scaled_train_x.shape[0]

    feature_wave_list = []  # 这个就是输出结果了
    features = scaled_train_x.drop('UNIQUENO', axis=1).columns  # 所有的feature
    for feature in features:
        feature_wave = get_wave(feature)
        feature_wave_list.append(feature_wave)

    weight = soft_max(feature_wave_list)  # 通过softmax函数计算权重

    partial_corr_sign = partial_sign()
    signed_weight = partial_corr_sign * weight

    feature_wave_df = pd.DataFrame({'feature': features,
                                    'feature_wave': feature_wave_list,
                                    'weight': signed_weight})
    feature_wave_df.to_csv('./output/feature_wave_df.csv', index=False)
    # 目前的输出：scaled_train_x, min_max，weight

    # 下面根据输出在测试集上进行预测
    feature_wave_df = pd.read_csv('./output/feature_wave_df.csv')
    scaled_test_x = standard_scaler.transform(test_x[scaled_train_x.columns])  # 归一化操作
    X = scaled_test_x.drop('UNIQUENO', axis=1).fillna(0)  # 填补为0之后不影响内积的计算
    weighted_score = np.dot(X, np.array(feature_wave_df['weight']).reshape(-1, 1))  # 计算加权和

    print(metrics.roc_auc_score(test_y['FLAG'], weighted_score))  # 0.8087
    # 可调参数：前后100分钟，将维在扩充label时的范围，倾向值的匹配

    # 计算一个0-100的得分，并输出
    win_ratio = get_win_ratio()
    result = pd.DataFrame({'id': scaled_test_x['UNIQUENO'],
                           'score': win_ratio})
    result.to_csv('./output/score.csv', index=False)
    print(metrics.roc_auc_score(test_y['FLAG'], -result['score']))

    # ------------------------------------------------------------
    # 下面都是为了做PPT
    roc_auc = metrics.roc_curve(test_y['FLAG'], -result['score']/100)
    plt.plot(roc_auc[0], roc_auc[1], linewidth=5, color='blue')
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100),
             linestyle='dashed', color='red', linewidth=5)
    plt.scatter([0, 1], [0, 1], s=100, color='red')
    plt.title('ROC-Curve', fontsize=30)
    plt.xlabel('FPR', fontsize=30)
    plt.ylabel('TPR', fontsize=30)
    plt.text(0.45, 0.3, 'AUC: 0.705', fontsize=30)
    # 上面这段画ROC曲线的倒是可以借鉴一下
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # 下面是对全部数据进行计算score
    for_plot, min_max = get_qualified(his_x)
    for_plot.fillna(0, inplace=True)
    id_all = his_x['UNIQUENO']
    time_occured = his_x.index
    all_score = np.dot(for_plot.drop('UNIQUENO', axis=1), np.array(feature_wave_df['weight']).reshape(-1, 1))

    all_score = -all_score
    x_min = min(all_score)
    x_max = max(all_score)
    all_score = (all_score - x_min)/(x_max - x_min) * 100

    # 把ID，时间，得分放到一个dataframe里边，方面找子集
    all_df = pd.DataFrame({'id': id_all,
                           'time': pd.to_datetime(time_occured),
                           'score': all_score.reshape(1, -1)[0]})

    selected = all_df[(all_df['id'] == 201707201730009538)
                      & (all_df['time'] > pd.to_datetime('2017-03-01'))
                      & (all_df['time'] < pd.to_datetime('2018-10-12'))]
    selected.index = selected['time']
    plt.plot(selected['score'])
    plt.title('ID: 201707201730009538')
    # ------------------------------------------------------------
    # ------------------------------------------------------------

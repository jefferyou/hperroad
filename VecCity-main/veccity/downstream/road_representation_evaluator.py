import csv
import importlib
import json
import numpy as np
import pandas as pd
import os
from logging import getLogger
from veccity.downstream.abstract_evaluator import AbstractEvaluator
from veccity.downstream.embedding_wrapper import EmbeddingWrapper
from veccity.downstream.utils import generate_road_representaion_downstream_data


class RoadRepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config = config
        self.evaluate_tasks = self.config.get('evaluate_task', ["speed_inference", "travel_time_estimation", "similarity_search"])
        self.evaluate_model = self.config.get('evaluate_model', ["SpeedInferenceModel", "TravelTimeEstimationModel", "SimilaritySearchModel"])
        self.result = {}
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = config.get('geo_file', self.dataset)
        self.output_dim = config.get('output_dim', 32)
        self.data_feature = data_feature
        self.embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.result_path = './veccity/cache/{}/evaluate_cache/result_{}_{}_{}.json' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)

        # 加载标签数据（与HHGCLEvaluator类似）
        self.label_data_path = os.path.join('veccity', 'cache', 'dataset_cache', self.dataset, 'label_data')
        self._load_label_data()

    def _load_label_data(self):
        """加载下游任务的标签数据"""
        # 确保标签数据文件存在
        data_path1 = os.path.join("veccity/cache/dataset_cache", self.dataset, "label_data", "avg_speeds.csv")
        data_path2 = os.path.join("veccity/cache/dataset_cache", self.dataset, "label_data", "time.csv")

        if not os.path.exists(data_path1) or not os.path.exists(data_path2):
            self._logger.info(f"Label data not found, generating...")
            generate_road_representaion_downstream_data(self.dataset)

        # 加载speed inference标签
        speed_label = pd.read_csv(os.path.join(self.label_data_path, "avg_speeds.csv"))
        speed_label.sort_values(by="index", inplace=True, ascending=True)

        # 加载travel time estimation标签
        min_len, max_len = self.config.get("tte_min_len", 1), self.config.get("tte_max_len", 100)
        time_label = pd.read_csv(os.path.join(self.label_data_path, "time.csv"))
        time_label['path'] = time_label['trajs'].map(eval)
        time_label['path_len'] = time_label['path'].map(len)
        time_label = time_label.loc[
            (time_label['path_len'] > min_len) & (time_label['path_len'] < max_len)
        ]

        # 获取节点数量
        num_nodes = self.data_feature.get('num_nodes', len(speed_label))

        # 构建与HHGCLEvaluator兼容的标签结构
        if "label" not in self.data_feature:
            self.data_feature["label"] = {}

        self.data_feature["label"]["speed_inference"] = {
            'speed': speed_label
        }
        self.data_feature["label"]["travel_time_estimation"] = {
            'time': time_label,
            'padding_id': num_nodes
        }

        self._logger.info(f"Label data loaded: speed_inference={len(speed_label)}, travel_time_estimation={len(time_label)}")

    def get_downstream_model(self, model):
        try:
            return getattr(importlib.import_module('veccity.downstream.downstream_models'), model)(self.config)
        except AttributeError:
            raise AttributeError('evaluate model is not found')

    def collect(self, batch):
        pass

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_uid, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_uids = list(geofile['geo_uid'])
        self.num_nodes = len(self.geo_uids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_uids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_uids)))
        return geofile

    def evaluate(self):
        def add_prefix_to_keys(dictionary, prefix):
            new_dictionary = {}
            for key, value in dictionary.items():
                new_key = prefix + str(key)
                new_dictionary[new_key] = value
            return new_dictionary
        
        def dict_to_csv(dictionary, filename):
            # Ensure directory exists before writing file
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
                writer.writeheader()
                writer.writerow(dictionary)

        road_emb = np.load(self.embedding_path)  # (N, F)

        # 获取设备配置
        device = self.config.get('device', 'cpu')

        # 包装embeddings以提供encode接口，避免下游任务重新运行模型
        embedding_wrapper = EmbeddingWrapper(road_emb, device=device)

        for task, model in zip(self.evaluate_tasks, self.evaluate_model):
            downstream_model = self.get_downstream_model(model)
            x = embedding_wrapper  # 使用wrapper而不是原始numpy数组

            # SimilaritySearchModel不需要label参数，它自己处理数据加载
            if model == "SimilaritySearchModel":
                self._logger.info(f"Running {task} (no label required)...")
                result = downstream_model.run(x)
            else:
                # SpeedInferenceModel和TravelTimeEstimationModel需要label
                label = self.data_feature["label"][task]
                result = downstream_model.run(x, label)

            self.result.update(add_prefix_to_keys(result, task + '_'))

        # 移除不需要的best epoch键（如果存在）
        if 'travel_time_estimation_best epoch' in self.result:
            del self.result['travel_time_estimation_best epoch']
        print(f'Evaluate result: {self.result}')
        self._logger.info(f'Evaluate result: {self.result}')
        result_path = './raw_data/new/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, self.output_dim)
        # result_path = './veccity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
        #     format(self.exp_id, self.exp_id, self.model, self.dataset, self.output_dim)
        dict_to_csv(self.result, result_path)
        self._logger.info('Evaluate result is saved at {}'.format(result_path))
        return

        # !这个load_geo必须跟dataset部分相同，也就是得到同样的geo_uid和index的映射，否则就会乱码
        # TODO: 把dataset部分得到的geo_to_ind和ind_to_geo传过来
        rid_file = self._load_geo()
        # 记录每个类别都有哪些geo实体
        result_token = dict()
        for i in range(len(y_predict)):
            kind = int(y_predict[i])
            if kind not in result_token:
                result_token[kind] = []
            result_token[kind].append(self.ind_to_geo[i])
        result_path = './veccity/cache/{}/evaluate_cache/kmeans_category_{}_{}_{}_{}.json'. \
            format(self.exp_id, self.model, self.dataset, str(self.output_dim), str(kinds))
        json.dump(result_token, open(result_path, 'w'))
        self._logger.info('Kmeans category is saved at {}'.format(result_path))

        # QGIS可视化
        rid_type = rid_file['type'][0]
        rid_pos = rid_file['geo_location']
        rid2wkt = dict()
        if rid_type == 'LineString':
            for i in range(rid_pos.shape[0]):
                rid_list = eval(rid_pos[i])  # [(lat1, lon1), (lat2, lon2)...]
                wkt_str = 'LINESTRING('
                for j in range(len(rid_list)):
                    rid = rid_list[j]
                    wkt_str += (str(rid[0]) + ' ' + str(rid[1]))
                    if j != len(rid_list) - 1:
                        wkt_str += ','
                wkt_str += ')'
                rid2wkt[i] = wkt_str
        elif rid_type == 'Point':
            for i in range(rid_pos.shape[0]):
                rid_list = eval(rid_pos[i])  # [lat1, lon1]
                wkt_str = 'Point({} {})'.format(rid_list[0], rid_list[1])
                rid2wkt[i] = wkt_str
        else:
            raise ValueError('Error geo type!')

        df = []
        for i in range(len(y_predict)):
            df.append([i, self.ind_to_geo[i], y_predict[i], rid2wkt[i]])
        df = pd.DataFrame(df)
        df.columns = ['id', 'rid', 'class', 'wkt']
        df = df.sort_values(by='class')
        result_path = './veccity/cache/{}/evaluate_cache/kmeans_qgis_{}_{}_{}_{}.csv'. \
            format(self.exp_id, self.model, self.dataset, str(self.output_dim), str(kinds))
        df.to_csv(result_path, index=False)
        self._logger.info('Kmeans result for QGIS is saved at {}'.format(result_path))

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass

import os
import json
import torch
import random
from veccity.config import ConfigParser
from veccity.data import get_dataset
from veccity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)

    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))

    logger.info(config.config)
    # seed
    seed = config.get('seed', 31)
    set_random_seed(seed)
    model_cache_file = './veccity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)

    # === 检测embeddings是否已存在 ===
    representation_object = config.get('representation_object', 'region')
    output_dim = config.get('output_dim', 128)
    embed_size = config.get('embed_size', 128)

    if representation_object == 'road':
        embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'.format(
            exp_id, model_name, dataset_name, embed_size)
    else:
        embedding_path = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'.format(
            exp_id, model_name, dataset_name, output_dim)

    # 如果embeddings已存在且用户允许跳过训练
    skip_training = config.get('skip_if_embeddings_exist', False)
    embeddings_exist = os.path.exists(embedding_path)

    if embeddings_exist and skip_training:
        logger.info(f'Embeddings already exist at {embedding_path}, skipping training')
        train = False
    # === END 检测embeddings ===

    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    if train or not os.path.exists(model_cache_file):
        train_data, valid_data, test_data = dataset.get_data()
    else:
        test_data=None
    data_feature = dataset.get_data_feature()
    # 加载执行器

    model = get_model(config, data_feature)
    # model=None
    total_num = sum([param.nelement() for param in model.parameters()])
    logger.info('Number of model parameters: {}'.format(total_num))
    executor = get_executor(config, model, data_feature)
    # 训练
    if train or not os.path.exists(model_cache_file):
        if embeddings_exist and skip_training:
            logger.info('Skipping training phase, will only run downstream tasks')
        else:
            executor.train(train_data, valid_data)
            if saved_model:
                executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)

    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)
    abl=config.get('abl')
    logger.info(f'ablation is {abl}')


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, hyper_config_dict)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    best_valid_score = executor.train(train_data, valid_data)
    test_result = executor.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }

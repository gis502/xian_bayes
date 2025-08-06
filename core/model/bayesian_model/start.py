import threading

from core.model.bayesian_model.bayesian_network_model import BayesianNetworkModel

# 全局变量和同步工具
model = None  # 存储加载的模型
training_lock = threading.Lock()  # 训练锁，防止并发训练
retrain_event = threading.Event()  # 重训练事件触发器
bayesianNetworkModel = BayesianNetworkModel()


class ModelTrainer:
    @staticmethod
    def init_model():
        global model
        global bayesianNetworkModel
        # 1. 加载和预处理数据
        data = bayesianNetworkModel.load_and_preprocess_data()

        # 2. 构建贝叶斯网络
        model = bayesianNetworkModel.build_bayesian_network(data)

    @staticmethod
    def train_model():
        global model
        global bayesianNetworkModel

        # 一直进行模型训练
        while True:
            # 等待触发信号
            retrain_event.wait()

            # 获取锁并重置事件
            with training_lock:
                retrain_event.clear()

                try:
                    print('正在重构贝叶斯网络模型......')
                    ModelTrainer.init_model()
                    print('贝叶斯网络模型重构完成')
                except Exception as e:
                    print(f"训练出错: {str(e)}")

    # 启动新的线程来训练模型
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()

    # 初始化模型
    print('正在对贝叶斯网络模型进行第一次训练，请稍后......')
    init_model()
    print('贝叶斯网络模型训练完成')

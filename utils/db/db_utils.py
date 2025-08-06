import yaml
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from utils.file.file_utils import FileUtils

# 创建ORM模型的基础类，所有数据模型类都将继承这个Base类
Base = declarative_base()


class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 加载配置文件
        config_path = FileUtils.path_convert(os.path.join(FileUtils.get_project_root_path(), 'utils/db/db_config.yaml'))
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化连接池
        self._init_engine()

        # 创建会话工厂
        self.Session = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False
        )

        self._initialized = True

    def _init_engine(self):
        """初始化数据库引擎，配置连接池参数"""
        db_config = self.config['database']

        # 构建数据库连接字符串
        connection_string = f"{db_config['driver']}://{db_config['username']}:{db_config['password']}@" \
                            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"

        # 连接池配置参数
        pool_config = self.config.get('pool', {})
        pool_size = pool_config.get('size', 5)  # 连接池大小
        max_overflow = pool_config.get('max_overflow', 10)  # 最大溢出连接数
        pool_recycle = pool_config.get('recycle', 3600)  # 连接回收时间(秒)
        pool_pre_ping = pool_config.get('pre_ping', True)  # 连接前检测
        pool_timeout = pool_config.get('timeout', 30)  # 获取连接超时时间(秒)

        # 创建引擎并配置连接池
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
            pool_timeout=pool_timeout,
            echo=db_config.get('echo_sql', False)  # 是否打印SQL语句
        )

    def create_all_tables(self):
        """创建所有数据表"""
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """获取一个新的会话"""
        return self.Session()

    @contextmanager
    def session_scope(self):
        """会话上下文管理器，自动处理提交和回滚"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


# 实例化数据库工具类
db = Database()


# 快捷获取会话上下文管理器
@contextmanager
def get_db():
    """获取数据库会话的上下文管理器"""
    with db.session_scope() as session:
        yield session

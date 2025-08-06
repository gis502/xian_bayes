import os


class FileUtils:
    """
        文件相关工具类
    """

    @staticmethod
    def get_project_root_path():
        """
            获取项目根目录
            :return: 项目根目录
        """
        current_file_path = os.path.abspath(__file__)

        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        return project_root

    @staticmethod
    def path_convert(path):
        """
        将路径中\\替换成/
        :param path: 路径
        :return: 替换后的路径
        """
        return str.replace(path, "\\", "/")


import os.path
import sys
import subprocess
import platform

from pathlib import Path

from utils.file.file_utils import FileUtils

REQUIRED_VERSION = (3, 10)  # 指定所需的最低版本


def check_python_version():
    """检查Python版本是否符合要求"""
    current_version = sys.version_info[:2]  # 获取主次版本 (3, 8)

    if current_version < REQUIRED_VERSION:
        print(f"\n⚠️ 当前Python版本: {platform.python_version()}")
        print(f"⚠️ 项目需要Python {'.'.join(map(str, REQUIRED_VERSION))} 或更高版本")
        sys.exit(1)  # 退出程序


def ensure_virtualenv(venv_name=".venv"):
    """
    确保虚拟环境存在，不存在则创建
    :param venv_name: 虚拟环境的名称
    :return:
    """
    venv_path = Path(venv_name)
    if not venv_path.exists():
        print(f"创建虚拟环境: {venv_name}")
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
    return venv_path


def get_venv_python(venv_path):
    """
    获取虚拟环境中的Python解释器路径

    :param venv_path: 虚拟环境路径
    :return:
    """
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_dependencies(venv_python, requirements="requirements.txt"):
    """
    使用虚拟环境的pip安装依赖

    :param venv_python: 虚拟环境
    :param requirements: 依赖文件
    :return: 安装结果
    """
    if not Path(requirements).exists():
        print(f"没有找到依赖文件:{requirements}")
        return False

    print("正在安装相关依赖......")
    try:
        subprocess.check_call([
            str(venv_python),
            "-m", "pip", "install",
            "--disable-pip-version-check",
            "-r", requirements
        ])
        return True
    except subprocess.CalledProcessError:
        print("安装失败")
        return False


def main():
    # 检查python版本是否符合要求
    check_python_version()

    # 确保虚拟环境存在
    venv_path = ensure_virtualenv()
    venv_python = get_venv_python(venv_path)

    # 主程序入口
    program_entrance = FileUtils.path_convert(os.path.join(FileUtils.get_project_root_path(), 'core/web/web_server.py'))

    # 2. 检查并安装依赖
    if install_dependencies(venv_python):
        # 3. 使用虚拟环境启动主程序
        print("开始执行......")
        subprocess.check_call([
            str(venv_python),
            program_entrance  # 主程序入口
        ])
    else:
        print("程序因为依赖安装失败终止执行")


if __name__ == "__main__":
    main()

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from yaml import safe_load

from utils.file.file_utils import FileUtils

# 导入路由模块
from core.web.apis import bayesian_network

app = FastAPI(title="西安项目Python服务器")

# 加载路由
bayesian_network.register_routes(app)

if __name__ == "__main__":
    config_path = FileUtils.path_convert(os.path.join(FileUtils.get_project_root_path(),
                                                      'config/web_config.yaml'))
    with open(config_path, 'r', encoding='utf-8') as f:
        config = safe_load(f)

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config['web']['allow_origin'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(app, host=config['web']['host'], port=config['web']['port'], log_level='info')
